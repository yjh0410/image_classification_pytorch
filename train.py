from copy import deepcopy
import os
import time
import math
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
from torchvision import transforms as tf

from models import build_model
from utils import distributed_utils
from utils.misc import ModelEMA, accuracy
from utils.com_flops_params import FLOPs_and_Params


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--img_size', type=int,
                        default=224, help='input image size')
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--path_to_save', type=str, 
                        default='weights/')
    parser.add_argument('--fp16', action='store_true', default=False, 
                        help='enable amp training.')
    # Epoch
    parser.add_argument('--wp_epoch', type=int, default=20, 
                        help='warmup epoch')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='start epoch')
    parser.add_argument('--max_epoch', type=int, default=300, 
                        help='max epoch')
    parser.add_argument('--eval_epoch', type=int, default=1, 
                        help='max epoch')
    # Optimizer
    parser.add_argument('--grad_accumulate', type=int, default=1,
                        help='gradient grad_accumulate')
    parser.add_argument('--base_lr', type=float,
                        default=1e-3, help='learning rate for training model')
    parser.add_argument('--min_lr', type=float,
                        default=1e-6, help='the final lr')
    # Model
    parser.add_argument('-m', '--model', type=str, default='darknet19',
                        help='resnet18, resnet34, ...')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='number of classes')
    # Dataset
    parser.add_argument('-root', '--data_path', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset')
    parser.add_argument('--use_pixel_statistic', action='store_true', default=False,
                        help='path to dataset')
    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()

    
def main():
    args = parse_args()
    print(args)

    path_to_save = os.path.join(args.path_to_save, args.model)
    os.makedirs(path_to_save, exist_ok=True)
    
    # ---------------------------- Build DDP ----------------------------
    local_rank = local_process_rank = -1
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()
    world_size = distributed_utils.get_world_size()
    print("LOCAL RANK: ", local_rank)
    print("LOCAL_PROCESS_RANL: ", local_process_rank)
    print('WORLD SIZE: {}'.format(world_size))

    # ------------------------- Build CUDA -------------------------
    if torch.cuda.is_available():
        print("use cuda")
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        print("use cpu")
        device = torch.device("cpu")

    # ------------------------- Build Dataset -------------------------
    pixel_mean = [0.485, 0.456, 0.406] if args.use_pixel_statistic else [0., 0., 0.]
    pixel_std  = [0.229, 0.224, 0.225] if args.use_pixel_statistic else [1., 1., 1.]
    print("Pixel mean: {}".format(pixel_mean))
    print("Pixel std:  {}".format(pixel_std))
    ## train dataset
    train_dataset = torchvision.datasets.ImageFolder(
                        root=os.path.join(args.data_path, 'train'),
                        transform=tf.Compose([
                            tf.RandomResizedCrop(args.img_size),
                            tf.RandomHorizontalFlip(),
                            tf.ToTensor(),
                            tf.Normalize(pixel_mean, pixel_std)]))
    train_loader = build_dataloader(args, train_dataset, args.batch_size // world_size)
    epoch_size = len(train_loader)
    ## val dataset
    val_dataset = torchvision.datasets.ImageFolder(
                        root=os.path.join(args.data_path, 'val'), 
                        transform=tf.Compose([
                            tf.Resize(int(256 / 224 * args.img_size)),
                            tf.CenterCrop(args.img_size),
                            tf.ToTensor(),
                            tf.Normalize(pixel_mean, pixel_std)]))
    val_loader = torch.utils.data.DataLoader(
                        dataset=val_dataset,
                        batch_size=256, 
                        shuffle=False,
                        num_workers=args.num_workers, 
                        pin_memory=True)
    print('=================== Dataset Information ===================')
    print('Train data length : ', len(train_dataset))
    print('Val data length : ', len(val_dataset))

    # ------------------------- Build Model -------------------------
    ## build model
    model = build_model(args)
    model.train().to(device)
    print(model)
    ## compute FLOPs & Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model)
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, size=args.img_size)
        model_copy.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()
    ## EMA Model
    if args.ema:
        print('use EMA ...')
        model_ema = ModelEMA(model, args.start_epoch*epoch_size)
    else:
        model_ema = None

    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        if args.sybn:
            print('use SyncBatchNorm ...')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ---------------------------------- Build Grad Scaler ----------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ---------------------------------- Build Optimizer ----------------------------------
    args.base_lr = args.base_lr * args.batch_size * args.grad_accumulate / 1024
    args.min_lr  = args.min_lr  * args.batch_size * args.grad_accumulate / 1024
    print("Base lr: {}".format(args.base_lr))
    print("Min lr : {}".format(args.min_lr))
    optimizer = optim.AdamW(model_without_ddp.parameters(), lr=args.base_lr, weight_decay=0.05)
    start_epoch = 0
    if args.resume and args.resume != "None":
        print('keep training: ', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch") + 1
        del checkpoint, checkpoint_state_dict

    # ------------------------- Build Lr Scheduler -------------------------
    lf = lambda x: ((1 - math.cos(x * math.pi / args.max_epoch)) / 2) * (args.min_lr / args.base_lr - 1) + 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    lr_scheduler.last_epoch = start_epoch - 1  # do not move
    if args.resume and args.resume != 'None':
        lr_scheduler.step()

    # ------------------------- Build Criterion -------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # ------------------------- Training Pipeline -------------------------
    t0 = time.time()
    best_acc1 = -1.
    print("=================== Start training ===================")
    for epoch in range(start_epoch, args.max_epoch):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        for iter_i, (images, target) in enumerate(train_loader):
            ni = iter_i + epoch * epoch_size
            nw = args.wp_epoch * epoch_size
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                for x in optimizer.param_groups:
                    x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr'] * lf(epoch)])

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Inference
            with torch.cuda.amp.autocast(enabled=args.fp16):
                output = model(images)
                loss = criterion(output, target)
                loss /= args.grad_accumulate

            # Accuracy
            acc = accuracy(output, target, topk=(1, 5,))            

            # Backward
            scaler.scale(loss).backward()

            # Update
            if ni % args.grad_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Model EMA update
                if args.ema:
                    model_ema.update(model)

            # Logs
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch + 1, args.max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(cur_lr[0])
                # loss infor
                log += '[loss: {:.6f}]'.format(loss.item() * args.grad_accumulate)
                # other infor
                log += '[acc1: {:.2f}]'.format(acc[0].item())
                log += '[acc5: {:.2f}]'.format(acc[1].item())
                log += '[time: {:.2f}]'.format(t1 - t0)

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

        # evaluate
        if distributed_utils.is_main_process():
            if (epoch % args.eval_epoch) == 0 or (epoch == args.max_epoch - 1):
                print('evaluating ...')
                model_eval = model_ema.ema if args.ema else model_without_ddp
                loss, acc1 = validate(device, val_loader, model_without_ddp, criterion)
                print('Eval Results: [loss: %.2f][acc1: %.2f]' % (loss.item(), acc1[0].item()), flush=True)

                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                if is_best:
                    print('saving the model ...')
                    weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.model, epoch, acc1[0].item())
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model_eval.state_dict(),
                                'mAP': -1.,
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch,
                                }, 
                                checkpoint_path)               

        lr_scheduler.step()


def validate(device, val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    acc1_num_pos = 0.
    count = 0.
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i % 100 == 0:
                print("[%d]/[%d] ..." % (i, len(val_loader)))
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # inference
            output = model(images)

            # loss
            loss = criterion(output, target)

            # accuracy
            cur_acc1 = accuracy(output, target, topk=(1,))

            # Count the number of positive samples
            bs = images.shape[0]
            count += bs
            acc1_num_pos += cur_acc1[0] * bs
        
        # top1 acc
        acc1 = acc1_num_pos / count

    # switch to train mode
    model.train()

    return loss, acc1


def build_dataloader(args, dataset, batch_size):
    # distributed
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_sampler=batch_sampler_train,
                                             num_workers=args.num_workers,
                                             pin_memory=True)
    
    return dataloader
    


if __name__ == "__main__":
    main()