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
import torch.cuda.amp as amp

import torchvision
from torchvision import transforms as tf

from models import build_model
from utils import distributed_utils
from utils.misc import ModelEMA, accuracy
from utils.com_flops_params import FLOPs_and_Params


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--path_to_save', type=str, 
                        default='weights/')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
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
    parser.add_argument('-opt', '--optimizer', type=str, default='adamw',
                        help='sgd, adam')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('-mn', '--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--grad_accumulate', type=int, default=1,
                        help='gradient grad_accumulate')
    parser.add_argument('--base_lr', type=float,
                        default=1e-3, help='learning rate for training model')
    parser.add_argument('--min_lr', type=float,
                        default=1e-6, help='the final lr')
    # Model
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='resnet18, resnet34, ...')
    parser.add_argument('-p', '--pretrained', action='store_true', default=False,
                        help='use imagenet pretrained weight.')
    parser.add_argument('--norm_type', type=str, default='BN',
                        help='normalization layer.')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='number of classes')
    # Dataset
    parser.add_argument('-root', '--data_path', type=str, default='/mnt/share/ssd2/dataset',
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
    
    # ------------------------- Build DDP environment -------------------------
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # ------------------------- Build CUDA -------------------------
    if args.cuda:
        print("use cuda")
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ------------------------- Build Tensorboard -------------------------
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)

    # ------------------------- Build Dataset -------------------------
    ## train dataset
    train_dataset = torchvision.datasets.ImageFolder(
                        root=os.path.join(args.data_path, 'train'),
                        transform=tf.Compose([
                            tf.RandomResizedCrop(224),
                            tf.RandomHorizontalFlip(args.hflip),
                            tf.ToTensor(),
                            tf.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])]))
    train_loader = build_dataloader(args, train_dataset)
    ## val dataset
    val_dataset = torchvision.datasets.ImageFolder(
                        root=os.path.join(args.data_path, 'val'), 
                        transform=tf.Compose([
                            tf.Resize(256),
                            tf.CenterCrop(224),
                            tf.ToTensor(),
                            tf.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])]))
    val_loader = torch.utils.data.DataLoader(
                        dataset=val_dataset,
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=args.num_workers, 
                        pin_memory=True)
    print('=================== Dataset Information ===================')
    print('Train data length : ', len(train_dataset))
    print('Val data length : ', len(val_dataset))

    # ------------------------- Build Model -------------------------
    ## build model
    model = build_model(args.model, args.pretrained, args.num_classes, args.resume)
    model.train().to(device)
    print(model)
    ## compute FLOPs & Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, size=224)
        model_copy.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()
    ## EMA Model
    if args.ema:
        print('use EMA ...')
        ema = ModelEMA(model, args.start_epoch*epoch_size)
    else:
        ema = None

    # ------------------------- Build DDP Model -------------------------
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------------------------- Train Config -------------------------
    best_acc1 = -1.
    epoch_size = len(train_loader)

    # ---------------------------------- Build Optimizer ----------------------------------
    print("Optimizer: {}".format(args.optimizer))
    base_lr = args.base_lr * args.batch_size * args.grad_accumulate / 1024
    min_lr = args.min_lr * args.batch_size * args.grad_accumulate / 1024
    optimizer = optim.AdamW(model_without_ddp.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    # ------------------------- Build Lr Scheduler -------------------------
    lf = lambda x: ((1 - math.cos(x * math.pi / args.max_epoch)) / 2) * (args.min_lr / args.base_lr - 1) + 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ------------------------- Build Criterion -------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # ------------------------- Training Pipeline -------------------------
    t0 = time.time()
    print("=================== Start training ===================")
    for epoch in range(args.start_epoch, args.max_epoch):
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
            output = model(images)

            # Loss
            loss = criterion(output, target)

            # Accuracy
            acc = accuracy(output, target, topk=(1, 5,))            

            # Backward
            loss /= args.grad_accumulate
            loss.backward() 

            # Update
            if ni % args.grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            if args.ema:
                ema.update(model)

            # Logs
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('loss',  loss.item() * args.grad_accumulate,  ni)
                    tblogger.add_scalar('acc1',  acc[0].item(),  ni)
                    tblogger.add_scalar('acc5',  acc[1].item(),  ni)
                
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
                model_eval = ema.ema if args.ema else model_without_ddp
                loss, acc1 = validate(device, val_loader, model_without_ddp, criterion)
                print('Eval Results: [loss: %.2f][acc1: %.2f]' % (loss.item(), acc1[0].item()), flush=True)

                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                if is_best:
                    print('saving the model ...')
                    weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.model, epoch, acc1[0].item())
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model_eval.state_dict()}, checkpoint_path)                      
            
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


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_dataloader(args, dataset):
    # distributed
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                        args.batch_size, 
                                                        drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_sampler=batch_sampler_train,
                                             num_workers=args.num_workers,
                                             pin_memory=True)
    
    return dataloader
    


if __name__ == "__main__":
    main()