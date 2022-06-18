from copy import deepcopy
import os
import time
import math
import argparse

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
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--wp_epoch', type=int, default=20, 
                        help='warmup epoch')
    parser.add_argument('--max_epoch', type=int, default=300, 
                        help='max epoch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--base_lr', type=float,
                        default=4e-3, help='learning rate for training model')
    parser.add_argument('--min_lr', type=float,
                        default=1e-6, help='the final lr')
    parser.add_argument('--path_to_save', type=str, 
                        default='weights/')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    # optimization
    parser.add_argument('-opt', '--optimizer', type=str, default='adamw',
                        help='sgd, adam')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('-mn', '--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('-accu', '--accumulation', type=int, default=1,
                        help='gradient accumulation')

    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    # Model
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='resnet18, resnet34, ...')
    parser.add_argument('-p', '--pretrained', action='store_true', default=False,
                        help='use imagenet pretrained weight.')
    parser.add_argument('--norm_type', type=str, default='BN',
                        help='normalization layer.')

    # dataset
    parser.add_argument('-root', '--data_path', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset')

    # model config
    parser.add_argument('--num_classes', type=int, default=16,
                        help='number of classes')

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

    # dist
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))


    path_to_save = os.path.join(args.path_to_save)
    os.makedirs(path_to_save, exist_ok=True)
    
    # use gpu
    if args.cuda:
        print("use cuda")
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # EMA
    if args.ema:
        print('use EMA ...')

    # tensorboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)

    # dataset
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]
    train_data_root = os.path.join(args.data_path, 'train')
    val_data_root = os.path.join(args.data_path, 'val')
    ## train dataset
    train_dataset = torchvision.datasets.ImageFolder(
                        root=train_data_root,
                        transform=tf.Compose([
                            tf.RandomResizedCrop(224),
                            tf.RandomHorizontalFlip(),
                            tf.ToTensor(),
                            tf.Normalize(pixel_mean,
                                         pixel_std)]))
    train_loader = build_dataloader(args, train_dataset)
    ## val dataset
    val_dataset = torchvision.datasets.ImageFolder(
                        root=val_data_root, 
                        transform=tf.Compose([
                            tf.Resize(256),
                            tf.CenterCrop(224),
                            tf.ToTensor(),
                            tf.Normalize(pixel_mean,
                                        pixel_std)]))
    val_loader = torch.utils.data.DataLoader(
                        dataset=val_dataset,
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=args.num_workers, 
                        pin_memory=True)

    print('========================')
    print('Train data length : ', len(train_dataset))
    print('Val data length : ', len(val_dataset))

    # model
    model = build_model(model_name=args.model, 
                        pretrained=args.pretrained, 
                        norm_type=args.norm_type,
                        num_classes=args.num_classes)
    model.train().to(device)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    ema = ModelEMA(model) if args.ema else None

    # FLOPs * Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.eval()
        FLOPs_and_Params(model=model, size=224)
        model_copy.train()

    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()


    # basic config
    best_acc1 = -1.
    base_lr = args.base_lr
    min_lr = args.min_lr
    tmp_lr = base_lr
    epoch_size = len(train_loader)
    wp_iter = len(train_loader) * args.wp_epoch
    total_epochs = args.max_epoch + args.wp_epoch
    lr_schedule = True
    warmup = True

    # optimizer
    if args.optimizer == 'adamw':
        print('Optimizer: AdamW')
        optimizer = optim.AdamW(
            model_without_ddp.parameters(), 
            lr=base_lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'sgd':
        print('Optimizer: SGD')
        optimizer = optim.SGD(
            model_without_ddp.parameters(),
            momentum=args.momentum,
            lr=base_lr,
            weight_decay=args.weight_decay
            )

    # loss
    criterion = torch.nn.CrossEntropyLoss().to(device)

    t0 = time.time()
    print("-------------- start training ----------------")
    for epoch in range(total_epochs):
        if not warmup:
            # use cos lr decay
            T_max = total_epochs - 15
            if epoch + 1 > T_max and lr_schedule:
                print('Cosine annealing is over !!')
                lr_schedule = False
                set_lr(optimizer, min_lr)

            if lr_schedule:
                # Cosine Annealing Scheduler
                tmp_lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*epoch / T_max))
                set_lr(optimizer, tmp_lr)

        # train one epoch
        for iter_i, (images, target) in enumerate(train_loader):
            ni = iter_i + epoch * epoch_size

            # warmup
            if ni < wp_iter and warmup:
                alpha = ni / wp_iter
                warmup_factor = 0.00066667 * (1 - alpha) + alpha
                tmp_lr = base_lr * warmup_factor
                set_lr(optimizer, tmp_lr)
            elif ni >= wp_iter and warmup:
                print('Warmup is Over !!!')
                warmup = False
                set_lr(optimizer, base_lr)

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # inference
            output = model(images)

            # loss
            loss = criterion(output, target)

            if torch.isnan(loss):
                continue

            # accu
            acc = accuracy(output, target, topk=(1, 5,))            

            # bp
            loss /= args.accumulation
            loss.backward() 

            if ni % args.accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                # ema
                if args.ema:
                    ema.update(model)


            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('loss',  loss.item(),  ni)
                    tblogger.add_scalar('acc1',  acc[0].item(),  ni)
                    tblogger.add_scalar('acc5',  acc[1].item(),  ni)
                
                t1 = time.time()
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch+1, args.max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(tmp_lr)
                # loss infor
                log += '[loss: {:.6f}]'.format(loss.item())
                # other infor
                log += '[acc1: {:.2f}]'.format(acc[0].item())
                log += '[acc5: {:.2f}]'.format(acc[1].item())
                log += '[time: {:.2f}]'.format(t1 - t0)

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

        # evaluate
        if distributed_utils.is_main_process():
            print('evaluating ...')
            model_eval = ema.ema if args.ema else model_without_ddp
            loss, acc1 = validate(
                device=device,
                val_loader=val_loader,
                model=model_eval,
                criterion=criterion
                )
            print('On val dataset: [loss: %.2f][acc1: %.2f]' 
                    % (loss.item(), 
                    acc1[0].item()),
                    flush=True)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                print('saving the model ...')
                checkpoint_path = os.path.join(path_to_save, 'best_model.pth')
                torch.save({'model': model_eval.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args}, 
                            checkpoint_path)                      
            

def validate(device, val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    acc1_num_pos = 0.
    count = 0.
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i % 100 == 0:
                print("[%d]/[%d] ...".format(i, len(val_loader)))
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
        
        # top1 acc & top5 acc
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