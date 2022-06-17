from copy import deepcopy
import os
import time
import math
import argparse

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms as tf

from models import build_model
from utils.misc import ModelEMA, accuracy
from utils.com_flops_params import FLOPs_and_Params


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--wp_epoch', type=int, default=5, 
                        help='warmup epoch')
    parser.add_argument('--max_epoch', type=int, default=300, 
                        help='max epoch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--base_lr', type=float,
                        default=0.1, help='learning rate for training model')
    parser.add_argument('--min_lr_ratio', type=float,
                        default=0.05, help='the final lr')
    parser.add_argument('--path_to_save', type=str, 
                        default='weights/')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='sgd, adam')
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


    return parser.parse_args()

    
def main():
    args = parse_args()

    path_to_save = os.path.join(args.path_to_save)
    os.makedirs(path_to_save, exist_ok=True)
    
    # use gpu
    if args.cuda:
        print("use cuda")
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
    train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,
                        num_workers=args.num_workers, 
                        pin_memory=True)
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
    ema = ModelEMA(model) if args.ema else None

    # FLOPs * Params
    model_copy = deepcopy(model)
    model_copy.eval()
    FLOPs_and_Params(model=model, size=args.img_size)
    model_copy.train()

    # basic config
    best_acc1 = -1.
    base_lr = args.base_lr
    min_lr = base_lr * args.min_lr_ratio
    tmp_lr = base_lr
    epoch_size = len(train_loader)
    wp_iter = len(train_loader) * args.wp_epoch
    total_epochs = args.max_epoch + args.wp_epoch
    lr_schedule = True
    warmup = True

    # optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=base_lr,
                                weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                                lr=base_lr,
                                weight_decay=1e-4)

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
            if ni < wp_iter:
                alpha = ni / wp_iter
                warmup_factor = 0.00066667 * (1 - alpha) + alpha
                tmp_lr = base_lr * warmup_factor
                set_lr(optimizer, tmp_lr)
            elif ni == wp_iter:
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        print('evaluating ...')
        loss, acc1 = validate(device, val_loader, model, criterion)
        print('On val dataset: [loss: %.2f][acc1: %.2f]' 
                % (loss.item(), 
                   acc1[0].item()),
                flush=True)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            print('saving the model ...')
            checkpoint_path = os.path.join(path_to_save, 'best_model.pth')
            torch.save({'model': model.state_dict(),
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


if __name__ == "__main__":
    main()