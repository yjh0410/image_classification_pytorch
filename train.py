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
    parser.add_argument('--max_epoch', type=int, default=30, 
                        help='max epoch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='learning rate for training model')
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
    parser.add_argument('-size', '--img_size', type=int, default=224,
                        help='input size')
    parser.add_argument('--num_classes', type=int, default=16,
                        help='number of classes')


    return parser.parse_args()

    
def main():
    args = parse_args()

    path_to_save = os.path.join(args.path_to_save)
    os.makedirs(path_to_save, exist_ok=True)
    
    # 指定cuda来调用GPU训练，默认不用，但要是安装了GPU版torch，一定要用
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 这是一个trick，默认不用
    if args.ema:
        print('use EMA ...')

    # 调用tensorboard来保存训练结果，默认不用
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)

    # 构建数据集
    pixel_mean = [0.]   # 色彩通道均值，根据你的数据集，给0就行
    pixel_std = [1.0]   # 色彩通道方差，根据你的数据集，给1就行
    train_data_root = os.path.join(args.data_path, 'train')  # 训练集路径
    val_data_root = os.path.join(args.data_path, 'val')      # 验证集路径
    ## 训练集，用于读取训练集图像
    train_dataset = torchvision.datasets.ImageFolder(
                        root=train_data_root,
                        transform=tf.Compose([ # 这个参数是传入数据预处理
                            tf.RandomResizedCrop(args.img_size),  ## resize操作
                            tf.RandomHorizontalFlip(),            ## 随机水平翻转
                            tf.ToTensor(),                        ## 转换成torch所需的tensor类型数据
                            tf.Normalize(pixel_mean,             ## 图片归一化操作
                                         pixel_std)]))
    train_loader = torch.utils.data.DataLoader(
                        dataset=train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,
                        num_workers=args.num_workers, 
                        pin_memory=True)
    ## 验证集，用于读取验证集图像
    val_dataset = torchvision.datasets.ImageFolder(
                        root=val_data_root, 
                        transform=tf.Compose([
                            tf.Resize(args.img_size),
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

    # 构建模型
    model = build_model(model_name=args.model, 
                        pretrained=args.pretrained, 
                        norm_type=args.norm_type,
                        num_classes=args.num_classes)
    # 将模型转换为train模式
    model.train().to(device)
    ema = ModelEMA(model) if args.ema else None  # EMA是一个训练技巧，我暂时都没用，所以这里ema=None

    # 计算模型的FLOPs和参数量，可能对你的任务不需要这两个指标
    model_copy = deepcopy(model)
    model_copy.eval()
    FLOPs_and_Params(model=model, size=args.img_size)
    model_copy.train()

    # 一些基础设置
    best_acc1 = -1.
    base_lr = args.lr
    tmp_lr = base_lr
    epoch_size = len(train_loader)

    # 构建优化器，用SGD就行
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=base_lr,
                                weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                                lr=base_lr,
                                weight_decay=1e-4)

    # 定义loss函数，这是标准的分类问题使用的交叉熵函数
    criterion = torch.nn.CrossEntropyLoss().to(device)

    t0 = time.time()
    print("-------------- start training ----------------")
    for epoch in range(args.max_epoch):
        # Cosine学习率衰减策略
        tmp_lr = 1e-5 + 0.5*(base_lr - 1e-5)*(1 + math.cos(math.pi*epoch / args.max_epoch))
        # 调整优化器中的学习率
        set_lr(optimizer, tmp_lr)

        # 训练一个完整的epoch，一个epoch就是把数据集中的数据全都训练一次
        for iter_i, (images, target) in enumerate(train_loader):
            ni = iter_i + epoch * epoch_size
                
            # 将读进来的数据放在指定的device上，device在上面已定义
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 将数据喂给模型，得到输出
            output = model(images)

            # 计算模型输出与label之间的损失
            loss = criterion(output, target)

            # 检查loss是不是None，一般情况下这一步不会有问题
            if torch.isnan(loss):
                continue

            # 测试模型当前的预测精度，默认使用top1指标
            acc1 = accuracy(output, target, topk=(1,))            

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # ema
            if args.ema:
                ema.update(model)

            if iter_i % 10 == 0:
                # 一些输出界面的信息，能让你看到训练情况
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('loss',  loss.item(),  ni)
                    tblogger.add_scalar('acc1',  acc1[0].item(),  ni)
                
                t1 = time.time()
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch+1, args.max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(tmp_lr)
                # loss infor
                log += '[loss: {:.6f}]'.format(loss.item())
                # other infor
                log += '[acc1: {:.2f}]'.format(acc1[0].item())
                log += '[time: {:.2f}]'.format(t1 - t0)

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

        # 每训练完一个epoch，在验证集上测试一下top1准确率
        print('evaluating ...')
        loss, acc1 = validate(device, val_loader, model, criterion)
        print('On val dataset: [loss: %.2f][acc1: %.2f]' 
                % (loss.item(), 
                   acc1[0].item()),
                flush=True)

        # 保存准确率最高的模型
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