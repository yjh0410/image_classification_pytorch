import torch
import torchvision
from torchvision import transforms as tf

import os
import time
import argparse

from models import build_model
from utils.misc import accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    # Model
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='resnet18, resnet34, ...')
    parser.add_argument('--weight', type=str, default='weights/',
                        help='path to weight')

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

    # 指定cuda来调用GPU训练，默认不用，但要是安装了GPU版torch，一定要用
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 构建数据集
    pixel_mean = [0.]
    pixel_std = [1.0]
    val_data_root = os.path.join(args.data_path, 'val')
    val_dataset = torchvision.datasets.ImageFolder(
                        root=val_data_root, 
                        transform=tf.Compose([
                            tf.Resize(args.img_size),
                            tf.ToTensor(),
                            tf.Normalize(pixel_mean,
                                        pixel_std)]))
    val_loader = torch.utils.data.DataLoader(
                        dataset=val_dataset,
                        batch_size=1, 
                        shuffle=False,
                        num_workers=args.num_workers, 
                        pin_memory=True)
    
    print('total validation data size : ', len(val_dataset))

    # 构建模型
    model = build_model(model_name=args.model, 
                        num_classes=args.num_classes)
    # 加载训练好的模型文件，args.weight就是指向模型权重文件的路径
    model.load_state_dict(torch.load(args.weight, map_location='cpu')["model"], strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    # 定义loss函数，这是标准的分类问题使用的交叉熵函数
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print("-------------- start training ----------------")
    acc1_num_pos = 0.
    count = 0.
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i % 100 == 0:
                print("[{}]/[{}] ...".format(i, len(val_loader)))
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

    print('On val dataset: [loss: %.2f][acc1: %.2f]' 
            % (loss.item(), 
                acc1.item()),
            flush=True)



if __name__ == "__main__":
    main()