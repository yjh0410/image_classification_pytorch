import torch
import torchvision
from torchvision import transforms as tf

import os
import cv2
import numpy as np
import argparse

from models import build_model


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
    parser.add_argument('--weight', type=str, 
                        default='weights/')

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

    # build model
    model = build_model(model_name=args.model, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weight, map_location='cpu')["model"], strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    print("-------------- run demo ----------------")
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i % 100 == 0:
                print("[{}]/[{}] ...".format(i, len(val_loader)))
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 将数据喂给模型，得到输出
            output = model(images)

            # 用softmax函数处理输出，得到每个类别预测的概率
            output = torch.softmax(output, dim=-1)

            # 因为output的维度是[B, C]，B就是batch size，已经在dataloader里面设置成了1，所以我们你不需要这个维度
            output = output[0]  # [C]

            # score是类别的概率，是01之间的数，label是对应的标签。
            score, label = torch.topk(output, 1)
            print('Score: {:.3f} || Label: {}'.format(score.item(), label.item()))

            # 为了可视化，我们将images先放到cpu上，然而将[B,C,H,W]转换成[H,W,C]的格式，再变成numpy
            # 注意，这里的image已经被归一化了，所以我们为了方便可视化，看到熟悉的图片，要做个反归一化操作
            image = images[0].cpu().permute(1, 2, 0).numpy()
            # denormalize
            # # to BGR，这段被注释掉了，是因为你的zita01数据集都会黑白的，就没必要调整RGB通道了，但也说一下为什么要有这一步
            # # 因为我用opencv来可视化，需要图像的颜色通道为BGR顺序，而image是RGB顺序，所以需要将
            # # RGB通道顺序转成BGR。
            # image = image[..., (2, 1, 0)]
            # 下面是反归一化操作
            image = (image * np.array(pixel_std) + np.array(pixel_mean)) * 255.
            # 将image的数据格式从float变成uint8，符合opencv的图片格式
            image = image.astype(np.uint8)

            # opencv操作
            cv2.imshow('classification', image)
            cv2.waitKey(0)



if __name__ == "__main__":
    main()