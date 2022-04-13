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

    # use cuda
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    val_data_root = os.path.join(args.data_path, 'val')
    val_dataset = torchvision.datasets.ImageFolder(
                        root=val_data_root, 
                        transform=tf.Compose([
                            tf.Resize(args.img_size),
                            tf.ToTensor(),
                            tf.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])]))
    val_loader = torch.utils.data.DataLoader(
                        dataset=val_dataset,
                        batch_size=1, 
                        shuffle=False,
                        num_workers=args.num_workers, 
                        pin_memory=True)
    
    print('total validation data size : ', len(val_dataset))

    # build model
    model = build_model(model_name=args.model, 
                        pretrained=args.pretrained,
                        num_classes=args.num_classes)
    model.load_state_dict(args.weight)
    model = model.to(device).eval()
    print('Finished loading model!')

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print("-------------- start training ----------------")
    acc1_num_pos = 0.
    acc5_num_pos = 0.
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
            cur_acc1, cur_acc5 = accuracy(output, target, topk=(1, 5))

            # Count the number of positive samples
            bs = images.shape[0]
            count += bs
            acc1_num_pos += cur_acc1 * bs
            acc5_num_pos += cur_acc5 * bs
        
        # top1 acc & top5 acc
        acc1 = acc1_num_pos / count
        acc5 = acc5_num_pos / count

    print('On val dataset: [loss: %.2f][acc1: %.2f][acc5: %.2f]' 
            % (loss.item(), 
                acc1.item(), 
                acc5.item()),
            flush=True)



if __name__ == "__main__":
    main()