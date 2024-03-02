import argparse
import os
import torch

import sys
sys.path.append("..")
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size.')    
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of the classes.')    
    # Model
    parser.add_argument('-m', '--model', type=str, default='vit_tiny',
                        help='model name')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')

    return parser.parse_args()

    
def main():
    args = parse_args()

    model = build_model(args)
    model.eval()
    path = os.path.split(args.resume)

    print('Resave: {}'.format(args.model.upper()))
    checkpoint_path = '{}/{}_pure.pth'.format(path[0], args.model)
    torch.save({'model': model.state_dict(),}, 
                checkpoint_path)

if __name__ == "__main__":
    main()