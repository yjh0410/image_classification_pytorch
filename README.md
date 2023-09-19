# ImageNet Classification


# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n imagenet python=3.6
```

- Then, activate the environment:
```Shell
conda activate imagenet
```

- Requirements:
```Shell
pip install -r requirements.txt 
```
PyTorch >= 1.9.1 and Torchvision >= 0.10.1

# Experiments
## ImageNet val

* YOLOv2~v4's Backbone

|    Model              | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|-----------------------|-------|------|-------|--------|--------|---------|
| DarkNet-19            | 90    | 224  |  72.9 | 5.4    | 20.8 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth) |
| DarkNet-53-LReLU      | 120   | 224  |  75.7 | 14.2   | 41.6 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53.pth) |
| DarkNet-53-SiLU       | 100   | 224  |  74.4 | 14.3   | 41.6 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53_silu.pth) |
| CSP-DarkNet-53-SiLU   | 100   | 224  |  75.0 | 9.4    | 27.3 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet53_silu.pth) |
| DarkNet-Tiny          | 100   | 224  |  60.1 | 0.5    | 1.6 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet_tiny.pth) |
| CSPDarkNet-Tiny       | 100   | 224  |  61.1 | 0.4    | 1.3 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_tiny.pth) |

* YOLOv5's Backbone

|    Model            | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------------|-------|------|-------|--------|--------|---------|
| CSPDarkNet-Nano     | 100   | 224  | 60.6  | 0.3    | 1.3 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_nano.pth) |
| CSPDarkNet-Small    | 100   | 224  | 69.8  | 1.3    | 4.6 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_small.pth) |
| CSPDarkNet-Medium   | 100   | 224  | 72.9  | 3.8    | 12.8 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_medium.pth) |
| CSPDarkNet-Large    | 100   | 224  | 75.1  | 8.6    | 27.5 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_large.pth) |
| CSPDarkNet-Huge     | 100   | 224  |       | 16.3   | 50.5 M |  |

* YOLOv7's Backbone

|    Model            | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------------|-------|------|-------|--------|--------|---------|
| ELANNet-Nano        | 100   | 224  |  48.7 | 0.03   | 0.4 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_nano.pth) |
| ELANNet-Tiny        | 100   | 224  |  64.8 | 0.3    | 1.4 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_tiny.pth) |
| ELANNet-Large       | 100   | 224  |  75.1 | 4.1    | 14.4 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_large.pth) |
| ELANNet-Huge        | 100   | 224  |  76.2 | 7.5    | 26.4 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/yolov7_elannet_huge.pth) |

* Modified YOLOv7's Backbone

|    Model            | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------------|-------|------|-------|--------|--------|---------|
| ELANNet-Pico        | 100   | 224  |  57.6 | 0.2    | 0.6 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_pico.pth) |
| ELANNet-Nano        | 100   | 224  |  59.5 | 0.4    | 0.9 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_nano.pth) |
| ELANNet-Tiny        | 100   | 224  |  66.9 | 1.2    | 1.9 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_tiny.pth) |
| ELANNet-Small       | 100   | 224  |  70.1 | 1.6    | 3.2 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_small.pth) |
| ELANNet-Medium      | 100   | 224  |  74.1 | 4.3    | 8.3 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_medium.pth) |
| ELANNet-Large       | 100   | 224  |  75.7 | 9.2    | 17.1 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_large.pth) |
| ELANNet-Huge        | 100   | 224  |  76.5 | 16.6   | 30.7 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_huge.pth) |

* YOLOv8's Backbone

|    Model            | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------------|-------|------|-------|--------|--------|---------|
| ELAN-CSPNet-Nano    | 100   | 224  |  61.4 | 0.4    | 1.3 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_nano.pth) |
| ELAN-CSPNet-Small   | 100   | 224  |  70.5 | 1.4    | 4.9 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_small.pth) |
| ELAN-CSPNet-Medium  | 100   | 224  |  74.1 | 4.7    | 11.6 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_medium.pth) |
| ELAN-CSPNet-Large   | 100   | 224  |  75.8 | 10.5   | 19.6 M | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_large.pth) |
| ELAN-CSPNet-Huge    | 100   | 224  |       | 16.3   | 30.6 M |  |

* ELANNet-v2

|    Model            | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------------|-------|------|-------|--------|--------|---------|
| ELANNetv2-Pico      | 100   | 224  |  59.8 | 0.2    | 0.6 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_pico.pth) |
| ELANNetv2-Nano      | 100   | 224  |  60.8 | 0.4    | 0.9 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_nano.pth) |
| ELANNetv2-Tiny      | 100   | 224  |  67.1 | 0.9    | 1.9 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_tiny.pth) |
| ELANNetv2-Small     | 100   | 224  |  70.4 | 1.7    | 3.3 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_small.pth) |
