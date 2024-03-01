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

* ELANNet-v2

|    Model            | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------------|-------|------|-------|--------|--------|---------|
| ELANNetv2-Pico      | 100   | 224  |  59.8 | 0.2    | 0.6 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_pico.pth) |
| ELANNetv2-Nano      | 100   | 224  |  60.8 | 0.4    | 0.9 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_nano.pth) |
| ELANNetv2-Tiny      | 100   | 224  |  67.1 | 0.9    | 1.9 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_tiny.pth) |
| ELANNetv2-Small     | 100   | 224  |  70.4 | 1.7    | 3.3 M  | [ckpt](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_v2_small.pth) |

* RTCNet (Yolov8's backbone)

|    Model      | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------|-------|------|-------|--------|--------|---------|
| RTCNet-P      | 200   | 224  |       |        |        |  |
| RTCNet-N      | 200   | 224  |       |        |        |  |
| RTCNet-S      | 200   | 224  |       |        |        |  |
| RTCNet-M      | 200   | 224  |       |        |        |  |
| RTCNet-L      | 200   | 224  |       |        |        |  |
| RTCNet-X      | 200   | 224  |       |        |        |  |
