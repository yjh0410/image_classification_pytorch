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

|    Model            | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------------|-------|------|-------|--------|--------|---------|
| DarkNet-19          | 90    | 224  |  72.9 | 5.4    | 20.8 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth) |
| DarkNet-53          | 120   | 224  |  75.7 | 14.2   | 41.6 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53.pth) |
| DarkNet-53-SiLU     | 100   | 224  |  74.4 | 14.3   | 41.6 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53_silu.pth) |
| CSP-DarkNet-53-SiLU | 100   | 224  |  75.0 | 9.4    | 27.3 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet53_silu.pth) |
| ELANNet-Pico        | 100   | 224  |  57.0 | 0.2    | 0.8 M  | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_pico.pth) |
| ELANNet-Nano        | 100   | 224  |  59.5 | 0.4    | 0.9 M  | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_nano.pth) |
| ELANNet-Small       | 100   | 224  |  70.1 | 1.6    | 3.2 M  | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_small.pth) |
| ELANNet-Medium      | 100   | 224  |  74.1 | 4.3    | 8.3 M  | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_medium.pth) |
| ELANNet-Large       | 100   | 224  |  75.7 | 9.2    | 17.1 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_large.pth) |
| ELANNet-Huge        | 100   | 224  |  76.5 | 16.6   | 30.7 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_huge.pth) |
| ELAN-CSPNet-Nano    | 100   | 224  |   | 0.4    | 1.3 M  |  |
| ELAN-CSPNet-Small   | 100   | 224  |   | 1.4    | 4.9 M  |  |
| ELAN-CSPNet-Medium  | 100   | 224  |   | 4.7    | 11.6 M |  |
| ELAN-CSPNet-Large   | 100   | 224  |   | 10.5   | 19.6 M |  |
| ELAN-CSPNet-Huge    | 100   | 224  |   | 16.3   | 30.6 M |  |
| CSPDarkNet-Nano     | 100   | 224  |   | 0.3    | 1.3 M  |  |
| CSPDarkNet-Small    | 100   | 224  |   | 1.3    | 4.6 M  |  |
| CSPDarkNet-Medium   | 100   | 224  |   | 3.8    | 12.8 M |  |
| CSPDarkNet-Large    | 100   | 224  |   | 8.6    | 27.5 M |  |
| CSPDarkNet-Huge     | 100   | 224  |   | 16.3   | 50.5 M |  |

