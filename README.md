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

|    Model     | size | acc@1 | FLOPs  | Params |  Weight |
|--------------|------|-------|--------|--------|---------|
| DarkNet-19   | 224  |  72.9 | 2.7 B  | 20.8 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth) |
| DarkNet-53   | 224  |  75.7 | 7.1 B  | 41.6 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53.pth) |
| ELANNet      | 224  |  75.1 | 4.1 B  | 14.4 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet.pth) |
| ELANNet-Huge | 224  |   | 7.5 B | 26.4 M |  |
| ELANNet-Tiny | 224  |  64.8 | 0.3 B  | 1.4 M  | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_tiny.pth) |
| ELANNet-Nano | 224  |  48.7 | 0.03 B | 0.4 M  | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet_nano.pth) |
