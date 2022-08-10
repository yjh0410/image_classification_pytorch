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

|    Model     | size | acc@1 | FLOPs | Params |  Weight |
|--------------|------|-------|-------|--------|---------|
| CSPDarkNet-L | 224  |  75.4 | 5.3 B | 33.4 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_l.pth) |
| ELANNet      | 224  |  75.1 | 4.1 B | 14.4 M | [github](https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elannet.pth) |
| ELANNet-Tiny | 224  |   | 0.3 B | 1.3 M  | [github]() |
