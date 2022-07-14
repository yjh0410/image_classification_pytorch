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

|    Model     | size | acc@1 | acc@5 | FLOPs | Params |  Weight |
|--------------|------|-------|-------|-------|--------|---------|
| CSPDarkNet   | 224  |       |       | 5.3 B | 33.4 M | [github]() |
| ELANNet      | 224  |       |       | 4.1 B | 14.4 M | [github]() |

