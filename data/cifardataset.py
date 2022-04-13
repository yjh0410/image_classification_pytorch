import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from .cifar_tools import get_data

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.tensor([sample[1]]).long())
    return torch.stack(imgs, 0), torch.tensor(targets).view(-1)

CIFAR_CLASS = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

class cifar(object):
    def __init__(self, path_to_data, dataset_type='train'):
        self.total_img, self.total_label = get_data(path_to_data, dataset_type)
        self.total = self.total_img.shape[0]

    def pull_item(self, index):
        img = torch.from_numpy(self.total_img[index, :].reshape(32,32,3)).permute(2, 0, 1)
        label = self.total_label[index]
        return img, label
    
    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.total_img)

    def pull_img(self, index):
        return self.total_img[index, :].reshape(32,32,3)
    
    def pull_label(self, index):
        return self.total_label[index]


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == "__main__":
    path_to_train = '../cifar-10/cifar_train/'
    path_to_test = '../cifar-10/cifar_test/'

    dataset = cifar(path_to_train, 'train')
    print(len(dataset))
    batch_size = 100
    data_loader = data.DataLoader(dataset, batch_size,
                                  num_workers=1,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    batch_iterator = iter(cycle(data_loader))
    for i in range(10000):
        img, label = next(batch_iterator)
        plt.imshow(img[0].permute(1,2,0).numpy())
        plt.title(CIFAR_CLASS[label[0].item()])
        plt.show()
        print(i)