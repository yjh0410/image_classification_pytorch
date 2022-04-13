import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(path, dataset_type=None):
    # 获取图像数据
    file = h5py.File(path + '/cifar_' + dataset_type + '.h5','r')
    img_datas = file['data'][:]
    file.close()
    # 获取标签数据
    label_datas = np.loadtxt(path + '/cifar_' + dataset_type + '_labels.txt')
    return img_datas, label_datas

def draw_results(path_to_save):
    loss = np.loadtxt(path_to_save + '/train_error.txt')
    accu = np.loadtxt(path_to_save + '/train_accuracy.txt')
    plt.subplot(121)
    plt.plot(loss)
    plt.subplot(122)
    plt.plot(accu)
    plt.show()



if __name__ == "__main__":
    file = 'CIFAR/cifar-10/cifar-10-batches/'
    data_batch = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    img_flattens = []
    label_list = []
    for filename in data_batch:
        path = file + filename
        batch = unpickle(path)
        # 获取当前 batch 的标签
        labels = batch[b'labels']
        label_list.append(labels)
        # 获取当前 batch 的图像信息
        img_datas = batch[b'data']
        # 获取当前 batch 的图像名字
        img_names = batch[b'filenames']
        #print(img_data.shape)
        # 还原图像
        for img_data in img_datas:
            img_r = np.expand_dims(img_data[:1024].reshape(32,32),axis=2)
            img_g = np.expand_dims(img_data[1024:2048].reshape(32,32),axis=2)
            img_b = np.expand_dims(img_data[2048:].reshape(32,32),axis=2)
            img = np.concatenate([img_r, img_g, img_b], axis=2)
            # 存储图像数据
            img_flattens.append(img.reshape(1,-1))
            #img = Image.fromarray(img)
            #img.show()
    # 整合所有的batch的图片信息，shape=50000 x 3072
    img_flattens = np.concatenate(img_flattens)
    print(img_flattens.shape)
    file = h5py.File('CIFAR/cifar-10/cifar_train/cifar_train.h5','w')
    file['data'] = img_flattens
    file.close()
    # 整合所有的batch的标签信息，shape= 50000
    label_list = np.array(label_list).reshape(1,-1)[0]
    print(label_list)
    np.savetxt('CIFAR/cifar-10/cifar_train/cifar_train.txt',label_list)

    #img = img_flattens[100].reshape(32,32,3)
    #img = Image.fromarray(img)
    #img.show()
