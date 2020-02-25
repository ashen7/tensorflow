import pickle
import numpy as np

from image_augmentation import image_augment, image_augment_batch

# 加载数据(字节流)到python dict
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

# 得到一张图片
def get_image(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]
    r = np.reshape(r, [32, 32, 1])
    g = pixel[1024:2048]
    g = np.reshape(g, [32, 32, 1])
    b = pixel[2048:3072]
    b = np.reshape(b, [32, 32, 1])

    image = np.concatenate([r, g ,b], -1)
    return image

# 得到训练数据集
def get_training_dataset(is_image_augmentation=False):
    cifar = "cifar-10-batches-py/data_batch_"
    batch_label = list()
    label = list()
    sample = list()
    filenames = list()
    for i in range(1, 1+5):
        batch_label.append(unpickle(cifar + str(i))[b'batch_label'])
        label += unpickle(cifar + str(i))[b'labels']
        sample.append(unpickle(cifar + str(i))[b'data'])
        filenames += unpickle(cifar + str(i))[b'filenames']
    # 5个一维数组组成的二维数组拼接成一个一维数组
    sample = np.concatenate(sample, 0)

    if is_image_augmentation:
        image_aug_sample = list()
        image_aug_label = list()
        array = np.ndarray([len(sample), 32 ,32 ,3], dtype=np.uint8)
        for i in range(len(sample)):
            array[i] = get_image(sample[i])
            for j in range(10):
                image_aug_label.append(label[i])

        image_aug_sample = image_augment_batch(array, len(array))

        print('Successfully load cifar10 training dataset with image augmentation')
        return image_aug_sample, image_aug_label
    else:
        array = np.ndarray([len(sample), 32, 32, 3], dtype=np.uint8)
        for i in range(len(sample)):
            array[i] = get_image(sample[i])

        print('Successfully load cifar10 training dataset')
        return array, label

# 得到测试数据集
def get_test_dataset():
    cifar = "cifar-10-batches-py/test_batch"
    batch_label = list()
    filenames = list()

    batch_label.append(unpickle(cifar)[b'batch_label'])
    label = unpickle(cifar)[b'labels']
    sample = unpickle(cifar)[b'data']
    filenames += unpickle(cifar)[b'filenames']

    array = np.ndarray([len(sample), 32 ,32 ,3], dtype=np.uint8)
    for i in range(len(sample)):
        array[i] = get_image(sample[i])
    print('Successfully load cifar10 test dataset')

    return array, label

if __name__ == '__main__':
    a,b=get_training_dataset()
    print(a[0].shape)
