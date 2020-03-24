from tensorflow.contrib.keras.api.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.contrib.keras.api.keras.models import Model, load_model
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.utils import to_categorical

from cifar_data_load import get_training_dataset, get_test_dataset
import numpy as np

def LeNet(input):
    # 卷积层conv1
    conv1 = Conv2D(6, 5, (1, 1), 'valid', use_bias=True, activation='relu')(input)
    # 最大池化层max_pool1
    max_pool1 = MaxPool2D((2, 2), (2, 2), 'valid')(conv1)
    # 卷积层conv2
    conv2 = Conv2D(6, 5, (1, 1), 'valid', use_bias=True, activation='relu')(max_pool1)
    # 最大池化层max_pool2
    max_pool2 = MaxPool2D((2, 2), (2, 2), 'valid')(conv2)
    # 卷积层conv3
    conv3 = Conv2D(16, 5, (1, 1), 'valid', use_bias=True, activation='relu')(max_pool2)
    # 展开
    flatten = Flatten()(conv3)
    # 全连接层dense1
    dense1 = Dense(120, activation='relu')(flatten)
    # 全连接层dense2
    dense2 = Dense(84, activation='relu')(dense1)
    # 全连接层dense3
    dense3 = Dense(10, activation='softmax')(dense2)

    return dense3

def train():
    # 输入
    input = Input([32, 32, 3])
    # 构建网络
    output = LeNet(input)
    # 建立模型
    model = Model(input, output)

    # 定义优化器
    adam = Adam(lr=0.0003)
    # 编译模型
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # 准备数据
    # 获得输入特征sample
    train_sample, train_label = get_training_dataset()
    # 获得one-hot之后的label
    train_label = to_categorical(train_label)

    # 训练模型
    model.fit(train_sample, train_label, 200, 50, 1, validation_split=0.2, shuffle=True)
    # 保存模型
    model.save('lenet.h5')

def test():
    # 加载训练好的模型
    model = load_model('model/lenet/lenet.h5')
    # 获得测试数据
    test_sample, test_label = get_test_dataset()
    print("test accuracy: {}%".format(np.sum(np.equal(test_label, np.argmax(model.predict(test_sample), 1))) / len(test_label) * 100))

if __name__ == '__main__':
    test()