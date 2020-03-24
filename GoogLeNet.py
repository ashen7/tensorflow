import os
import time
import random

from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from GoogleNet.inceptionv1 import GoogLeNet_slim_v1
from GoogleNet.inceptionv2 import GoogLeNet_slim_v2
from GoogleNet.inceptionv3 import GoogLeNet_slim_v3
from cifar_data_load import get_training_dataset, get_test_dataset
from common import get_onehot_label

val_acc_list = list()
test_acc_list = list()
# 训练
def train(GoogLeNet, image_size, version, batch_size):
    global val_acc_list
    global test_acc_list
    # 加载训练集
    train_sample, train_label = get_training_dataset()
    learning_rate_val = 0.001
    momentum = 0.9
    class_num = 10

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1] ,image_size[2]], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')
    learning_rate = tf.placeholder(dtype=tf.float32)

    # 建立网络
    output = GoogLeNet(x, class_num, is_training=True)
    # 定义训练的loss函数 得到一个batch的平均值
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    # 定义优化器 动量SGD
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss)
    # 定义准确率
    val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    # saver持久化 保存模型
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    epochs = 4
    train_num = int(len(train_sample) * 0.8)
    val_num = int(len(train_sample) * 0.2)
    assert len(train_sample) == len(train_label)
    assert train_num + val_num == len(train_sample)
    print('training number is: ', train_num)
    print('validation number is: ', val_num)

    with tf.Session() as session:
        # 编译 静态图
        session.run(init)

        # 加载模型
        model = tf.train.get_checkpoint_state("./model/googlenet/" + version)
        if model and model.model_checkpoint_path:
            saver.restore(session, model.model_checkpoint_path)
            print('Successfully Load GoogLeNet Model!')

        for epoch in range(epochs):
            begin = time.time()
            # 生成并打乱训练集的顺序
            index = np.arange(len(train_sample))
            random.shuffle(index)

            # batch size为100 训练集前40000训练 后10000验证
            for i in tqdm(range(0, train_num, batch_size)):
                train_batch_sample = list()
                train_batch_label = list()
                for j in range(0, batch_size):
                    # train_batch_sample.append(train_sample[index[i + j]] / 255)
                    train_batch_sample.append(cv.resize(train_sample[index[i + j]], (image_size[0], image_size[1])) / 255)
                    train_batch_label.append(train_label[index[i + j]])
                train_batch_label = get_onehot_label(train_batch_label, class_num=class_num)
                # 训练
                a, b, c = session.run([optimizer, loss, val_acc], feed_dict={
                    x: train_batch_sample,
                    y: train_batch_label,
                    learning_rate: learning_rate_val
                })
                if (i % 10000 == 0):
                    print('train loss: {}, train acc: {}'.format(b, c))

            acc = 0
            iter_count = val_num / batch_size
            for i in tqdm(range(train_num, train_num+val_num, batch_size)):
                val_batch_sample = list()
                val_batch_label = list()
                for j in range(0, batch_size):
                    # val_batch_sample.append(train_sample[index[i + j]] / 255)
                    val_batch_sample.append(cv.resize(train_sample[index[i + j]], (image_size[0], image_size[1])) / 255)
                    val_batch_label.append(train_label[index[i + j]])
                val_batch_label = get_onehot_label(val_batch_label, class_num=class_num)
                acc += session.run(val_acc, feed_dict={
                    x: val_batch_sample,
                    y: val_batch_label,
                })
            # 输出 一共50次验证数据相加
            val_acc_list.append(acc)
            print('====================Epoch {}: validation acc: {}, spend time: {}s==================='.format(epoch + 1, acc / iter_count, time.time() - begin))

            # 每训练2轮 测试一次
            test_sample, test_label = get_test_dataset()
            test_acc = 0
            test_num = len(test_sample)
            test_iter_count = test_num / batch_size
            for i in tqdm(range(0, test_num, batch_size)):
                test_batch_sample = list()
                test_batch_label = list()
                for j in range(0, batch_size):
                    test_batch_sample.append(cv.resize(test_sample[i + j], (image_size[0], image_size[1])) / 255)
                    test_batch_label.append(test_label[i + j])
                test_batch_label = get_onehot_label(test_batch_label, class_num=class_num)
                test_acc += session.run(val_acc, feed_dict={
                    x: test_batch_sample,
                    y: test_batch_label,
                })
            # 输出 一共50次验证数据相加
            test_acc_list.append(test_acc)
            print('====================Epoch {}: test acc: {}, spend time: {}s==================='.format(epoch + 1, test_acc / test_iter_count, time.time() - begin))

            # 每训练5epoch 保存模型
            if (epoch + 1) % 5 == 0:
                # 保存模型
                saver.save(session, "model/googlenet/" + version + "/googlenet.ckpt")

        # 保存模型
        saver.save(session, "model/googlenet/" + version + "/googlenet.ckpt")
        plot_acc()

def test(GoogLeNet, image_size, version, batch_size):
    # 加载训练集
    test_sample, test_label = get_test_dataset()
    class_num = 10
    test_num = len(test_sample)

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1] ,image_size[2]], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, class_num], name='y')

    # 建立网络
    output = GoogLeNet(x, class_num)
    # 预测
    predict_output = tf.argmax(output, 1)
    ground_truth = tf.argmax(y, 1)

    # 定义准确率
    test_acc = tf.reduce_mean(tf.cast(tf.equal(predict_output, ground_truth), tf.float32))
    acc = 0
    # saver持久化 保存模型
    saver = tf.train.Saver()

    with tf.Session() as session:
        # 加载模型
        model = tf.train.get_checkpoint_state("./model/googlenet/" + version)
        if model and model.model_checkpoint_path:
            saver.restore(session, model.model_checkpoint_path)
            print('Successfully Load GoogLeNet Model!')

        for i in tqdm(range(0, test_num, batch_size)):
            test_batch_sample = list()
            test_batch_label = list()
            for j in range(batch_size):
                # test_batch_sample.append(test_sample[i + j] / 255)
                test_batch_sample.append(cv.resize(test_sample[i + j], (image_size[0], image_size[1])) / 255)
                test_batch_label.append(test_label[i + j])
            test_batch_label = get_onehot_label(test_batch_label, class_num=class_num)
            a, b = session.run([test_acc, output], feed_dict={
                x: test_batch_sample,
                y: test_batch_label
            })
            acc += a
        print('test acc is {}%'.format(acc / test_num * 100))

        class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        image_path = os.getcwd() + "/../DL_HotNet_Tensorflow/net/"
        for i in range(1, 11):
            name = image_path + str(i) + '.jpg'
            img = cv.imread(name)
            img = cv.resize(img, (32, 32))
            img = cv.resize(img, (image_size[0], image_size[1])) / 255
            img = np.array([img])
            predict = session.run(predict_output, feed_dict={x: img})
            print('{} image detect result is : {}'.format(class_list[i - 1], class_list[predict[0]]))

# 画出损失函数的趋势图
def plot_acc():
    plt.title('Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('val acc')
    plt.plot(range(0, len(val_acc_list)), val_acc_list, 'g', label='val acc')
    plt.plot(range(0, len(test_acc_list)), test_acc_list, 'r', label='test acc')
    plt.legend(['validation accuracy', 'test accuracy'], loc='best')
    plt.show()

if __name__ == '__main__':
    os.chdir('/home/yipeng/workspace/python/tensorflow_samples')

    version = 'v3'
    if version == 'v1':
        # 训练GoogLeNet v1
        train(GoogLeNet=GoogLeNet_slim_v1, image_size=[224, 224, 3], version=version, batch_size=100)
        test(GoogLeNet=GoogLeNet_slim_v1, image_size=[224, 224, 3], version=version, batch_size=100)
    elif version == 'v2':
        # 训练GoogLeNet v2
        train(GoogLeNet=GoogLeNet_slim_v2, image_size=[224, 224, 3], version=version, batch_size=100)
        # test(GoogLeNet=GoogLeNet_slim_v2, image_size=[224, 224, 3], version=version, batch_size=100)
    elif version == 'v3':
        # 训练GoogLeNet v3
        train(GoogLeNet=GoogLeNet_slim_v3, image_size=[299, 299, 3], version=version, batch_size=50)
        # test(GoogLeNet=GoogLeNet_slim_v3, image_size=[299, 299, 3], version=version, batch_size=50)
    else:
        raise NameError('GoogLeNet Inception没有这个版本')
