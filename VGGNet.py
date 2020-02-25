import os
import time
import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from cifar_data_load import get_training_dataset, get_test_dataset
from common import get_onehot_label

def VGGNet_slim(input, class_num, p=0.5):
    vgg = input

    with tf.variable_scope('vgg_net'):
        # tf的slim 代码瘦身 高度封装 写出简介的代码
        # arg_scope 给选定的层设置默认值
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # activation_fn=tf.nn.relu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.constant_initializer(0.0)):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='same', stride=1):
                # 重复调用一个函数 上一次输出作为下一次输入参数
                # 调用两次卷积运算 filter 3 3 64个 步长1 slim的conv2d封装了relu激活函数和biases_add
                vgg = slim.repeat(vgg, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                # 最大池化层 filter 2 2 步长2
                vgg = slim.max_pool2d(vgg, [2, 2], stride=2, scope='max_pool1')

                # 调用两次卷积运算 filter 3 3 128个 步长1 slim的conv2d封装了relu激活函数和biases_add
                vgg = slim.repeat(vgg, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                # 最大池化层 filter 2 2 步长2
                vgg = slim.max_pool2d(vgg, [2, 2], stride=2, scope='max_pool2')

                # 调用三次卷积运算 filter 3 3 256个 步长1 slim的conv2d封装了relu激活函数和biases_add
                vgg = slim.repeat(vgg, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                # 最大池化层 filter 2 2 步长2
                vgg = slim.max_pool2d(vgg, [2, 2], stride=2, scope='max_pool3')

                # 调用三次卷积运算 filter 3 3 512个 步长1 slim的conv2d封装了relu激活函数和biases_add
                vgg = slim.repeat(vgg, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                # 最大池化层 filter 2 2 步长2
                vgg = slim.max_pool2d(vgg, [2, 2], stride=2, scope='max_pool4')

                # 调用三次卷积运算 filter 3 3 512个 步长1 slim的conv2d封装了relu激活函数和biases_add
                vgg = slim.repeat(vgg, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                # 最大池化层 filter 2 2 步长2
                vgg = slim.max_pool2d(vgg, [2, 2], stride=2, scope='max_pool5')

                # 展开
                vgg = slim.flatten(vgg, scope='flatten')

                # 调用三次全连接层 每次输入参数分别取列表的1 2 3个值
                # vgg = slim.stack(vgg, slim.fully_connected, [1024, 1024, class_num], scope='full_connected')
                vgg = slim.fully_connected(vgg, 1024, scope='fc6')
                vgg = slim.dropout(vgg, p, scope='dropout6')
                vgg = slim.fully_connected(vgg, 1024, scope='fc7')
                vgg = slim.dropout(vgg, p, scope='dropout7')
                vgg = slim.fully_connected(vgg, 10, activation_fn=None, scope='fc8')

        return vgg

def VGG16(input, class_num, p=0.5):
    # 定义卷积层conv1
    with tf.name_scope('conv1'):
        # 第一层卷积层 filter 64 3 3 3 步长1
        conv1_kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv1 = tf.nn.conv2d(input, conv1_kernel, [1, 1, 1, 1], padding='SAME')
        conv1_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]),
                                   trainable=True, name='biases')
        conv1_bias = tf.nn.bias_add(conv1, conv1_biases)
        conv1 = tf.nn.relu(conv1_bias, name='conv1')

    # 定义卷积层conv2
    with tf.name_scope('conv2'):
        # 第二层卷积层 filter 64 3 3 3 步长1
        conv2_kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv2 = tf.nn.conv2d(conv1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        conv2_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]),
                                   trainable=True, name='biases')
        conv2_bias = tf.nn.bias_add(conv2, conv2_biases)
        conv2 = tf.nn.relu(conv2_bias, name='conv2')

    # 定义池化层pool1
    with tf.name_scope('pool1'):
        # 最大池化层 filter 2 2 步长2
        pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

    # 定义卷积层conv3
    with tf.name_scope('conv3'):
        # 第二层卷积层 filter 128 64 3 3 步长1
        conv3_kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv3 = tf.nn.conv2d(pool1, conv3_kernel, [1, 1, 1, 1], padding='SAME')
        conv3_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128]),
                                   trainable=True, name='biases')
        conv3_bias = tf.nn.bias_add(conv3, conv3_biases)
        conv3 = tf.nn.relu(conv3_bias, name='conv3')

    # 定义卷积层conv4
    with tf.name_scope('conv4'):
        # 第二层卷积层 filter 128 128 3 3 步长1
        conv4_kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv4 = tf.nn.conv2d(conv3, conv4_kernel, [1, 1, 1, 1], padding='SAME')
        conv4_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128]),
                                   trainable=True, name='biases')
        conv4_bias = tf.nn.bias_add(conv4, conv4_biases)
        conv4 = tf.nn.relu(conv4_bias, name='conv4')

    # 定义池化层pool2
    with tf.name_scope('pool2'):
        # 最大池化层 filter 2 2 步长2
        pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

    # 定义卷积层conv5
    with tf.name_scope('conv5'):
        # 第二层卷积层 filter 256 128 3 3 步长1
        conv5_kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv5 = tf.nn.conv2d(pool2, conv5_kernel, [1, 1, 1, 1], padding='SAME')
        conv5_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]),
                                   trainable=True, name='biases')
        conv5_bias = tf.nn.bias_add(conv5, conv5_biases)
        conv5 = tf.nn.relu(conv5_bias, name='conv5')

    # 定义卷积层conv6
    with tf.name_scope('conv6'):
        # 第二层卷积层 filter 256 256 3 3 步长1
        conv6_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv6 = tf.nn.conv2d(conv5, conv6_kernel, [1, 1, 1, 1], padding='SAME')
        conv6_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]),
                                   trainable=True, name='biases')
        conv6_bias = tf.nn.bias_add(conv6, conv6_biases)
        conv6 = tf.nn.relu(conv6_bias, name='conv6')

    # 定义卷积层conv7
    with tf.name_scope('conv7'):
        # 第二层卷积层 filter 256 256 3 3 步长1
        conv7_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv7 = tf.nn.conv2d(conv6, conv7_kernel, [1, 1, 1, 1], padding='SAME')
        conv7_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]),
                                   trainable=True, name='biases')
        conv7_bias = tf.nn.bias_add(conv7, conv7_biases)
        conv7 = tf.nn.relu(conv7_bias, name='conv7')

    # 定义池化层pool3
    with tf.name_scope('pool3'):
        # 最大池化层 filter 2 2 步长2
        pool3 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')

    # 定义卷积层conv8
    with tf.name_scope('conv8'):
        # 第二层卷积层 filter 512 256 3 3 步长1
        conv8_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv8 = tf.nn.conv2d(pool3, conv8_kernel, [1, 1, 1, 1], padding='SAME')
        conv8_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]),
                                   trainable=True, name='biases')
        conv8_bias = tf.nn.bias_add(conv8, conv8_biases)
        conv8 = tf.nn.relu(conv8_bias, name='conv8')

    # 定义卷积层conv9
    with tf.name_scope('conv9'):
        # 第二层卷积层 filter 512 512 3 3 步长1
        conv9_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv9 = tf.nn.conv2d(conv8, conv9_kernel, [1, 1, 1, 1], padding='SAME')
        conv9_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]),
                                   trainable=True, name='biases')
        conv9_bias = tf.nn.bias_add(conv9, conv9_biases)
        conv9 = tf.nn.relu(conv9_bias, name='conv9')

    # 定义卷积层conv10
    with tf.name_scope('conv10'):
        # 第二层卷积层 filter 256 256 3 3 步长1
        conv10_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv10 = tf.nn.conv2d(conv9, conv10_kernel, [1, 1, 1, 1], padding='SAME')
        conv10_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]),
                                   trainable=True, name='biases')
        conv10_bias = tf.nn.bias_add(conv10, conv10_biases)
        conv10 = tf.nn.relu(conv10_bias, name='conv10')

    # 定义池化层pool4
    with tf.name_scope('pool4'):
        # 最大池化层 filter 2 2 步长2
        pool4 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')

    # 定义卷积层conv11
    with tf.name_scope('conv11'):
        # 第二层卷积层 filter 512 512 3 3 步长1
        conv11_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv11 = tf.nn.conv2d(pool4, conv11_kernel, [1, 1, 1, 1], padding='SAME')
        conv11_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]),
                                   trainable=True, name='biases')
        conv11_bias = tf.nn.bias_add(conv11, conv11_biases)
        conv11 = tf.nn.relu(conv11_bias, name='conv11')

    # 定义卷积层conv12
    with tf.name_scope('conv12'):
        # 第二层卷积层 filter 512 512 3 3 步长1
        conv12_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                       stddev=1e-1, name='weights'))
        conv12 = tf.nn.conv2d(conv11, conv12_kernel, [1, 1, 1, 1], padding='SAME')
        conv12_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]),
                                   trainable=True, name='biases')
        conv12_bias = tf.nn.bias_add(conv12, conv12_biases)
        conv12 = tf.nn.relu(conv12_bias, name='conv12')

    # 定义卷积层conv13
    with tf.name_scope('conv13'):
        # 第二层卷积层 filter 256 256 3 3 步长1
        conv13_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                        stddev=1e-1, name='weights'))
        conv13 = tf.nn.conv2d(conv12, conv13_kernel, [1, 1, 1, 1], padding='SAME')
        conv13_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]),
                                    trainable=True, name='biases')
        conv13_bias = tf.nn.bias_add(conv13, conv13_biases)
        conv13 = tf.nn.relu(conv13_bias, name='conv13')

    # 定义池化层pool5
    with tf.name_scope('pool5'):
        # 最大池化层 filter 2 2 步长2
        pool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')

    with tf.name_scope('flatten'):
        flatten = tf.reshape(pool5, [-1, 1*1*512])

    # 定义全连接层dense1
    with tf.name_scope('dense1'):
        # 输入9216 输出4096
        fc1_weights = tf.Variable(tf.truncated_normal([1*1*512, 1024], mean=0, stddev=0.01))
        fc1_biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                 trainable=True, name='biases')
        fc1 = tf.nn.sigmoid(tf.matmul(flatten, fc1_weights) + fc1_biases, name='dense1')
        dense1 = tf.nn.dropout(fc1, p, name='dropout1')

    # 定义全连接层dense2
    with tf.name_scope('dense2'):
        # 输入4096 输出4096
        fc2_weights = tf.Variable(tf.truncated_normal([1024, 1024], mean=0, stddev=0.01))
        fc2_biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                 trainable=True, name='biases')
        fc2 = tf.nn.sigmoid(tf.matmul(dense1, fc2_weights) + fc2_biases, name='dense2')
        dense2 = tf.nn.dropout(fc2, p, name='dropout2')

    # 定义全连接层dense3
    with tf.name_scope('output'):
        # 输入4096 输出10
        fc3_weights = tf.Variable(tf.truncated_normal([1024, 10], mean=0, stddev=0.01))
        fc3_biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32),
                                 trainable=True, name='biases')
        dense3 = tf.matmul(dense2, fc3_weights) + fc3_biases

    return dense3

val_acc_list = list()
# 训练
def train():
    global val_acc_list
    # 加载训练集
    train_sample, train_label = get_training_dataset(is_image_augmentation=True)
    learning_rate_val = 0.001
    momentum = 0.9

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32 ,3], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')
    learning_rate = tf.placeholder(dtype=tf.float32)

    # 建立网络
    output = VGGNet_slim(x, 10)
    # 定义训练的loss函数 得到一个batch的平均值
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    # 定义优化器 动量SGD
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss)
    # 定义准确率
    val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    # saver持久化 保存模型
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    epochs = 10
    batch_size = 200
    train_num = int(len(train_sample) * 0.8)
    val_num = int(len(train_sample) * 0.2)
    assert len(train_sample) == len(train_label)
    assert train_num + val_num == len(train_sample)

    with tf.Session() as session:
        # 编译 静态图
        session.run(init)

        # 加载模型
        model = tf.train.get_checkpoint_state("./model/vggnet")
        if model and model.model_checkpoint_path:
            saver.restore(session, model.model_checkpoint_path)
            print('Successfully Load VGGNet Model!')

        for epoch in range(epochs):
            begin = time.time()
            # 生成并打乱训练集的顺序
            index = np.arange(len(train_sample))
            random.shuffle(index)

            # batch size为200 训练集前400000训练 后100000验证
            for i in range(0, train_num, batch_size):
                train_batch_sample = list()
                train_batch_label = list()
                for j in range(0, batch_size):
                    train_batch_sample.append(train_sample[index[i + j]] / 255)
                    # train_batch_sample.append(cv.resize(train_sample[index[i + j]], (224, 224)) / 255)
                    train_batch_label.append(train_label[index[i + j]])
                train_batch_label = get_onehot_label(train_batch_label, class_num=10)
                # 训练
                a, b, c = session.run([optimizer, loss, val_acc], feed_dict={
                    x: train_batch_sample,
                    y: train_batch_label,
                    learning_rate: learning_rate_val
                })
                if (i % 10000 == 0):
                    print('train loss: {}, train acc: {}'.format(b, c))

            acc = 0
            for i in range(train_num, train_num+val_num, batch_size):
                val_batch_sample = list()
                val_batch_label = list()
                for j in range(0, batch_size):
                    val_batch_sample.append(train_sample[index[i + j]] / 255)
                    # val_batch_sample.append(cv.resize(train_sample[index[i + j]], (224, 224)) / 255)
                    val_batch_label.append(train_label[index[i + j]])
                val_batch_label = get_onehot_label(val_batch_label, class_num=10)
                acc += session.run(val_acc, feed_dict={
                    x: val_batch_sample,
                    y: val_batch_label,
                    learning_rate: learning_rate_val
                })
            # 输出 一共50次验证数据相加
            val_acc_list.append(acc)
            print('====================Epoch {}: validation acc: {}, spend time: {}s==================='.format(epoch + 1, acc / 500, time.time() - begin))
            # 每训练5epoch 保存模型
            if (epoch + 1) % 5 == 0:
                # 保存模型
                saver.save(session, "model/vggnet/vggnet.ckpt")

        # 保存模型
        saver.save(session, "model/vggnet/vggnet.ckpt")
        plot_acc()

def test():
    # 加载训练集
    test_sample, test_label = get_test_dataset()

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32 ,3], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

    # 建立网络
    output = VGGNet_slim(x, 10)
    # 预测
    predict_output = tf.argmax(output, 1)
    ground_truth = tf.argmax(y, 1)

    # 定义准确率
    val_acc = tf.reduce_mean(tf.cast(tf.equal(predict_output, ground_truth), tf.float32))
    acc = 0
    # saver持久化 保存模型
    saver = tf.train.Saver()

    batch_size = 200
    test_num = len(test_sample)
    with tf.Session() as session:
        # 加载模型
        saver.restore(session, 'model/vggnet/vggnet.ckpt')

        for i in range(0, test_num, batch_size):
            test_batch_sample = list()
            test_batch_label = list()
            for j in range(batch_size):
                test_batch_sample.append(test_sample[i + j] / 225)
                # test_batch_sample.append(cv.resize(test_sample[i + j], (224, 224)) / 255)
                test_batch_label.append(test_label[i + j])
            test_batch_label = get_onehot_label(test_batch_label, class_num=10)
            a, b = session.run([predict_output, ground_truth], feed_dict={
                x: test_batch_sample,
                y: test_batch_label
            })
            for k in range(batch_size):
                if a[k] == b[k]:
                    acc += 1
        print('test acc is {}%'.format(acc / test_num * 100))

        class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        image_path = os.getcwd() + "/../DL_HotNet_Tensorflow/net/"
        for i in range(1, 11):
            name = image_path + str(i) + '.jpg'
            img = cv.imread(name)
            img = cv.resize(img, (32, 32)) / 255
            img = np.array([img])
            predict = session.run(predict_output, feed_dict={x: img})
            print('{} image detect result is : {}'.format(class_list[i - 1], class_list[predict[0]]))

# 画出损失函数的趋势图
def plot_acc():
    plt.title('Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('val acc')
    plt.plot(range(0, len(val_acc_list)), val_acc_list)
    plt.show()

if __name__ == '__main__':
    os.chdir('/home/yipeng/workspace/python/tensorflow_samples')
    # train()
    test()
