import os
import time
import random

import cv2 as cv
import numpy as np
import tensorflow as tf

from cifar_data_load import get_training_dataset, get_test_dataset
from common import get_onehot_label

def AlexNet(input, p=0.5):
    # 定义卷积层conv1
    with tf.name_scope('conv1'):
        # 第一层卷积层 filter 64 3 11 11 步长4
        conv1_kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                   stddev=1e-1, name='weights'))
        conv1 = tf.nn.conv2d(input, conv1_kernel, [1, 4, 4, 1], padding='SAME')
        conv1_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]),
                                               trainable=True, name='biases')
        conv1_bias = tf.nn.bias_add(conv1, conv1_biases)
        conv1 = tf.nn.relu(conv1_bias, name='conv1')

        # local response normalization LRN局部响应归一化
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        # 最大池化层 filter 3 3 步长2
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')

    # 定义卷积层conv2
    with tf.name_scope('conv2'):
        # 第二层卷积层 filter 192 64 5 5 步长1
        conv2_kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                   stddev=1e-1, name='weights'))
        conv2 = tf.nn.conv2d(pool1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        conv2_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[192]),
                                   trainable=True, name='biases')
        conv2_bias = tf.nn.bias_add(conv2, conv2_biases)
        conv2 = tf.nn.relu(conv2_bias, name='conv2')

        # local response normalization LRN局部响应归一化
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        # 最大池化层 filter 3 3 步长2
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')

    # 定义卷积层conv3
    with tf.name_scope('conv3'):
        # 第二层卷积层 filter 384 192 3 3 步长1
        conv3_kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32,
                                   stddev=1e-1, name='weights'))
        conv3 = tf.nn.conv2d(pool2, conv3_kernel, [1, 1, 1, 1], padding='SAME')
        conv3_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384]),
                                               trainable=True, name='biases')
        conv3_bias = tf.nn.bias_add(conv3, conv3_biases)
        conv3 = tf.nn.relu(conv3_bias, name='conv3')

    # 定义卷积层conv4
    with tf.name_scope('conv4'):
        # 第二层卷积层 filter 256 384 3 3 步长1
        conv4_kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,
                                   stddev=1e-1, name='weights'))
        conv4 = tf.nn.conv2d(conv3, conv4_kernel, [1, 1, 1, 1], padding='SAME')
        conv4_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]),
                                   trainable=True, name='biases')
        conv4_bias = tf.nn.bias_add(conv4, conv4_biases)
        conv4 = tf.nn.relu(conv4_bias, name='conv4')

    # 定义卷积层conv5
    with tf.name_scope('conv5'):
        # 第二层卷积层 filter 256 256 3 3 步长1
        conv5_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                   stddev=1e-1, name='weights'))
        conv5 = tf.nn.conv2d(conv4, conv5_kernel, [1, 1, 1, 1], padding='SAME')
        conv5_biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]),
                                         trainable=True, name='biases')
        conv5_bias = tf.nn.bias_add(conv5, conv5_biases)
        conv5 = tf.nn.relu(conv5_bias, name='conv5')

        # 最大池化层 filter 3 3 步长2
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool5')

        # 把值展开 reshape送入全连接层
        flatten = tf.reshape(pool5, [-1, 6*6*256])

    # 定义全连接层dense1
    with tf.name_scope('dense1'):
        # 输入9216 输出4096
        fc1_weights = tf.Variable(tf.truncated_normal([6*6*256, 4096], mean=0, stddev=0.01))
        fc1_biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                 trainable=True, name='biases')
        fc1 = tf.nn.sigmoid(tf.matmul(flatten, fc1_weights) + fc1_biases, name='dense1')
        dense1 = tf.nn.dropout(fc1, p)

    # 定义全连接层dense2
    with tf.name_scope('dense2'):
        # 输入4096 输出4096
        fc2_weights = tf.Variable(tf.truncated_normal([4096, 4096], mean=0, stddev=0.01))
        fc2_biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                 trainable=True, name='biases')
        fc2 = tf.nn.sigmoid(tf.matmul(dense1, fc2_weights) + fc2_biases, name='dense2')
        dense2 = tf.nn.dropout(fc2, p)

    # 定义全连接层dense3
    with tf.name_scope('output'):
        # 输入4096 输出10
        fc3_weights = tf.Variable(tf.truncated_normal([4096, 10], mean=0, stddev=0.01))
        fc3_biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32),
                                 trainable=True, name='biases')
        dense3 = tf.matmul(dense2, fc3_weights) + fc3_biases

    return dense3

# 训练
def train():
    # 加载训练集
    train_sample, train_label = get_training_dataset()

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224 ,3], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

    # 建立网络
    output = AlexNet(x, 0.5)
    # 定义训练的loss函数 得到一个batch的平均值
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.09).minimize(loss)
    # 定义准确率
    val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    # saver持久化 保存模型
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    epochs = 40
    with tf.Session() as session:
        # 编译 静态图
        session.run(init)

        # 加载模型
        model = tf.train.get_checkpoint_state("./model/alexnet")
        if model and model.model_checkpoint_path:
            # saver.restore(session, model.model_checkpoint_path)
            print('Successfully Load AlexNet Model!')

        for epoch in range(epochs):
            begin = time.time()
            # 生成并打乱训练集的顺序
            index = np.arange(50000)
            random.shuffle(index)

            # batch size为200 训练集前40000训练 后10000验证
            for i in range(0, 0+40000, 200):
                train_batch_sample = list()
                train_batch_label = list()
                for j in range(0, 200):
                    # train_batch_sample.append(train_sample[index[i + j]] / 255)
                    train_batch_sample.append(cv.resize(train_sample[index[i + j]], (224, 224)) / 255)
                    train_batch_label.append(train_label[index[i + j]])
                train_batch_label = get_onehot_label(train_batch_label, class_num=10)
                # 训练
                a, b = session.run([optimizer, loss], feed_dict={
                    x: train_batch_sample,
                    y: train_batch_label,
                })
                print('train loss: ', b)

            acc = 0
            for i in range(40000, 40000+10000, 200):
                val_batch_sample = list()
                val_batch_label = list()
                for j in range(0, 200):
                    # val_batch_sample.append(train_sample[index[i + j]].astype(np.uint8) / 225)
                    val_batch_sample.append(cv.resize(train_sample[index[i + j]].astype(np.uint8), (224, 224)) / 255)
                    val_batch_label.append(train_label[index[i + j]])
                val_batch_label = get_onehot_label(val_batch_label, class_num=10)
                acc += session.run(val_acc, feed_dict={
                    x: val_batch_sample,
                    y: val_batch_label
                })
            # 输出 一共50次验证数据相加
            print('====================Epoch {}: validation acc: {}, spend time: {}s==================='.format(epoch + 1, acc / 50, time.time() - begin))
            # 每训练5epoch 保存模型
            if (epoch + 1) % 5 == 0:
                pass
                # 保存模型
                saver.save(session, "model/alexnet/alexnet.ckpt")

        # 保存模型
        saver.save(session, "model/alexnet/alexnet.ckpt")

def test():
    # 加载训练集
    test_sample, test_label = get_test_dataset()

    # 定义tf占位符 输入输出格式
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224 ,3], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

    # 建立网络
    output = AlexNet(x, 0.5)
    # 预测
    predict_output = tf.argmax(output, 1)
    ground_truth = tf.argmax(y, 1)

    # 定义准确率
    val_acc = tf.reduce_mean(tf.cast(tf.equal(predict_output, ground_truth), tf.float32))
    acc = 0
    # saver持久化 保存模型
    saver = tf.train.Saver()

    with tf.Session() as session:
        # 加载模型
        saver.restore(session, 'model/alexnet/alexnet.ckpt')

        for i in range(0, 10000, 200):
            test_batch_sample = list()
            test_batch_label = list()
            for j in range(200):
                test_batch_sample.append(cv.resize(test_sample[i + j], (224, 224)) / 255)
                test_batch_label.append(test_label[i + j])
            test_batch_label = get_onehot_label(test_batch_label, class_num=10)
            a, b = session.run([predict_output, ground_truth], feed_dict={
                x: test_batch_sample,
                y: test_batch_label
            })
            for k in range(200):
                if a[k] == b[k]:
                    acc += 1
        print('test acc is {}%'.format(acc / 10000 * 100))

        class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        image_path = os.getcwd() + "/../DL_HotNet_Tensorflow/net/"
        for i in range(1, 11):
            name = image_path + str(i) + '.jpg'
            img = cv.imread(name)
            img = cv.resize(img, (32, 32))
            img = cv.resize(img, (224, 224)) / 255
            img = np.array([img])
            predict = session.run(predict_output, feed_dict={x: img})
            print('{} image detect result is : {}'.format(class_list[i - 1], class_list[predict[0]]))

if __name__ == '__main__':
    os.chdir('/home/yipeng/workspace/python/tensorflow_samples')
    train()
    # test()