import os
import time
import random

import cv2 as cv
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from yolo import YOLO
from voc_data_load import PascalVOC
from tensorflow_samples.cifar_data_load import get_training_dataset, get_test_dataset
from tensorflow_samples.common import get_onehot_label

class Solver(object):
    def __init__(self, model, datasets=None):
        self.model = model
        self.datasets = datasets
        self.log_dir = cfg.log_dir
        self.detect_model_path = cfg.detect_model_path
        self.pre_trian_model_path = cfg.pre_trian_model_path

        self.batch_size = cfg.batch_size
        self.learning_rate_val = 0.001
        self.learning_rate = tf.placeholder(tf.float32)
        self.momentum = cfg.momentum
        self.epoch = cfg.epoch
        self.pre_train = True

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.model.total_loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        now_time = time.strftime("%Y-%m-%d %H-%M", time.localtime())
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, now_time), self.session.graph)
        self.saver = tf.train.Saver(max_to_keep=4)

    def train_classify(self):
        train_sample, train_label = get_training_dataset()
        model = tf.train.get_checkpoint_state(self.pre_trian_model_path)
        if model and model.model_checkpoint_path:
            self.saver.restore(self.session, model.model_checkpoint_path)
            print('Successfully Load Yolo PreTrain Model!')

        for epoch in range(self.epoch):
            # 生成并打乱训练集的顺序
            index = np.arange(len(train_sample))
            random.shuffle(index)
            for i in tqdm(range(0, len(train_sample), self.batch_size)):
                train_batch_sample = list()
                train_batch_label = list()
                for j in range(0, self.batch_size):
                    train_batch_sample.append(cv.resize(train_sample[index[i + j]], (224, 224)) / 255)
                    train_batch_label.append(train_label[index[i + j]])
                train_batch_label = get_onehot_label(train_batch_label, class_num=10)
                # 训练
                a, b, c = self.session.run([self.optimizer, self.model.total_loss, self.model.accuracy], feed_dict={
                    self.model.input: train_batch_sample,
                    self.model.label: train_batch_label,
                    self.model.is_training: True,
                    self.learning_rate: self.learning_rate_val
                })
                if (i % 10000 == 0):
                    print('Epoch {}: , train loss: {}, train acc: {}%'.format(epoch + 1, b, c))

            # 每训练1轮 测试一次
            test_sample, test_label = get_test_dataset()
            test_acc = 0
            test_num = len(test_sample)
            test_iter_count = test_num / self.batch_size
            for i in tqdm(range(0, test_num, self.batch_size)):
                test_batch_sample = list()
                test_batch_label = list()
                for j in range(0, self.batch_size):
                    test_batch_sample.append(cv.resize(test_sample[i + j], (224, 224)) / 255)
                    test_batch_label.append(test_label[i + j])
                test_batch_label = get_onehot_label(test_batch_label, class_num=10)
                acc = self.session.run(self.model.accuracy, feed_dict={
                    self.model.input: test_batch_sample,
                    self.model.label: test_batch_label,
                    self.model.is_training: False,
                })
                test_acc += acc
                print(acc)
            # 输出 一共50次验证数据相加
            # test_acc_list.append(test_acc)
            print('====================Epoch {}: test accuracy is : {}%==================='.format(epoch + 1,
                                                                                          test_acc / test_iter_count))
            # 每训练5epoch 保存模型
            if (epoch + 1) % 5 == 0:
                # 保存模型
                self.saver.save(self.session, self.pre_trian_model_path + "yolov1.ckpt")

        # 保存模型
        self.saver.save(self.session, self.pre_trian_model_path + "yolov1.ckpt")
        print('yolov1 pre train network task was run over')

    def train_detector(self):
        self.set_detector_params()
        iter_step = int(len(self.train_labels) / self.batch_size)

        for epoch in range(self.epoch):
            for step in tqdm(range(iter_step)):
                train_batch_sample, train_batch_label = self.datasets.next_batch(self.train_labels, self.batch_size)
                _,train_loss = self.session.run([self.optimizer, self.model.total_loss], feed_dict={
                    self.model.input: train_batch_sample,
                    self.model.label: train_batch_label,
                    self.model.is_training: True,
                    self.learning_rate: self.learning_rate_val
                })

                if step % 100 == 0:
                    print('Epoch {}: , train loss: {}'.format(epoch + 1, train_loss))

            # test sets sum :4962
            test_loss = 0.
            test_iter = 10  # 取10个批次求均值
            for _ in range(test_iter):
                test_batch_sample, test_batch_label = self.datasets.next_batch(self.test_labels, self.batch_size)
                test_loss += self.sess.run(self.model.total_loss, feed_dict={
                    self.model.input: test_batch_sample,
                    self.model.label: test_batch_label,
                    self.model.is_training: False,
                })

                mean_loss = sum_loss / test_iter
                print('====================Epoch {} , step {} , test loss is : {}===================='.format(epoch + 1, step, mean_loss))

            self.saver.save(self.session, self.detect_model_path + 'yolov1.ckpt')
            print('save yolov1 model successful')

    def set_detector_params(self):
        self.train_labels = self.datasets.preprocess('train')
        self.test_labels = self.datasets.preprocess('test')
        if self.pre_train:
            self.load_detect_model()
        else:
            self.load_pre_train_model()

    def load_pre_train_model(self):
        model_variables = slim.get_model_variables()
        model_file = tf.train.latest_checkpoint(self.pre_trian_model_path)
        reader = tf.train.NewCheckpointReader(model_file)
        model_vars = reader.get_variable_to_shape_map()
        exclude = ['yolov1/classify_fc1/weights', 'yolov1/classify_fc1/biases']

        vars_restore_map = {}
        for var in model_variables:
            if var.op.name in model_vars and var.op.name not in exclude:
                vars_restore_map[var.op.name] = var

        self.saver = tf.train.Saver(vars_restore_map, max_to_keep=4)
        self.saver.restore(self.session, model_file)
        self.saver = tf.train.Saver(var_list=model_variables, max_to_keep=4)

        print('Successfully Load Yolo PreTrain Model!')

    def load_detect_model(self):
        model_variables = slim.get_model_variables()
        self.saver = tf.train.Saver(model_variables, max_to_keep=4)

        model = tf.train.get_checkpoint_state(self.detect_model_path)
        if model and model.model_checkpoint_path:
            self.saver.restore(self.session, model.model_checkpoint_path)
            print('Successfully Load Yolo Model!')

def train_yolo_classify():
    yolov1 = YOLO(is_pre_training=True)

    sovler = Solver(model=yolov1)
    print('start yolov1 pre train')
    sovler.train_classify()

def train_yolo_detector():
    yolov1 = YOLO(is_pre_training=False)

    pascal_voc = PascalVOC()
    sovler = Solver(model=yolov1, datasets=pascal_voc)
    print('start yolov1 train')
    sovler.train_detector()

if __name__ == '__main__':
    train_yolo_detector()