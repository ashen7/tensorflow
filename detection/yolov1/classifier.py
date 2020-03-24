import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import config as cfg
from yolo import YOLO

class Classifier(object):
    def __init__(self, net):
        self.net = net
        self.pre_trian_model_path = cfg.pre_trian_model_path

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=4)

    def test(self):
        # self.load_model()
        self.load_part_model()
        class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                    'truck']
        path = r'/home/yipeng/workspace/python/DL_HotNet_Tensorflow/net/'
        for i in range(1, 11):
            name = str(i) + '.jpg'
            img = cv.imread(path + name)
            img = cv.resize(img, (224, 224))
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img / 255.
            img = np.array([img])
            result = self.session.run(self.net.output, feed_dict={self.net.input: img, self.net.is_training: False})

            # print(res)
            print('{}.jpg detect result is : '.format(str(i)) + class_list[np.argmax(result)])

    def load_model(self):
        model_file = tf.train.latest_checkpoint(self.pre_trian_model_path)
        self.saver.restore(self.sess, model_file)

    def load_part_model(self):
        model_vars = slim.get_model_variables()
        model_file = tf.train.latest_checkpoint(self.pre_trian_model_path)
        reader = tf.train.NewCheckpointReader(model_file)
        dic = reader.get_variable_to_shape_map()
        print(dic)
        # for var in dic:
        #     print(self.sess.run(var))
        exclude = ['yolov1/classify_fc1/weights', 'yolov1/classify_fc1/biases']
        # vars_to_restore = slim.get_variables_to_restore(exclude=exclude)
        # self.saver = tf.train.Saver(vars_to_restore)

        vars_restore_map = {}
        for var in model_vars:
            if var.op.name in dic and var.op.name not in exclude:
                vars_restore_map[var.op.name] = var

        self.saver = tf.train.Saver(vars_restore_map)
        self.saver.restore(self.session, model_file)

if __name__ == '__main__':
    yolov1 = YOLO(is_pre_training=True)
    classifier = Classifier(yolov1)
    classifier.test()