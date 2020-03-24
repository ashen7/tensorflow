import os

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import cv2 as cv

import config as cfg

class PascalVOC(object):
    def __init__(self):
        self.classes = cfg.voc07_class
        self.data_path = cfg.voc07_data_path
        self.image_size = cfg.image_size
        self.cell_size = cfg.cell_size
        self.phase = None
        self.labels = None
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self.current_index = 0
        self.batch_size = cfg.batch_size
        self.epoch = 0

    def preprocess(self, phase):
        # phase 阶段 （训练 或 测试）
        self.phase = phase
        labels = self.load_labels()
        np.random.shuffle(labels)
        self.labels = labels

        return labels

    def load_labels(self):
        if self.phase == 'train':
            file_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        elif self.phase == 'test':
            file_name = os.path.join(self.data_path, 'ImageSets','Main','test.txt')
        else:
            raise NameError('没有此状态')
        with open(file_name, 'r') as f:
            # 得到所有训练/测试 图片的索引
            self.image_index = [image_index.strip() for image_index in f.readlines()]

        labels = list()
        # 得到每个标签(图片)的标注信息
        for index in self.image_index:
            label, objs_num = self.load_label_annotation(index)
            if objs_num == 0:
                continue
            img_name = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            labels.append({'img_name': img_name, 'label': label})
        # 返回一个列表 每个值是一个字典 有图像的路径和对应label的值
        return labels

    # 导入PASCAL的annotation 标注  也就是gt
    def load_label_annotation(self, index):
        # 7 7个网格 每个网格有25个值 前5个值：中心cell是否存在对象 x,y,w,h 后20个值是类别的概率one-hot
        label = np.zeros((self.cell_size, self.cell_size, 25))
        # 标注文件xml格式
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        # etree 解析xml文件
        tree = ET.parse(filename)
        objs = tree.findall('object')
        size = tree.find('size')
        # 水平和垂直方向的缩放 原图对于要放入神经网络的图(448,448)的比例
        h_ratio = 1.0 * self.image_size / int(size.find('height').text)
        w_ratio = 1.0 * self.image_size / int(size.find('width').text)
        for obj in objs:
            bbox = obj.find('bndbox')
            # 乘以比例后的gt bbox相对位置 x y w h
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            # 该类别的索引
            class_index = self.class_to_idx[obj.find('name').text.lower().strip()]

            # x,y为该类别gt bbox的中心点坐标  w,h 为bbox的宽高
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]

            # 根据x,y 判断中点在7*7的第几个cell中
            x_index = int(boxes[0] * self.cell_size / self.image_size)
            y_index = int(boxes[1] * self.cell_size / self.image_size)
            # 　如果第(y_index,x_index)格子中已经存在对象就忽略了 一个cell最多只负责一个对象的预测
            if label[y_index, x_index, 0] == 1:
                continue
            label[y_index, x_index, 0] = 1  # 对应(y_ind,x_ind)的cell中存在对象，置１
            label[y_index, x_index, 1:5] = boxes  # 储存gt bbox的中心点坐标x,y, bbox的宽高w,h
            label[y_index, x_index, 5 + class_index] = 1  # 对应one-hot形式类设置为1

        return label, len(objs)

    def image_read(self, img_name):  # 读取图片
        image = cv.imread(img_name)
        image = cv.resize(image, (self.image_size, self.image_size))  # resize大小
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0
        return image

    def next_batch(self, labels, batch_size):
        batch_sample = np.zeros((batch_size, self.image_size, self.image_size, 3))
        batch_label = np.zeros((batch_size, self.cell_size, self.cell_size, 25))
        count = 0

        while count < batch_size:
            img_name = labels[self.current_index]['img_name']
            batch_sample[count,:,:,:] = self.image_read(img_name)
            batch_label[count,:,:,:] = labels[self.current_index]['label']
            count += 1
            self.current_index += 1
            if self.current_index >= len(labels):
                np.random.shuffle(labels)
                self.current_index = 0
                self.epoch += 1

        return batch_sample, batch_label

if __name__ == '__main__':
    pascal = PascalVOC()
    train_labels = pascal.preprocess('train')

    batch_sample, batch_label = pascal.next_batch(train_labels, 32)
    print(batch_sample.shape, batch_label.shape)
