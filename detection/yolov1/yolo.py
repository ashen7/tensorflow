import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import config as cfg
from voc_data_load import PascalVOC

# YOLO的主干 神经网络
class YOLO(object):
    def __init__(self, is_pre_training=False, is_training=True):
        '''
        :param is_pre_training:  预训练(目标分类)
        :param is_training:  训练(目标检测)
        output_size: 检测结果大小 7*7个网格 每个网格预测2个边界框bbox
        每个边界框包含5个值 0/1 x,y,w,h  第一个值是该bbox是否是对象的中心cell 后面20个是one-hot类别的概率
        '''
        self.detect_classes = cfg.voc07_class
        self.detect_class_num = len(self.detect_classes)
        self.classify_class_num = cfg.pre_train_class_num

        self.image_size = cfg.image_size
        self.cell_size = cfg.cell_size
        self.per_cell_bbox = cfg.per_cell_bbox
        self.bound_box1 = self.cell_size * self.cell_size * self.detect_class_num
        self.bound_box2 = self.bound_box1 + self.cell_size * self.cell_size * self.per_cell_bbox
        self.output_size = (self.cell_size * self.cell_size) * (5 * self.per_cell_bbox + self.detect_class_num)

        self.scale = 1.0 * self.image_size / self.cell_size
        self.object_scale = cfg.object_confident_scale
        self.no_object_scale = cfg.no_object_confident_scale
        self.class_scale = cfg.class_scale
        self.coord_scale = cfg.coord_scale

        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.epoch = cfg.epoch
        self.keep_prob = cfg.keep_prob
        self.is_pre_training = is_pre_training
        self.bn_params = cfg.batch_norm_params

        self.offset = np.transpose(
            np.reshape(
                np.array(
                    [np.arange(self.cell_size)] * self.cell_size * self.per_cell_bbox
                ), (self.per_cell_bbox, self.cell_size, self.cell_size)
            ), (1, 2, 0)
        )

        # 得到网络输出
        self.is_training = tf.placeholder(tf.bool)
        if self.is_pre_training:
            self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        else:
            self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='input')
        self.output = self.build_yolo_net(self.input, is_training=self.is_training)

        # 得到loss 和精度(正确率)
        if is_training:
            if self.is_pre_training:
                print('=========================yolov1预训练网络(目标分类)=========================')
                self.label = tf.placeholder(tf.float32, [None, self.classify_class_num], name='label')
                self.classify_loss(self.output, self.label)
                self.total_loss = tf.losses.get_total_loss()
                self.accuracy = self.classify_evaluate(self.output, self.label)
                print(self.accuracy)
            else:
                print('=========================yolov1识别网络(目标检测)===========================')
                self.label = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.detect_class_num], name='label')
                self.detect_loss(self.output, self.label)
                self.total_loss = tf.losses.get_total_loss()

    # 构建yolo网络 网络借鉴了GoogLeNet的思想 1x1后3x3的卷积 代替了inception卷积结构
    def build_yolo_net(self, input, is_training, scope='yolov1'):
        with tf.variable_scope(scope):
            # 卷积层 和全连接层 权重L2正则化
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(0.00004)):
                # 卷积层 权值初始化 使用BN归一化每层输入 激活函数是leaky_relu
                with slim.arg_scope([slim.conv2d],
                                    weights_initializer=slim.xavier_initializer(),
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=self.bn_params,
                                    activation_fn=slim.nn.leaky_relu):
                    # 卷积层 最大池化层 设置默认补0填充 步长1
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                         padding='SAME', stride=1):
                        # BN和dropout 设置默认的训练/测试 模式
                        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                            # 64个特征图 filter 7*7 步长2
                            yolo = slim.conv2d(input, 64, [7, 7], stride=2, scope='layer1')
                            yolo = slim.max_pool2d(yolo, [2, 2], stride=2, scope='pool1')

                            # 192个特征图 filter 3*3 步长1
                            yolo = slim.conv2d(yolo, 192, [3, 3], stride=1, scope='layer2')
                            yolo = slim.max_pool2d(yolo, [2, 2], stride=2, scope='pool2')

                            # 2个 1x1卷积 后接3x3卷积
                            yolo = slim.conv2d(yolo, 128, [1, 1], stride=1, scope='layer3_1')
                            yolo = slim.conv2d(yolo, 256, [3, 3], stride=1, scope='layer3_2')
                            yolo = slim.conv2d(yolo, 256, [1, 1], stride=1, scope='layer3_3')
                            yolo = slim.conv2d(yolo, 512, [3, 3], stride=1, scope='layer3_4')
                            yolo = slim.max_pool2d(yolo, [2, 2], stride=2, scope='pool3')

                            # 5个  1x1卷积 后接3x3卷积
                            yolo = slim.conv2d(yolo, 256, [1, 1], stride=1, scope='layer4_1')
                            yolo = slim.conv2d(yolo, 512, [3, 3], stride=1, scope='layer4_2')
                            yolo = slim.conv2d(yolo, 256, [1, 1], stride=1, scope='layer4_3')
                            yolo = slim.conv2d(yolo, 512, [3, 3], stride=1, scope='layer4_4')
                            yolo = slim.conv2d(yolo, 256, [1, 1], stride=1, scope='layer4_5')
                            yolo = slim.conv2d(yolo, 512, [3, 3], stride=1, scope='layer4_6')
                            yolo = slim.conv2d(yolo, 256, [1, 1], stride=1, scope='layer4_7')
                            yolo = slim.conv2d(yolo, 512, [3, 3], stride=1, scope='layer4_8')
                            yolo = slim.conv2d(yolo, 512, [1, 1], stride=1, scope='layer4_9')
                            yolo = slim.conv2d(yolo, 1024, [3, 3], stride=1, scope='layer4_10')
                            yolo = slim.max_pool2d(yolo, [2, 2], stride=2, scope='pool4')

                            # 1x1卷积 后接3x3卷积
                            yolo = slim.conv2d(yolo, 512, [1, 1], stride=1, scope='layer5_1')
                            yolo = slim.conv2d(yolo, 1024, [3, 3], stride=1, scope='layer5_2')
                            yolo = slim.conv2d(yolo, 512, [1, 1], stride=1, scope='layer5_3')
                            yolo = slim.conv2d(yolo, 1024, [3, 3], stride=1, scope='layer5_4')

                            # 如果是预训练 经过一个平均池化和一个全连接得到分类概率
                            if self.is_pre_training:
                                yolo = slim.avg_pool2d(yolo, [7, 7], stride=1, padding='VALID', scope='classify_avg_pool5')
                                yolo = slim.flatten(yolo)
                                yolo = slim.fully_connected(yolo, self.classify_class_num,
                                                            activation_fn=slim.nn.leaky_relu, scope='classify_fc1')
                                return yolo

                            # 如果是检测 多加4层3x3卷积后 接3层全连接层
                            yolo = slim.conv2d(yolo, 1024, [3, 3], stride=1, scope='layer5_5')
                            yolo = slim.conv2d(yolo, 1024, [3, 3], stride=2, scope='layer5_6')

                            yolo = slim.conv2d(yolo, 1024, [3, 3], stride=1, scope='layer6_1')
                            yolo = slim.conv2d(yolo, 1024, [3, 3], stride=1, scope='layer6_2')

                            yolo = slim.flatten(yolo)
                            yolo = slim.fully_connected(yolo, 1024, activation_fn=slim.nn.leaky_relu, scope='fc1')
                            yolo = slim.dropout(yolo, keep_prob=self.keep_prob)
                            yolo = slim.fully_connected(yolo, 4096, activation_fn=slim.nn.leaky_relu, scope='fc2')
                            yolo = slim.dropout(yolo, keep_prob=self.keep_prob)
                            yolo = slim.fully_connected(yolo, self.output_size, activation_fn=None, scope='fc3')
        return yolo

    # 得到分类的loss(预训练)
    def classify_loss(self, output, label):
        with tf.name_scope('classify_loss') as scope:
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label)
            loss = tf.reduce_mean(loss)
            tf.losses.add_loss(loss)

    # 分类的评估(预训练)
    def classify_evaluate(self, output, label):
        with tf.name_scope('classify_evaluate') as scope:
            correct = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    # 得到检测的loss
    def detect_loss(self, output, label, scope='detect_loss'):
        '''
        @:param output shape->[N,7x7x30]
        @:param label  shape->[N,7,7,25]  <==>[N,h方向,w方向,25] ==>[N,7,7,25(1:是否负责检测,2-5:坐标,6-25:类别one-hot)]
        '''
        with tf.variable_scope(scope):
            # 前20个是每个cell预测的20个类别
            predict_classes = tf.reshape(output[:, :self.bound_box1], [-1, 7, 7, 20])
            # 后2个是每个cell预测2个bbox的置信度
            predict_confident = tf.reshape(output[:, self.bound_box1:self.bound_box2], [-1, 7, 7, 2])
            # 最后8个是每个cell的2个bbox的x, y, w, h
            predict_boxes = tf.reshape(output[:, self.bound_box2:], [-1, 7, 7, 2, 4])

            # 标签类别
            label_classes = label[:, :, :, 5:]
            # label置信度 7*7里值为1的就是负责一个对象检测的中心cell 为0就不是中心cell
            label_confident = tf.reshape(label[:, :, :, 0], [-1, 7, 7, 1])
            # tile在维度上添加x倍个值
            # 标签坐标 由于预测是2个 因此需要将标签也变成2个 同时对坐标进行yolo形式归一化
            label_boxes = tf.reshape(label[:, :, :, 1:5], [-1, 7, 7, 1, 4])
            label_boxes = tf.tile(label_boxes, [1, 1, 1, 2, 1]) / self.image_size

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, 7, 7, 2])
            offset = tf.tile(offset, [tf.shape(label_boxes)[0], 1, 1, 1])
            # stack和concat一样 数组拼接
            predict_boxes_tran = tf.stack([
                1. * (predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                1. * (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                tf.square(predict_boxes[:, :, :, :, 2]),
                tf.square(predict_boxes[:, :, :, :, 3])
            ], axis=-1)
            # predict_boxes_tran = tf.transpose(predict_boxes_tran,[1,2,3,4,0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, label_boxes)
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * label_confident
            no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
            boxes_tran = tf.stack([
                1. * label_boxes[:, :, :, :, 0] * 7 - offset,
                1. * label_boxes[:, :, :, :, 1] * 7 - tf.transpose(offset, (0, 2, 1, 3)),
                tf.sqrt(label_boxes[:, :, :, :, 2]),
                tf.sqrt(label_boxes[:, :, :, :, 3])
            ], axis=-1)

            # 类别损失
            class_delta = label_confident * (predict_classes - label_classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                        name='class_loss') * self.class_scale

            # 对象损失
            object_delta = object_mask * (predict_confident - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                         name='object_loss') * self.object_scale

            # 无对象损失
            no_object_delta = no_object_mask * predict_confident
            no_object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(no_object_delta), axis=[1, 2, 3]),
                                            name='no_object_loss') * self.no_object_scale

            # 坐标损失
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                        name='coord_loss') * self.coord_scale
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(no_object_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', no_object_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
               Args:
                 boxes1: 4-D tensor [cell_size, cell_size, per_cell_bbox, 4]  ====> (x_center, y_center, w, h)
                 boxes2: 1-D tensor [cell_size, cell_size, per_cell_bbox, 4] ===> (x_center, y_center, w, h)
               Return:
                 iou: 3-D tensor [cell_size, cell_size, per_cell_bbox]
        """
        with tf.variable_scope(scope):
            # 数组拼接 batch_size 7 7 2 4
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0], axis=-1)
            # boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0], axis=-1)
            # boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # 取两个bbox的x,y中最大的x,y做左上坐标
            left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            # 取两个bbox的w, h中最小的w, h做右下坐标
            right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            intersection = tf.maximum(0.0, right_down - left_up)
            # 交集
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # 第一个bbox 和 第二个bbox的面积
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                      (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                      (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            # 并集 = 两个面积 - 交集
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        # 返回交并比
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

if __name__ == '__main__':
    # yolov1 = YOLO(is_pre_training=True)
    yolov1 = YOLO(is_pre_training=False)
    output = yolov1.output

    pascal = PascalVOC()
    train_labels = pascal.preprocess('train')
    batch_sample, batch_label = pascal.next_batch(train_labels, 5)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # val = session.run(tf.random_uniform((5, 224, 224, 3)))
        a = session.run(output, feed_dict={
            yolov1.input: batch_sample, yolov1.label: batch_label, yolov1.is_training: True
        })
        print(a)
        print(session.run(tf.shape(a)))
