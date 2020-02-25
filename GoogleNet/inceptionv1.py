import tensorflow as tf
import tensorflow.contrib.slim as slim

def inception_module_v1(google_net, filter_num, scope):
    with tf.variable_scope(scope):
        # inception v1的第一部分 1 × 1的卷积
        with tf.variable_scope('component1'):
            component1 = slim.conv2d(google_net, filter_num[0], 1, scope='component1_conv1_1x1')
        # inception v1的第二部分  1 × 1的卷积  3 × 3的卷积
        with tf.variable_scope('component2'):
            component2 = slim.conv2d(google_net, filter_num[1], 1, scope='component2_conv1_1x1')
            component2 = slim.conv2d(component2, filter_num[2], 3, scope='component2_conv2_3x3')
        # inception v1的第三部分  1 × 1的卷积  5 × 5的卷积
        with tf.variable_scope('component3'):
            component3 = slim.conv2d(google_net, filter_num[3], 1, scope='component3_conv1_1x1')
            component3 = slim.conv2d(component3, filter_num[4], 5, scope='component3_conv2_5x5')
        # inception v1的第四部分  3 × 3的池化  1 × 1的卷积
        with tf.variable_scope('component4'):
            component4 = slim.max_pool2d(google_net, 3, scope='component4_pool1_3x3')
            component4 = slim.conv2d(component4, filter_num[5], 1, scope='component4_conv1_1x1')
        # 将inception v1的卷积结构每个部分的结果 通道加起来
        google_net = tf.concat([component1, component2, component3, component4], axis=3)

    return google_net

def GoogLeNet_slim_v1(input, class_num, is_train=False, keep_prob=0.4, spatital_squeeze=True):
    # reshape 成224 224 3
    with tf.name_scope('reshape'):
        google_net = tf.reshape(input, [-1, 224, 224, 3])

    # 卷积层和全连接层 权重初始化和L2正则
    with tf.variable_scope('GoogLeNet_V1'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                             weights_regularizer=slim.l2_regularizer(5e-4),
                             weights_initializer=slim.xavier_initializer()):
            # 卷积层 和最大池化层 平均池化层 补0填充默认方法 和步长1
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                 padding='SAME', stride=1):
                # 卷积 64个卷积核 filter 7*7 步长2  最大池化 filter 3*3 步长2 局部响应归一化
                google_net = slim.conv2d(google_net, 64, 7, stride=2, scope='layer1')
                google_net = slim.max_pool2d(google_net, 3, stride=2, scope='layer2')
                google_net = tf.nn.lrn(google_net)
                # 卷积 64个卷积核 filter 1*1 步长1 卷积 192个卷积核 filter 3*3 局部响应归一化 最大池化 filter 3*3 步长2
                google_net = slim.conv2d(google_net, 64, 1, scope='layer3')
                google_net = slim.conv2d(google_net, 192, 3, scope='layer4')
                google_net = tf.nn.lrn(google_net)
                google_net = slim.max_pool2d(google_net, 3, stride=2, scope='layer5')
                # inception v1卷积组合结构
                google_net = inception_module_v1(google_net, filter_num=[64, 96, 128, 16, 32, 32], scope='layer6_inception_3a')
                google_net = inception_module_v1(google_net, filter_num=[128, 128, 192, 32, 96, 64], scope='layer8_inception_3b')
                google_net = slim.max_pool2d(google_net, 3, stride=2, scope='layer10')
                google_net = inception_module_v1(google_net, filter_num=[192, 96, 208, 16, 48, 64], scope='layer11_inception_4a')
                google_net = inception_module_v1(google_net, filter_num=[160, 112, 224, 24, 64, 64], scope='layer13_inception_4b')

                # 辅助分类器1 加入总结果权重*0.3
                auxiliary_classifier_1 = google_net
                google_net = inception_module_v1(google_net, filter_num=[128, 128, 256, 24, 64, 64], scope='layer15_inception_4c')
                google_net = inception_module_v1(google_net, filter_num=[112, 144, 288, 32, 64, 64], scope='layer17_inception_4d')
                # 辅助分类器2 加入总结果权重*0.3
                auxiliary_classifier_2 = google_net
                google_net = inception_module_v1(google_net, filter_num=[256, 160, 320, 32, 128, 128], scope='layer19_inception_4e')
                google_net = slim.max_pool2d(google_net, 3, stride=2, scope='layer21')
                google_net = inception_module_v1(google_net, filter_num=[256, 160, 320, 32, 128, 128], scope='layer22_inception_5a')
                google_net = inception_module_v1(google_net, filter_num=[384, 192, 384, 48, 128, 128], scope='layer24_inception_5b')

                # 用平均池化层 代替了全连接层 filter 7*7 步长1
                google_net = slim.avg_pool2d(google_net, 7, stride=1, padding='VALID', scope='layer26')
                # 不用全连接层 但依然使用dropout
                google_net = slim.dropout(google_net, keep_prob=keep_prob, scope='dropout')
                # 输出层 全连接层 不要激活函数和归一化
                google_net = slim.conv2d(google_net, class_num, 1, activation_fn=None, normalizer_fn=None, scope='layer27')
                if spatital_squeeze:
                    # 删除第2维和第三维 维度是1的维度
                    google_net = tf.squeeze(google_net, [1, 2], name='squeeze')
                # softmax
                google_net = slim.softmax(google_net, scope='softmax3')

                # 训练
                if is_train:
                    # 辅助分类器1 平均池化 filter 5*5 步长3 卷积 卷积核128个 filter 1*1 步长1 全连接层输出节点1024 输出层节点class_num
                    auxiliary_classifier_1 = slim.avg_pool2d(auxiliary_classifier_1, 5, padding='VALID', stride=3, scope='auxiliary1_avgpool')
                    auxiliary_classifier_1 = slim.conv2d(auxiliary_classifier_1, 128, 1, scope='auxiliary1_conv_1x1')
                    auxiliary_classifier_1 = slim.flatten(auxiliary_classifier_1)
                    auxiliary_classifier_1 = slim.fully_connected(auxiliary_classifier_1, 1024, scope='auxiliary1_fc1')
                    auxiliary_classifier_1 = slim.dropout(auxiliary_classifier_1, 0.7)
                    auxiliary_classifier_1 = slim.fully_connected(auxiliary_classifier_1, class_num, activation_fn=None, scope='auxiliary1_fc2')
                    auxiliary_classifier_1 = slim.softmax(auxiliary_classifier_1, scope='softmax1')

                    # 辅助分类器2 平均池化 filter 5*5 步长3 卷积 卷积核128个 filter 1*1 步长1 全连接层输出节点1024 输出层节点class_num
                    auxiliary_classifier_2 = slim.avg_pool2d(auxiliary_classifier_2, 5, padding='VALID', stride=3, scope='auxiliary2_avgpool')
                    auxiliary_classifier_2 = slim.conv2d(auxiliary_classifier_2, 128, 1, scope='auxiliary2_conv_1x1')
                    auxiliary_classifier_2 = slim.flatten(auxiliary_classifier_2)
                    auxiliary_classifier_2 = slim.fully_connected(auxiliary_classifier_2, 1024, scope='auxiliary2_fc1')
                    auxiliary_classifier_2 = slim.dropout(auxiliary_classifier_2, 0.7)
                    auxiliary_classifier_2 = slim.fully_connected(auxiliary_classifier_2, class_num, activation_fn=None, scope='auxiliary2_fc2')
                    auxiliary_classifier_2 = slim.softmax(auxiliary_classifier_2, scope='softmax2')

                    google_net = auxiliary_classifier_1 * 0.3 + auxiliary_classifier_2 * 0.3 + google_net * 0.4
                    print(google_net.shape)

    return google_net

def main():
    input = tf.random_uniform((5, 224, 224, 3))
    output = GoogLeNet_slim_v1(input, 10, True)
    print(output)

if __name__ == '__main__':
    main()