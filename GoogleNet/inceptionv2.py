import tensorflow as tf
import tensorflow.contrib.slim as slim

# inception v2模块 卷积结构
def inception_module_v2(google_net, filter_num, pool_type, stride, scope):
    with tf.variable_scope(scope):
        # inception v2的第一部分 1 × 1的卷积
        if filter_num[0] != 0:
            with tf.variable_scope('component1'):
                component1 = slim.conv2d(google_net, filter_num[0], 1, stride=stride, scope='component1_conv1_1x1')

        # inception v2的第二部分  1 × 1的卷积  3 × 3的卷积
        with tf.variable_scope('component2'):
            component2 = slim.conv2d(google_net, filter_num[1], 1, stride=1, scope='component2_conv1_1x1')
            component2 = slim.conv2d(component2, filter_num[2], 3, stride=stride, scope='component2_conv2_3x3')

        # inception v2的第三部分  1 × 1的卷积  3 × 3的卷积 3 × 3的卷积
        with tf.variable_scope('component3'):
            component3 = slim.conv2d(google_net, filter_num[3], 1, stride=1, scope='component3_conv1_1x1')
            component3 = slim.conv2d(component3, filter_num[4], 3, stride=1, scope='component3_conv2_3x3')
            component3 = slim.conv2d(component3, filter_num[5], 3, stride=stride, scope='component3_conv3_3x3')

        # inception v2的第四部分  3 × 3的平均/最大池化  1 × 1的卷积
        with tf.variable_scope('component4'):
            if pool_type == 'avg_pool':
                component4 = slim.avg_pool2d(google_net, 3, stride=stride, scope='component4_avg_pool_3x3')
            elif pool_type == 'max_pool':
                component4 = slim.max_pool2d(google_net, 3, stride=stride, scope='component4_max_pool_3x3')
            else:
                raise TypeError("没有此类型的池化层")

            if filter_num[0] != 0:
                component4 = slim.conv2d(component4, filter_num[6], 1, stride=1, scope='component4_conv_1x1')
                # 将inception v2的卷积结构每个部分的结果 通道加起来
                google_net = tf.concat([component1, component2, component3, component4], axis=3)
            else:
                google_net = tf.concat([component2, component3, component4], axis=3)

    return google_net

# GoogLeNet v2模型
def GoogLeNet_slim_v2(input, class_num, is_training=False, keep_prob=0.8, spatital_squeeze=True):
    # BN的参数
    batch_norm_params = {
        'decay': 0.998,
        'epsilon': 0.001,
        'scale': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
    }
    # reshape 成224 224 3
    with tf.name_scope('reshape'):
        google_net = tf.reshape(input, [-1, 224, 224, 3])

    # 卷积层和深度可分卷积层(每个通道单独做2d卷积 最后通道加起来 后面接1x1卷积) 权重初始化
    # BN mini-batch sample归一化 减少Internal Covariate Shift（内部神经元分布的改变）
    # 将每层的输入分布都拉到均值0 方差1的标准正态分布里来 加快收敛 并避免过拟合
    with tf.variable_scope('GoogLeNet_V2'):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                             weights_initializer=slim.xavier_initializer(),
                             normalizer_fn=slim.batch_norm,
                             # normalizer_fn=tf.layers.batch_normalization,
                             normalizer_params=batch_norm_params):
            # 卷积层 深度可分卷积层 和最大池化层 平均池化层 补0填充默认方法 和步长1
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.max_pool2d, slim.avg_pool2d],
                                 padding='SAME', stride=1):
                # BN层里 解包（keyword）设置每个默认参数
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    # 深度可分卷积 64个卷积核 filter 7*7 步长2  最大池化 filter 3*3 步长2
                    google_net = slim.separable_conv2d(google_net, 64, 7, depth_multiplier=8,
                                                       stride=2, scope='layer1')
                    google_net = slim.max_pool2d(google_net, 3, stride=2, scope='layer2')
                    # 卷积 64个卷积核 filter 1*1 步长1 卷积 192个卷积核 filter 3*3  最大池化 filter 3*3 步长2
                    google_net = slim.conv2d(google_net, 64, 1, scope='layer3')
                    google_net = slim.conv2d(google_net, 192, 3, scope='layer4')
                    google_net = slim.max_pool2d(google_net, 3, stride=2, scope='layer5')
                    # inception v2卷积组合结构
                    google_net = inception_module_v2(google_net, filter_num=[64, 64, 64, 64, 96, 96, 32],
                                                     pool_type='avg_pool', stride=1, scope='layer6_inception_3a')
                    google_net = inception_module_v2(google_net, filter_num=[64, 64, 96, 64, 96, 64, 64],
                                                     pool_type='avg_pool', stride=1, scope='layer9_inception_3b')
                    google_net = inception_module_v2(google_net, filter_num=[0, 128, 160, 64, 96, 96],
                                                     pool_type='max_pool', stride=2, scope='layer12_inception_3c')
                    google_net = inception_module_v2(google_net, filter_num=[224, 64, 96, 96, 128, 128, 128],
                                                     pool_type='avg_pool', stride=1, scope='layer15_inception_4a')
                    # 辅助分类器1 加入总结果权重*0.3
                    # auxiliary_classifier_1 = google_net
                    google_net = inception_module_v2(google_net, filter_num=[192, 96, 128, 96, 128, 128, 128],
                                                     pool_type='avg_pool', stride=1, scope='layer18_inception_4b')
                    google_net = inception_module_v2(google_net, filter_num=[160, 128, 160, 128, 160, 160, 128],
                                                     pool_type='avg_pool', stride=1, scope='layer21_inception_4c')
                    # 辅助分类器2 加入总结果权重*0.3
                    # auxiliary_classifier_2 = google_net
                    google_net = inception_module_v2(google_net, filter_num=[96, 128, 192, 160, 192, 192, 128],
                                                     pool_type='avg_pool', stride=1, scope='layer24_inception_4d')
                    google_net = inception_module_v2(google_net, filter_num=[0, 128, 192, 192, 256, 256],
                                                     pool_type='max_pool', stride=2, scope='layer27_inception_4e')
                    google_net = inception_module_v2(google_net, filter_num=[352, 192, 320, 160, 224, 224, 128],
                                                     pool_type='avg_pool', stride=1, scope='layer30_inception_5a')
                    google_net = inception_module_v2(google_net, filter_num=[352, 192, 320, 192, 224, 224, 128],
                                                     pool_type='max_pool', stride=1, scope='layer33_inception_5b')

                    # 用平均池化层 代替了全连接层 filter 7*7 步长1
                    google_net = slim.avg_pool2d(google_net, 7, stride=1, padding='VALID', scope='layer36')
                    # 不用全连接层 但依然使用dropout
                    google_net = slim.dropout(google_net, keep_prob=keep_prob, scope='dropout')
                    # 输出层 全连接层 不要激活函数和归一化
                    google_net = slim.conv2d(google_net, class_num, 1, activation_fn=None, normalizer_fn=None, scope='layer37')
                    if spatital_squeeze:
                        # 删除第2维和第三维 维度是1的维度
                        google_net = tf.squeeze(google_net, [1, 2], name='squeeze')
                    # softmax
                    google_net = slim.softmax(google_net, scope='softmax')

    return google_net

def main():
    input = tf.random_uniform((5, 224, 224, 3))
    output = GoogLeNet_slim_v2(input, 10, True)
    print(output)

if __name__ == '__main__':
    main()