import tensorflow as tf
import tensorflow.contrib.slim as slim

# inception v3模块 第一种卷积结构(也就是inception v1) 步长都是1
def inception_module_v3_1(google_net, filter_num, scope, stride=1):
    with tf.variable_scope(scope):
        # inception v3的第一部分 1 × 1的卷积
        with tf.variable_scope('component1'):
            component1 = slim.conv2d(google_net, filter_num[0], [1, 1], stride=stride, scope='component1_conv1_1x1')

        # inception v3的第二部分  3 × 3的平均池化  1 × 1的卷积
        with tf.variable_scope('component2'):
            component2 = slim.avg_pool2d(google_net, [3, 3], stride=stride, scope='component2_avg_pool_3x3')
            component2 = slim.conv2d(component2, filter_num[1], [1, 1], stride=stride, scope='component2_conv_1x1')

        # inception v3的第三部分  1 × 1的卷积  3 × 3的卷积
        with tf.variable_scope('component3'):
            component3 = slim.conv2d(google_net, filter_num[2], [1, 1], stride=stride, scope='component3_conv1_1x1')
            component3 = slim.conv2d(component3, filter_num[3], [3, 3], stride=stride, scope='component3_conv2_3x3')

        # inception v3的第四部分  1 × 1的卷积  3 × 3的卷积 3 × 3的卷积
        with tf.variable_scope('component4'):
            component4 = slim.conv2d(google_net, filter_num[4], [1, 1], stride=stride, scope='component4_conv1_1x1')
            component4 = slim.conv2d(component4, filter_num[5], [3, 3], stride=stride, scope='component4_conv2_3x3')
            component4 = slim.conv2d(component4, filter_num[6], [3, 3], stride=stride, scope='component4_conv3_3x3')
        # 所有通道加起来 axis=3 加到第四维上
        google_net = tf.concat([component1, component2, component3, component4], axis=3)

    return google_net

# inception v3模块 第二种卷积结构 使用了1xn和nx1代替n*n的filter
# n*n提取的是区域特征 1*n提取的是条纹 方向性特征 适合中型特征图12-20 前面的层用了效果不佳
def inception_module_v3_2(google_net, filter_num, scope, stride=1):
    with tf.variable_scope(scope):
        # inception v3的第一部分 1 × 1的卷积
        with tf.variable_scope('component1'):
            component1 = slim.conv2d(google_net, filter_num[0], [1, 1], stride=stride, scope='component1_conv1_1x1')

        # inception v3的第二部分  3 × 3的平均池化  1 × 1的卷积
        with tf.variable_scope('component2'):
            component2 = slim.avg_pool2d(google_net, [3, 3], stride=stride, scope='component2_avg_pool_3x3')
            component2 = slim.conv2d(component2, filter_num[1], [1, 1], stride=stride, scope='component2_conv_1x1')

        # inception v3的第三部分  1 x 1的卷积 1 × 7的卷积  7 × 1的卷积  代替7 x 7的卷积
        with tf.variable_scope('component3'):
            component3 = slim.conv2d(google_net, filter_num[2], [1, 1], stride=stride, scope='component3_conv1_1x1')
            component3 = slim.conv2d(component3, filter_num[3], [1, 7], stride=stride, scope='component3_conv2_1x7')
            component3 = slim.conv2d(component3, filter_num[4], [7, 1], stride=stride, scope='component3_conv3_7x1')

        # inception v3的第四部分  1 × 1的卷积  1 × 7的卷积  7 × 1的卷积 1 × 7的卷积  7 × 1的卷积
        with tf.variable_scope('component4'):
            component4 = slim.conv2d(google_net, filter_num[5], [1, 1], stride=stride, scope='component4_conv1_1x1')
            component4 = slim.conv2d(component4, filter_num[6], [1, 7], stride=stride, scope='component4_conv2_1x7')
            component4 = slim.conv2d(component4, filter_num[7], [7, 1], stride=stride, scope='component4_conv3_7x1')
            component4 = slim.conv2d(component4, filter_num[8], [1, 7], stride=stride, scope='component4_conv4_1x7')
            component4 = slim.conv2d(component4, filter_num[9], [7, 1], stride=stride, scope='component4_conv5_7x1')
        # 所有通道加起来 axis=3 加到第四维上
        google_net = tf.concat([component1, component2, component3, component4], axis=3)

    return google_net

# inception v3模块 第三种卷积结构 使用了1xn和nx1代替n*n的filter
def inception_module_v3_3(google_net, filter_num, scope, stride=1):
    with tf.variable_scope(scope):
        # inception v3的第一部分 1 × 1的卷积
        with tf.variable_scope('component1'):
            component1 = slim.conv2d(google_net, filter_num[0], [1, 1], stride=stride, scope='component1_conv1_1x1')

        # inception v3的第二部分  3 × 3的平均池化  1 × 1的卷积
        with tf.variable_scope('component2'):
            component2 = slim.avg_pool2d(google_net, [3, 3], stride=stride, scope='component2_avg_pool_3x3')
            component2 = slim.conv2d(component2, filter_num[1], [1, 1], stride=stride, scope='component2_conv_1x1')

        # inception v3的第三部分  1 x 1的卷积 3 × 1的卷积  1 × 3的卷积  代替3 x 3的卷积
        with tf.variable_scope('component3'):
            component3 = slim.conv2d(google_net, filter_num[2], [1, 1], stride=stride, scope='component3_conv1_1x1')
            component3 = slim.conv2d(component3, filter_num[3], [3, 1], stride=stride, scope='component3_conv2_3x1')
            component3 = slim.conv2d(component3, filter_num[4], [1, 3], stride=stride, scope='component3_conv3_1x3')

        # inception v3的第四部分  1 × 1的卷积  3 × 3的卷积 3 × 3的卷积
        with tf.variable_scope('component4'):
            component4 = slim.conv2d(google_net, filter_num[5], [1, 1], stride=stride, scope='component4_conv1_1x1')
            component4 = slim.conv2d(component4, filter_num[6], [3, 3], stride=stride, scope='component4_conv2_3x3')
            component4 = slim.conv2d(component4, filter_num[7], [3, 1], stride=stride, scope='component4_conv3_3x1')
            component4 = slim.conv2d(component4, filter_num[8], [1, 3], stride=stride, scope='component4_conv4_1x3')
        # 所有通道加起来 axis=3 加到第四维上
        google_net = tf.concat([component1, component2, component3, component4], axis=3)

    return google_net

# inception v3模块 用来减少grid-size
def inception_module_v3_reduce(google_net, filter_num, scope):
    with tf.variable_scope(scope):
        with tf.variable_scope('component1'):
            component1 = slim.max_pool2d(google_net, [3, 3], stride=2, padding='VALID', scope='component1_max_pool_3x3')

        with tf.variable_scope('component2'):
            component2 = slim.conv2d(google_net, filter_num[0], [1, 1], stride=1, scope='component2_conv1_1x1')
            component2 = slim.conv2d(component2, filter_num[1], [3, 3], stride=2, padding='VALID', scope='component2_conv2_3x3')

        with tf.variable_scope('component3'):
            component3 = slim.conv2d(google_net, filter_num[2], [1, 1], stride=1, scope='component3_conv1_1x1')
            component3 = slim.conv2d(component3, filter_num[3], [3, 3], stride=1, scope='component3_conv2_3x3')
            component3 = slim.conv2d(component3, filter_num[4], [3, 3], stride=2, padding='VALID', scope='component3_conv3_3x3')
        # 所有通道加起来 axis=3 加到第四维上
        google_net = tf.concat([component1, component2, component3], axis=3)

    return google_net


# GoogLeNet v3模型
def GoogLeNet_slim_v3(input, class_num, is_training=False, keep_prob=0.8, spatital_squeeze=True):
    # BN的参数
    batch_norm_params = {
        'decay': 0.998,
        'epsilon': 0.001,
        'scale': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    # reshape 成224 224 3
    with tf.name_scope('reshape'):
        google_net = tf.reshape(input, [-1, 299, 299, 3])

    # 卷积层和深度可分卷积层(每个通道单独做2d卷积 最后通道加起来 后面接1x1卷积) 权重初始化
    # BN mini-batch sample归一化 减少Internal Covariate Shift（内部神经元分布的改变）
    # 将每层的输入分布都拉到均值0 方差1的标准正态分布里来 加快收敛 并避免过拟合
    with tf.variable_scope('GoogLeNet_V3'):
        # 卷积层 全连接层 权重L2正则化
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                             weights_regularizer=slim.l2_regularizer(0.00004)):
            # 卷积层 激活函数relu前使用 BN batch sample归一化到均值0 方差1的标准正态分布
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=slim.xavier_initializer(),
                                normalizer_fn=slim.batch_norm,
                                # normalizer_fn=tf.layers.batch_normalization,
                                normalizer_params=batch_norm_params):
                # BN dropout 区分是训练还是测试
                with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                    # 卷积层 最大池化层 平均池化层 补0填充默认方法不填 和步长1
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                         padding='VALID', stride=1):
                        # 卷积 32个卷积核 filter 3*3 步长2
                        google_net = slim.conv2d(google_net, 32, [3, 3], stride=2, scope='layer1') #149x149
                        google_net = slim.conv2d(google_net, 32, [3, 3], scope='layer2')  # 147x147
                        google_net = slim.conv2d(google_net, 64, [3, 3], padding='SAME', scope='layer3')  # 147x147
                        google_net = slim.max_pool2d(google_net, [3, 3], stride=2, scope='layer4')  #73x73
                        google_net = slim.conv2d(google_net, 80, [3, 3], scope='layer5')  # 71x71
                        google_net = slim.conv2d(google_net, 192, [3, 3], stride=2, scope='layer6')  # 35x35

                    # 卷积层 最大池化层 平均池化层 补0填充默认方法填充 和步长1
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                        padding='SAME', stride=1):
                        google_net = slim.conv2d(google_net, 288, [3, 3], scope='layer7')
                        # 3 x inception v3第一种卷积组合结构 1*1后接n×n的卷积
                        google_net = inception_module_v3_1(google_net, filter_num=[64, 32, 48, 64, 64, 96, 96],
                                                           scope='layer8_inception_3a')  #35x35
                        google_net = inception_module_v3_1(google_net, filter_num=[64, 64, 48, 64, 64, 96, 96],
                                                           scope='layer11_inception_3b')
                        google_net = inception_module_v3_1(google_net, filter_num=[64, 64, 48, 64, 64, 96, 96],
                                                           scope='layer14_inception_3c')
                        print(google_net)

                        # 5 x inception v3第二种卷积组合结构 1*1后接1×n的卷积 再接n×1的卷积
                        google_net = inception_module_v3_reduce(google_net, filter_num=[192, 384, 64, 96, 96],
                                                                scope='layer17_inception_4a')  # 17x17
                        google_net = inception_module_v3_2(google_net,
                                                           filter_num=[192, 192, 128, 128, 192, 128, 128, 128, 128, 192],
                                                           scope='layer20_inception_4b')
                        google_net = inception_module_v3_2(google_net,
                                                           filter_num=[192, 192, 160, 160, 192, 160, 160, 160, 160, 192],
                                                           scope='layer25_inception_4c')
                        google_net = inception_module_v3_2(google_net,
                                                           filter_num=[192, 192, 160, 160, 192, 160, 160, 160, 160, 192],
                                                           scope='layer30_inception_4d')
                        google_net = inception_module_v3_2(google_net,
                                                           filter_num=[192, 192, 160, 160, 192, 160, 160, 160, 160, 192],
                                                           scope='layer35_inception_4e')
                        print(google_net)

                        # 5 x inception v3第三种卷积组合结构 1*1后接n×n的卷积 再接1×n nx1的卷积
                        google_net = inception_module_v3_reduce(google_net, filter_num=[192, 320, 192, 192, 192],
                                                                scope='layer40_inception_5a')  # 8x8
                        google_net = inception_module_v3_3(google_net,
                                                           filter_num=[320, 192, 384, 384, 384, 448, 384, 384, 384],
                                                           scope='layer43_inception_5b')
                        google_net = inception_module_v3_3(google_net,
                                                           filter_num=[320, 192, 384, 384, 384, 448, 384, 384, 384],
                                                           scope='layer46_inception_5c')
                        print(google_net)

                        # 用平均池化层 代替了全连接层 filter 8*8 步长1
                        google_net = slim.avg_pool2d(google_net, [8, 8], padding='VALID', scope='layer49')
                        # 不用全连接层 但依然使用dropout
                        google_net = slim.dropout(google_net, scope='dropout')
                        # 输出层 全连接层 不要激活函数和归一化
                        google_net = slim.conv2d(google_net, class_num, [1, 1], activation_fn=None, normalizer_fn=None, scope='layer50')
                        print(google_net)
                    if spatital_squeeze:
                        # 删除第2维和第三维 维度是1的维度
                        google_net = tf.squeeze(google_net, [1, 2], name='squeeze')
                    # softmax
                    google_net = slim.softmax(google_net, scope='softmax')

    return google_net

def main():
    input = tf.random_uniform((5, 299, 299, 3))
    output = GoogLeNet_slim_v3(input, 10, True)
    print(output)

if __name__ == '__main__':
    main()