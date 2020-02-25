import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784  # 输入层的节点数  对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10  # 输出层的节点数  这个等于类别的数目

# 定义与样本数据相关的参数
IMAGE_SIZE = 28     # 像素尺寸
CHANNELS_NUMBER = 1 # 通道数
LABELS_NUMBER = 10  # 手写数字类别数目

# 第一层卷积层的尺寸和深度
CONV1_FILTER_NUMBER = 32
CONV1_FILTER_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_FILTER_NUMBER = 64
CONV2_FILTER_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    """
    定义卷积神经网络的前向传播过程。这里添加了一个新的参数train，用于区分训练过程和测试过程。
    在这个程序中将用到dropout方法，dropout可以进一步提升模型可靠性并防止过拟合，dropout过程只在训练时使用。
    """

    # 声明第一层卷积层的变量并实现前向传播过程。
    # 通过使用不同的命名空间来隔离不同层的变量，这可以让每一层中的变量命名只需要考虑在当前层的作用，而不需要担心重名的问题。
    # 和标准LeNet-5模型不大一样，这里定义的卷积层输入为28x28x1的原始MNIST图片像素。
    # 因为卷积层中使用了全0填充，所以输出为28x28x32的矩阵。
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",
                                        [CONV1_FILTER_SIZE, CONV1_FILTER_SIZE,
                                         CHANNELS_NUMBER, CONV1_FILTER_NUMBER],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_FILTER_NUMBER], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充。
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，
    # 使用全0填充且移动的步长为2。这一层的输入是上一层的输出，也就是28x28x32 #的矩阵。输出为14x14x32的矩阵。
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 声明第三层卷积层的变量并实现前向传播过程。这一层的输入为14x14x32的矩阵。输出为14x14x64的矩阵。
    # 通过tf.get_variable的方式创建过滤器的权重变量和偏置项变量。
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [CONV2_FILTER_SIZE,
                                                   CONV2_FILTER_SIZE,
                                                   CONV1_FILTER_NUMBER,
                                                   CONV2_FILTER_NUMBER],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 卷积层的参数个数只与过滤器的尺寸、深度以及当前层节点矩阵的深度有关，所以这里声明的参数变
        # 量是一个四维矩阵，前面两个维度代表了过滤器的尺寸，第三个维度表示当前层的深度，第四个维度表示过滤器的深度。
        conv2_biases = tf.get_variable("bias", [CONV2_FILTER_NUMBER], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        # 使用边长为5,深度为64的过滤器，过滤器移动的步长为1，且使用全0填充。
        # tf.nn.Conv2d提供了一个非常方便的函数来实现卷积层前向传播的算法。这个函数的第一个输入为当前层的节点矩阵。
        # 注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一维对应一个输入batch。
        # 比如在输入层，input [0,:，:,:]表示第一张图片，input [1,:，:,:]表示第二张图片，以此类推。
        # tf.rm.Conv2d第二个参数提供了卷积层的权重，第三个参数为不同维度上的步长。
        # 虽然第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数字要求一定是1。
        # 这是因为卷积层的步长只对矩阵的长和宽有效。最后一个参数是填充(padding)的方法，
        # TensorFlow中提供SAME或是VALID两种选择。其中SAME表示添加全0填充

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        # tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用加法，
        # 因为矩阵上不同位置上的节点都需要加上同样的偏置项。虽然下一层神经网络的大小为2*2,但是偏置项只有一个数(因为深度为1),
        # 而2*2矩阵中的每一个值都需要加上这个偏置项。

        # 实现第四层池化层的前向传播过程。这一层和第二层的结构是一样的。
        # 这一层的输入为14x14x64的矩阵，输出为7x7x64的矩阵。
        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为7x7x64的矩阵，
            # 然而第五层全连接层需要的输入格式为向量，所以在这里需要将这个7x7x64的矩阵拉直成一
            # 个向量。 pool2.get_Shape函数可以得到第四层输出矩阵的维度而不需要手工计算。
            # 注意,因为每一层神经网络碎输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数。
            pool_shape = pool2.get_shape().as_list()

            # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽及深度的乘积。
            # 注意这里pool_shape[0]为一个batch中数据的个数。
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

            # 通过tf.reshape函数将第四层的输出变成一个batch的向量。
            reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

            # 声明第五层全连接层的变量并实现前向传播过程。这一层的输入是拉直之后的一组向量，
            # 向量长度为3136，输出是一组长度为512的向量。此处引入了 dropout的概念。
            # dropout在训练时会随机将部分节点的输出改为0。dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好。
            # dropout一般只在全连接层而不是卷积层或者池化层使用。

        with tf.variable_scope("layer5-fc1"):
            fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))

            # 只有全连接层的权重需要加入正则
            # 当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合。在这里
            # 使用了add_to_collection函数将一个张量加入一个集合，而这个集合的名称为losses。
            # 注意这是自定义的集合，不在TensorFlow自动管理的集合列表中。
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(fc1_weights))
            fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if train:
                fc1 = tf.nn.dropout(fc1, 0.5)

        # 声明第六层全连接层的变量并实现前向传播过程。这一层的输入为一组长度为512的向量，
        # 输出为一组长度为10的向量。这一层的输出通过Softmax之后就得到了最后的分类结果。
        with tf.variable_scope('layer6-fc2'):
            fc2_weights = tf.get_variable("weight", [FC_SIZE, LABELS_NUMBER],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [LABELS_NUMBER], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc1, fc2_weights) + fc2_biases

        # 返回第六层的输出。
        return logit

