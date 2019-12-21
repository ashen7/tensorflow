import time
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import LeNet_5
import LeNet_5_train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVALUATE_INTERVAL_SECONDS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:  # 将默认图设为g
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [mnist.validation.images.shape[0],
                                        LeNet_5.IMAGE_SIZE,
                                        LeNet_5.IMAGE_SIZE,
                                        LeNet_5.CHANNELS_NUMBER], name='x-input')
        y = tf.placeholder(tf.float32, [None, LeNet_5.OUTPUT_NODE], name='y-input')

        x_data = mnist.validation.images
        # 类似地将输入的测试数据格式调整为一个四维矩阵
        reshaped_x_data = np.reshape(x_data, (mnist.validation.images.shape[0],
                                              LeNet_5.IMAGE_SIZE,
                                              LeNet_5.IMAGE_SIZE,
                                              LeNet_5.CHANNELS_NUMBER))
        validate_feed = {x: reshaped_x_data,
                         y: mnist.validation.labels}

        # 直接通过调用封装好的函数来计算前向传播的结果
        # 测试时不关注过拟合问题，所以正则化输入为None
        train = False
        regular = None
        output = LeNet_5.inference(x, train, regular)

        # 使用前向传播的结果计算正确率，如果需要对未知的样例进行分类
        # 使用tf.argmax(y, 1)就可以得到输入样例的预测类别
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        # 首先将一个布尔型的数组转换为实数，然后计算平均值
        # 平均值就是网络在这一组数据上的正确率
        # True为1，False为0
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(LeNet_5_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        # 所有滑动平均的值组成的字典，处在/ExponentialMovingAverage下的值
        # 为了方便加载时重命名滑动平均量，tf.train.ExponentialMovingAverage类
        # 提供了variables_to_store函数来生成tf.train.Saver类所需要的变量
        saver = tf.train.Saver(variable_to_restore)  # 这些值要从模型中提取

        # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        # while True:
        for i in range(10):  # 为了降低个人电脑的压力，此处只利用最后生成的模型对测试数据集做测试
            with tf.Session() as session:
                # tf.train.get_checkpoint_state函数
                # 会通过checkpoint文件自动找到目录中最新模型的文件名
                model = tf.train.get_checkpoint_state("./model")
                if model and model.model_checkpoint_path:
                    # 加载模型
                    saver.restore(session, model.model_checkpoint_path)
                    # 得到所有的滑动平均值
                    # 通过文件名得到模型保存时迭代的轮数
                    global_epoch = model.model_checkpoint_path.split('-')[-1]
                    accuracy_score = session.run(accuracy, feed_dict=validate_feed)  # 使用此模型检验
                    # 没有初始化滑动平均值，只是调用模型的值，inference只是提供了一个变量的接口，完全没有赋值
                    print("After {} training epoch, validation accuracy = {}".format(global_epoch, accuracy_score))
                else:
                    print("Not found checkpoint file")
                    return
                time.sleep(EVALUATE_INTERVAL_SECONDS)
                # time sleep()函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。


def main(argv=None):
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
