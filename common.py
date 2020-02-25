import tensorflow as tf
import numpy as np

# 带BN的训练函数
def bn_optimizer(lr, loss, momentum=0.9):
    with tf.name_scope('optimzer_bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("BN parameters: ", update_ops)
        with tf.control_dependencies([tf.group(*update_ops)]):
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum)
            train_op = slim.learning.create_train_op(loss,optimizer)
    return train_op

# 得到独热编码的label
def get_onehot_label(label, class_num):
    batch_label = np.zeros([len(label), class_num])
    for i in range(len(label)):
        batch_label[i][label[i]] = 1
    return batch_label

