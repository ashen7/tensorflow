import tensorflow as tf

#全局变量
m = 1000  #样本数量
n = 15    #特征数量
p = 2     #类别数量

#在回归中定义损失函数或目标函数  就是为了找到损失最小化的系数W
#声明一个损失函数 将系数w定义为变量  将数据集定义为占位符 可以有一个常学习率和正则化常数

#标准线性回归是一个输入变量和一个输出变量
def standard_linear_regression():
    #训练数据
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    y = tf.compat.v1.placeholder(tf.float32, name='y')

    #给系数定义为变量  先初始化一下
    w0 = tf.Variable(0.0)
    w1 = tf.Variable(0.0)

    #线性回归模型 
    y_hat = w1 * x + w0  #y_hat是预测值 = 系数 * 输入 + 偏置 
    
    #损失函数  使观测值y - 预测值y_hat的差的平方和最小 
    loss = tf.square(y - y_hat, name='loss')

#多元线性回归是多个输入变量和一个输出变量
def multiple_linear_regression():
    #训练数据
    x = tf.compat.v1.placeholder(tf.float32, name='x', shape=[m, n])
    y = tf.compat.v1.placeholder(tf.float32, name='y')
    
    #给系数定义为变量  先初始化一下
    w0 = tf.Variable(0.0)
    w1 = tf.Variable(tf.random_normal([n, 1]))

    #线性回归模型   矩阵相乘 + w0
    y_hat = tf.matmul(x, w1) + w0
    
    #损失函数  使观测值y - 预测值y_hat的差的平方和最小 
    #reduce_mean 还原平均值
    loss = tf.reduce_mean(tf.square(y - y_hat, name='loss'))

#逻辑回归  用来确定一个事件的概率  用于分类问题
#损失函数定义为交叉熵 输出y的维数等于训练数据集中类别的数量p
def logic_regression():
    #训练数据
    x = tf.compat.v1.placeholder(tf.float32, name='x', shape=[m, n])
    y = tf.compat.v1.placeholder(tf.float32, name='y', shape=[m, p])
    
    #给系数定义为变量  先初始化一下
    w0 = tf.Variable(tf.zeros([1, p]), name='bias')
    w1 = tf.Variable(tf.random_normal([n, 1]))

    #线性回归模型   矩阵相乘 + w0
    y_hat = tf.matmul(x, w1) + w0
    
    #损失函数  交叉熵
    entropy = tf.nn.softmax_cross_entropy_with_logits(y_hat, y)
    #还原平均值
    loss = tf.reduce_mean(entropy)

    #如果想把L1正则化加到损失上 
    lamda = tf.constant(0.8)  #正则化参数 r(lamda)
    regularization_param = lamda * tf.reduce_sum(tf.abs(w1))
    #new loss
    loss += regularization_param

    #L2正则化
    lamda = tf.constant(0.8)
    regularization_param = lamda * tf.nn.12_loss(w1)
    #new loss
    loss += regularization_param

#函数在一阶导数为里零的地方达到其最大值和最小值
#梯度下降算法基于相同的原理 即调整系统（权重和偏置）来使损失函数的梯度下降
#在回归中 使用梯度下降来优化损失函数并获得系数


def main():
    #为确保收敛 损失函数应为凸的 一个光滑的 可谓分的凸损失函数可提供更好的收敛性
    #随着学习的进行 损失函数的值应该下降 最终变得稳定
    standard_linear_regression()
    multiple_linear_regression()
    logic_regression()
    

if __name__ == "__main__":
    main()
