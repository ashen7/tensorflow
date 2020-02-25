import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#global variables
x_train = '' 
y_train = ''
weight_value = ''
bias_value = ''
#存放loss函数 在学习过程中 值的变化
total = list()

#神经网络中 所有的输入都线性增加 为了使训练有效 输入应该被归一化
def normalize(x):
    mean = np.mean(x)    #均值
    std = np.std(x)      #标准差
    x = (x - mean) / std #x - 均值 /  方差
    return x

#简单线性回归 预测目标值
def simple_linear_regression():
    #改变全局变量的值
    global x_train
    global y_train
    global weight_value
    global bias_value
    global total

    #会话的配置  手动选择XLA_CPU 加速线性代数器 
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    #1. 从contrib数据集中加载波士顿房价数据集  分解为x_train,  y_train
    boston = tf.contrib.learn.datasets.load_dataset('boston')
    #对波士顿房价数据集的房间数量RM采用简单线性回归  目标是预测1000美元自有住房中值
    x_train, y_train = boston.data[:, 5], boston.target
    #样本的数量
    samples_number = len(x_train)
        
    #2. 为训练数据声明tensorflow占位符
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    y = tf.compat.v1.placeholder(tf.float32, name='y')
    
    #3. 创建权重和偏置变量
    weight = tf.Variable(0.0)
    bias = tf.Variable(0.0)

    #4. 定义用于预测的线性回归模型
    output = weight * x + bias
    
    #5. 定义损失函数  求真实值和预测值的平方差之和
    loss = tf.square(y - output, name='loss')

    #6. 选择梯度下降优化器 学习率设为0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    
    #7. 初始化运算操作符
    init_op = tf.compat.v1.global_variables_initializer()

    iter_count = 100
    #训练100次
    with tf.Session(config=config) as session:
        session.run(init_op)
        #写入事件文件
        # writer = tf.compat.v1.summary.FileWriter('graphs', session.graph)
        for i in range(iter_count):
            total_loss = 0
            #每次遍历506个样本 学习
            for x_data, y_data in zip(x_train, y_train):
                #通过feed_dict将数据集传入  执行计算图  优化器 损失函数
                optimizer_, loss_, output_ = session.run([optimizer, loss, output],
                                                feed_dict={x:x_data, y:y_data})
                total_loss += loss_
                print('y的值：', y_data, '   y的点估计(最优预测)：', output_, '     loss函数的值:', loss_,  '     w权重的值:',session.run(weight), '     b偏置的值', session.run(bias))
            #加入到loss函数值的列表  loss函数总值 / 样本总量
            total.append(total_loss / samples_number)
            print('\n=========================Epoch {0}: Loss{1}===========================\n'.format(i, total[i]))
        
        #这里训练完成  得到了最优的系数（权重）和偏置 就是使loss函数最小化的值
        # writer.close()
        weight_value, bias_value = session.run([weight, bias])
    
def visualization():   #可视化
    output = weight_value * x_train + bias_value
    print('Done')

    #画散点图
    plt.subplot(1, 2, 1)
    plt.title('Data Scatter Plot')
    plt.xlabel('Rooms Number')
    plt.ylabel('House Price in $1000')
    plt.plot(x_train, y_train, 'bo', label='Real Data')
    plt.plot(x_train, output, 'r', label='Predicted Data')
    plt.legend(loc='best')  #显示图例
    
    plt.subplot(1, 2, 2)
    plt.title('Loss Function')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.plot(total)

    #显示
    plt.show()

def main():
    simple_linear_regression()
    visualization()

if __name__ == "__main__":
    main()
