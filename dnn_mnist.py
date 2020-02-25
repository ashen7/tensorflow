from keras.models import Sequential  #序贯模型
from keras.layers.core import Dense, Dropout, Activation #网络层
from keras.optimizers import SGD     #优化方法
from keras.datasets import mnist     #数据集
import numpy as np

def main():
    # 1. 选择模型
    model = Sequential()

    # 2. 构建网络层
    # Dense是网络层中的常用层(全连接层) 输出维度为N * 500 输入维度是N * 784(28 * 28)
    model.add(Dense(500, input_shape=(784,)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(500))   #隐藏层节点500个
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(10))    #输出结果是10个类别
    model.add(Activation('softmax'))  #最后一层用softmax作为激活函数

    # 3. 编译
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #优化函数
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
                  # class_mode='categorical')  #使用交叉熵作为loss函数

    # 4. 训练(fit拟合) 和 评估
    '''
    shuffle: 是否把数据随机打乱 
    validation_split 拿数据集的多少比例用做交叉验证
    verbose: 屏显模式 0是不输出 1是输出进度 2是输出每次的训练结果
    '''
    (train_sample, train_label), (test_sample, test_label) = mnist.load_data()
    # 样本reshape成N * 784 标签reshape成N * 10
    train_sample = train_sample.reshape(train_sample.shape[0], train_sample.shape[1] * train_sample.shape[2])
    test_sample = test_sample.reshape(test_sample.shape[0], test_sample.shape[1] * test_sample.shape[2])
    train_label = (np.arange(10) == train_label[:, None]).astype(int)
    test_label = (np.arange(10) == test_label[:, None]).astype(int)

    model.fit(train_sample, train_label, batch_size=200, epochs=50, shuffle=True, verbose=2, validation_split=0.3)
    model.evaluate(test_sample, test_label, batch_size=200, verbose=2)

    # 5. 输出
    # 返回误差率
    score = model.evaluate(test_sample, test_label, batch_size=200, verbose=1)
    print("The test loss is: %f" % score)
    # 预测
    result = model.predict(test_sample, batch_size=200, verbose=2)
    print("result: {}, result.shape: {}".format(result, result.shape))

    result_max = np.argmax(result, axis=1)
    test_max = np.argmax(test_label, axis=1)
    print("predict result: {}".format(result_max))
    print("ground truth: {}".format(test_max))

    result_bool = np.equal(result_max, test_max)
    true_number = np.sum(result_bool)
    print("The accuracy of the model is {}".format(true_number / len(result_bool)))

if __name__ == '__main__':
    main()