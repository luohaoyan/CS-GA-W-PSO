# -*- coding: utf-8 -*-
import pandas as pd  # 解决数据分析任务，纳入了大量库和一些标准的数据模型
import numpy as np  # 开源数值计算，用来储存和处理大型矩阵
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# 数据流编程，广泛应用于机器学习以及深度学习等


tf.reset_default_graph()  # tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形

# Read Training Data
data = pd.read_csv("damilhl.csv", header=None)
data = np.array(data).astype('float32')

# Activate a Session
tsess = tf.InteractiveSession()  # 它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython

# Read Training Labels
train_data = data[:, :900]
train_data = np.array(train_data).astype('float32')
train_labels = data[:, 900]
train_labels = np.array(train_labels).astype('float32')
train_labels = tf.one_hot(tf.cast(train_labels, tf.int32), 6, 1, 0)  # 由于one-hot类型数据长度为depth位，将input转化为one-hot类型数据输出
train_labels = tf.squeeze(train_labels).eval(session=tsess)  # eval() 函数用来执行一个字符串表达式并返回表达式的值

Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_data, train_labels, test_size=0.33)


def create_np_nor(shape, mean=0, scale=1):
    """创建符合正太分布的对应shape的np数组"""
    return np.random.normal(size=shape, loc=mean, scale=scale)


def create_part_x():
    """创建part对应的初始位置"""
    con1_w = create_np_nor([16])
    con1_b = create_np_nor([4])
    con2_w = create_np_nor([288])
    con2_b = create_np_nor([8])
    con3_w = create_np_nor([1152])
    con3_b = create_np_nor([16])
    w_fc = create_np_nor([4704])
    b_fc = create_np_nor([6])
    all_con = np.hstack((con1_w, con1_b, con2_w, con2_b, con3_w, con3_b, w_fc, b_fc))
    part_x = all_con.reshape([1, all_con.size])
    return part_x


def get_perdict(x, part_x):
    """
    传入权重, 返回模型
    :param x:
    :param part_x: 输入的权重
    :return:
    """

    # 第一层卷积
    with tf.variable_scope("conv1"):
        input_x = tf.reshape(x, shape=[-1, 30, 30, 1])  # 输入数据格式转换
        # 卷积层部分
        part_w1 = part_x[0:16].reshape([2, 2, 1, 4])
        part_b1 = part_x[16:20].reshape([4])
        conv1_weights = tf.Variable(initial_value=part_w1)
        conv1_bias = tf.Variable(initial_value=part_b1)
        conv1_x = tf.nn.conv2d(input=input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias
        # 激活层
        h_conv1 = tf.nn.relu(conv1_x)
        pool1_x = tf.nn.max_pool(value=h_conv1, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding="VALID")

    # 第二层卷积
    with tf.variable_scope("conv2"):
        part_w2 = part_x[20:308].reshape([3, 3, 4, 8])
        part_b2 = part_x[308:316].reshape([8])
        conv2_weights = tf.Variable(initial_value=part_w2)
        conv2_bias = tf.Variable(initial_value=part_b2)

        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding="VALID") + conv2_bias

        h_conv2 = tf.nn.relu(conv2_x)

        pool2_x = tf.nn.max_pool(value=h_conv2, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 第三层卷积
    with tf.variable_scope("conv3"):
        part_w3 = part_x[316:1468].reshape([3, 3, 8, 16])
        part_b3 = part_x[1468:1484].reshape([16])

        conv3_weights = tf.Variable(initial_value=part_w3)
        conv3_bias = tf.Variable(initial_value=part_b3)
        conv3_x = tf.nn.conv2d(input=pool2_x, filter=conv3_weights, strides=[1, 1, 1, 1], padding="VALID") + conv3_bias

        h_conv3 = tf.nn.relu(conv3_x)
        pool3_x = tf.nn.max_pool(value=h_conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID")

    with tf.variable_scope("full_connection"):
        part_w_fc = part_x[1484:6188].reshape([7 * 7 * 16, 6])
        part_b_fc = part_x[6188:6194].reshape([6])

        x_fc = tf.reshape(pool3_x, shape=[-1, 7 * 7 * 16])
        weights_fc = tf.Variable(initial_value=part_w_fc)
        bias_fc = tf.Variable(initial_value=part_b_fc)
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

        all_w_b = [conv1_weights, conv1_bias, conv2_weights, conv2_bias, conv3_weights, conv3_bias, weights_fc, bias_fc]

    return y_predict, all_w_b


def train(part):
    part_x = part.reshape([part.size])  # w转为合适的权重格式
    x = tf.placeholder(tf.float64, [None, 900])
    y = tf.placeholder(tf.float64, [None, 6])
    prediction, all_w_b = get_perdict(x, part_x)

    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # 取每一行最大值做对比，看分类正确个数
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 准确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={x: Xtrain, y: Ytrain})
        test_accuracy, test_loss = sess.run([accuracy, loss], feed_dict={x: Xtest, y: Ytest})

    tf.reset_default_graph()

    return train_accuracy, train_loss, test_accuracy, test_loss


# def fit_fun(part_x):
#     train_accuracy, train_loss, test_accuracy, test_loss = train(part_x)
#     return (train_accuracy, train_loss)


def fit_fun(part_x):
    train_accuracy, train_loss, test_accuracy, test_loss = train(part_x)
    return 1-test_accuracy


def use_part_train(part, epoch_num=200):
    # 使用产生的part进行训练
    part_x = part.reshape([part.size])  # w转为合适的权重格式
    x = tf.placeholder(tf.float64, [None, 900])
    y = tf.placeholder(tf.float64, [None, 6])
    prediction, all_w_b = get_perdict(x, part_x)

    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # 训练函数
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)  # 模型训练过程
    # 取每一行最大值做对比，看分类正确个数
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 准确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化变量
    init = tf.global_variables_initializer()

    all_w_b_list = []

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_num):
            sess.run(train_step, feed_dict={x: Xtrain, y: Ytrain})
            train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={x: Xtrain, y: Ytrain})
            test_accuracy, test_loss = sess.run([accuracy, loss], feed_dict={x: Xtest, y: Ytest})
            # Show the Model Capability
            print("训练后:Iter " + str(epoch + 1) + ", Testing Accuracy: " + str(
                test_accuracy) + ", Training Accuracy: " + str(
                train_accuracy))
            print("训练后:Iter " + str(epoch + 1) + ", Testing Loss: " + str(test_loss) + ", Training Loss: " + str(
                train_loss))
            print('\n')

            if epoch >= int(epoch_num / 2) and epoch % int(epoch_num / 2 / 10) == 0:
                all_w_b_list.append(sess.run(all_w_b))
    return all_w_b_list, [test_accuracy, test_loss]


def w_b_to_part_x(all_w_b):
    # 将w_b转为单挑矩阵格式
    size_list = [16, 4, 288, 8, 1152, 16, 4704, 6]  # 每一层对应的size
    change_list = []
    for i in range(len(size_list)):
        change_list.append(all_w_b[i].reshape(size_list[i]))
    return np.hstack(change_list)

def fun(item):
    t = time.time()
    a = create_part_x()
    all_w_b, test_a_l = use_part_train(a, item)
    t = time.time() - t
    test_a_l.append(t)

    # 记录数据
    with open("%s次的值.csv" % item, 'a', newline='') as csvfile:
        cr = csv.writer(csvfile)
        cr.writerow(test_a_l)
    print("数据记录完毕")

if __name__ == '__main__':
    import time
    import csv
    item = 500
    fun(500)


