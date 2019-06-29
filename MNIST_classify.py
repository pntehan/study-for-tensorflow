import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 设置全局变量
INPUT_NODE = 784
OUTOUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LR_BASE = 0.8
LR_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    '''
    一个辅助函数，给定神经网络的输入和所有输出参数，计算神经网络的前向传播结果
    定义一个使用RELU激活函数的三层全链接神经网络，通过加入隐藏层实现多层网络结构
    通过RELU实现去线性化
    这个函数支持传入用于计算参数平均值的类，方便在测试中使用和滑动平均模型
    '''
    if avg_class == None:
        # 没有提供滑动平均类的时候，直接使用参数当前的取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用滑动平均类的average函数计算得出变量的滑动平均值，再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2)) +  avg_class.average(biases2)

def train(mnist):
    '''训练模型过程'''
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # 生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)
    )
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTOUT_NODE], stddev=0.1)
    )
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTOUT_NODE]))
    # 计算当前参数下神经网络前向传播的结果，avg_class is None，所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)
    # 给定滑动平均衰减率和训练轮的变量，初始滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均，tf.trainable_variables返回计算图上所以没有指定trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算使用了滑动平均之后的结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    # 计算交叉熵刻画预测值与真实值之间的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    # 计算所有样例交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(biases1)
    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LR_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LR_DECAY
    )
    # 使用优化器优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 在训练神经网络模型时，每过一遍数据既要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值
    # 为了一次性完成多个操作，tensorflow提供了tf.control_dependencies和tf.group两种机制
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    # 判断两个张量 的每一维是否相等，如果相等返回True，否则返回False
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 这个运算首先将一个布尔类型的数值转换为实数型，然后计算平均值，这个平均值就是模型准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            if (i+1)%1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g"%(i+1, validate_acc))
            # 产生这一轮使用的训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys}) 
        # 训练结束，在测试数据上检测神经网络模型的最终准确性
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g"%(TRAINING_STEPS, test_acc))

def main(argv=None):
    '''声明处理MNIST数据集的类，这个类在初始化时下载或者导入数据'''
    mnist = input_data.read_data_sets("./data/mnist_data", one_hot=True)
    train(mnist)

# tensorflow提供了一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()





