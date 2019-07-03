import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LR_BASE = 0.8
LR_DECAY = 0.99
REGU_RATE = 0.0001
TRAIN_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model3'

def train(mnist):
    '''训练神经网络'''
    # 定义输入输出
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    # 定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGU_RATE)
    # 前向传播
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    # 定义损失函数，学习率，滑动平均类及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    lr = tf.train.exponential_decay(LR_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LR_DECAY)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
    # 初始化持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if (i+1)%1000 == 0:
                print("After %d training steps, loss on training batch is %g."%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('data/mnist_data', one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()





