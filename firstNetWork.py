import tensorflow as tf
from numpy.random import RandomState
# 定义全局变量
BATCH_SIZE = 8
LR = 1e-3
STEPS = 5000
# 定义变量与placeholder
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
# 定义神经网络的前向传播过程
a = tf.matmul(x, w1)
y_pred = tf.matmul(a, w2)
# 定义损失函数与反向传播算法
y_logit = tf.sigmoid(y_pred)
cross_entropy = -tf.reduce_mean(
    y*tf.log(tf.clip_by_value(y_logit, 1e-10, 1.0))+(1-y)*tf.log(tf.clip_by_value(1-y_logit, 1e-10, 1.0))
)
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
# 定义模拟数据
rdm = RandomState(1)
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)
Y = [[int(x1+x2<1)] for (x1, x2) in X]
# 激活张量开始训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("Before Training...")
    print("w1:\n{}\nw2:\n{}".format(sess.run(w1), sess.run(w2)))
    for epoch in range(100):
        for i in range(STEPS):
            start = (i*BATCH_SIZE)%dataSet_size
            end = min(start+BATCH_SIZE, dataSet_size)
            sess.run(train_step, feed_dict={x: X[start:end], y: Y[start:end]})
            if (i+1)%1000 == 0:
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y: Y})
                print("EPOCH [%2d] After %d training steps, cross entropy on all data is %g"%(epoch+1, i+1, total_cross_entropy))
    print("After Training...")
    print("w1:\n{}\nw2:\n{}".format(sess.run(w1), sess.run(w2)))
