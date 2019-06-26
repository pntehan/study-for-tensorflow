import tensorflow as tf
from numpy.random import RandomState
# 神经网络模型的效果以及优化目的是通过损失函数来定义的
# 经典的损失函数有交叉熵函数和平方差函数
# 前者适合在高纬度情况中，适合分类问题。而后者适合预测比较，适合回归问题
# tensorflow还支持自定义的损失函数
'''
a = tf.constant([1., 2.])
b = tf.constant([[3., 1.], [1., 2.]])
# x = tf.placeholder([None, 2, 2, 1], tf.float32)
result = tf.reduce_mean(tf.where(tf.greater(a, b), (a-b), (b-a)))
with tf.Session() as sess:
    print(sess.run(result))
'''
'''
# 回归问题示例
batch_size = 8
# 设置输入与输出节点
x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
y_ = tf.placeholder(tf.float32,shape=[None, 1], name="y-input")
#  设置前向传播过程
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y势力= tf.matmul(x, w1)
# 设置损失权重
loss_less , loss_more= 1, 10
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))
# 设置反向传播函数
# 设置学习率衰减，可减少运行成本避免误差震荡不收敛
global_step = tf.Variable(0)
lr = tf.train.exponential_decay(0.1, global_step=global_step, decay_steps=100, decay_rate=0.5, staircase=True)
train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
# 设置数据集
rdm = RandomState(1)
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1, x2) in X]
# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size)%dataSet_size
        end = min(start+batch_size, dataSet_size)
        _, y_pred, learn_rate = sess.run([train_step, y_, lr], feed_dict={x:X[start:end], y_:Y[start:end]})
        if (i+1)%100 == 0:
            print("------------------------------------------------------")
            print("真实值:{}\n预测值:{}\n当前学习率:{}".format(Y[start:end], y_pred, learn_rate))
'''
# 防止过拟合的正则化示例
'''
def get_weights(shape, lam):
    ''生成一个变量''
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lam)(var))
    return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义每层网络的节点数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
# 设置当前层和节点数
cur_layer = x
in_dimension = layer_dimension[0]
# 网络构建
for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weights = get_weights(shape=[in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weights)+bias)
    in_dimension = layer_dimension[i]
# 定义损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))
'''
# 滑动平均模型示例
# 定义一个变量用于计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)
# 定义一个滑动平均类，初始化衰减率为0.99和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
maintrain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 通过ema.average(v1)获取滑动平均之后的变量的取值，初始化后都为0
    print(sess.run([v1, ema.average(v1)]))
    # 更新v1的值到5
    sess.run(tf.assign(v1, 5))
    sess.run(maintrain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    # 更新step到1000
    sess.run(tf.assign(step, 1000))
    # 更新v1到10
    sess.run(maintrain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    sess.run(maintrain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
