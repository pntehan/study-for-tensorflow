import tensorflow as tf

# tf.Variable是变量，需要指定其value，shape和type
# 变量的本质是张量的计算，输出的结果是tf.Variable计算出的张量
# 因为变量是一种特殊的张量，所以它也具有唯一类型
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y))




