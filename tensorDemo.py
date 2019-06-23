import tensorflow as tf
# constant是一个计算，这个计算结果为一个张量，保存在变量a 中
# 张量具有name,shape,type三个属性，其中shape属性决定了张量的维度，而type属性是张量的数据类型
# 张量具有类型唯一性，不支持不同数据类型张量计算，则可在定义张量时定义其数据类型，避免错误
a = tf.constant([1., 2.], name="a", dtype=tf.float32)
b = tf.constant([3., 4.], name="b", dtype=tf.float32)
result = tf.add(a, b, name="add")
print(result)

with tf.Session() as sess:
    print(sess.run(result))









