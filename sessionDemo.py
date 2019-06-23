import tensorflow as tf

a = tf.constant([1., 2.], name="a", dtype=tf.float32)
b = tf.constant([3., 4.], name="b", dtype=tf.float32)
result = tf.add(a, b, name="add")

# session会话用于获取资源，释放资源
with tf.Session() as sess:
    print(sess.run(result))
    # 此处方法用于提取张量的值，与上个方法有一样的功能
    with sess.as_default():
        print(result.eval())








