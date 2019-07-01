import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
# # 保存模型
# v1 = tf.Variable(tf.constant(1.0, shape=[1], name="v1"))
# v2 = tf.Variable(tf.constant(2.0, shape=[1], name="v2"))
# result = v1 + v2
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(result))
#     saver.save(sess, './model/model1.ckpt')
# # 导入模型
# v1 = tf.Variable(tf.constant(1.0, shape=[1], name="v1"))
# v2 = tf.Variable(tf.constant(2.0, shape=[1], name="v2"))
# result = v1 + v2
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
# # 导入计算图和模型
# with tf.Session() as sess:
#     # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
#     saver.restore(sess, './model/model1.ckpt')
#     print(sess.run(result))

# # 直接加载持久华的图
# saver = tf.train.import_meta_graph('./model/model1.ckpt.meta')
# with tf.Session() as sess:
#     saver.restore(sess, './model/model1.ckpt')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))

# #  模型保存在滑动平均变量上的使用
# v = tf.Variable(0, dtype=tf.float32, name="v")
# for variable in tf.global_variables():
#     print(variable.name)
# # 声明滑动平均模型
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_averages_op = ema.apply(tf.global_variables())
# for variable in tf.global_variables():
#     print(variable.name)
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     sess.run(tf.assign(v, 10))
#     sess.run(maintain_averages_op)
#     saver.save(sess, 'model/model2.ckpt')
#     print(sess.run([v, ema.average(v)]))
# # 通过变量重命名读取变量的graph_util滑动平均值
# v = tf.Variable(0, dtype=tf.float32, name="v")
# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
# with tf.Session() as sess:
#     saver.restore(sess, 'model/model2.ckpt')
#     print(sess.run(v))
# # 生成所需的变量重命名字典
# v = tf.Variable(0, dtype=tf.float32, name="v")
# ema = tf.train.ExponentialMovingAverage(0.99)
# print(ema.variables_to_restore())
# saver = tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess:
#     saver.restore(sess, 'model/model2.ckpt')
#     print(sess.run(v))

# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
# result = v1 + v2
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     graph_def = tf.get_default_graph().as_graph_def()
#     # print(graph_def)
#     output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
#     # print(output_graph_def)
#     with tf.io.gfile.GFile('model/combined_model.pb', 'wb') as f:
#         f.write(output_graph_def.SerializeToString())
# # 读取
# with tf.Session() as sess:
#     model_filename = 'model/combined_model.pb'
#     with gfile.FastGFile(model_filename, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     result = tf.import_graph_def(graph_def, return_elements=["add:0"])
#     print(sess.run(result))
