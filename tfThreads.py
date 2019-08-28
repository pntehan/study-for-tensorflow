#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date	: 2019-08-28 18:37:26
# Author: Pntehan

import tensorflow as tf

# 声明一个先进先出的队列，队列中最多有100个元素，类型为实数
queue = tf.FIFOQueue(100, "float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
# 第一个参数是被操作的队列，第二个参数表示了要启动的线程数量
qr = tf.train.QueueRunner(queue, [enqueue_op]*5)

# 将定义过的QueueRunner加入Tensorflow计算图指定的集合
# tf.train.add_queue_runner函数没有指定集合，则加入默认的tf.GraphKeys.QUEUE_RUNNER
tf.train.add_queue_runner(qr)
# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
	# 使用tf.train.Coordinator来协同启动的线程
	coord = tf.train.Coordinator()
	# 使用tf.train.start_queue_runners来启动所有线程
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	# 获取队列中的取值
	for _ in range(10):
		print(sess.run(out_tensor))
	coord.request_stop()
	coord.join(threads)




