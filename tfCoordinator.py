#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date	: 2019-08-28 18:26:10
# Author: Pntehan

import tensorflow as tf
import numpy as np
import threading
import time

# 线程中运行的程序，这个程序每隔1s判断是否需要停止打印自己的ID
def MyLoop(coord, work_id):
	# 使用tf.Coordinator类提供的协同工具判断是否需要停止打印自己的ID
	while not coord.should_stop():
		# 随即停止所有线程
		if np.random.rand() < 0.1:
			print("Stoping from id: {}\n".format(work_id))
			# 调用coord.request_stop()函数来通知其他线程停止
			coord.request_stop()
		else:
			# 打印当前线程的ID
			print("Working on id: {}\n".format(work_id))
		# pause 1s
		time.sleep(1)

# 声明一个tf.train.Coordinator类来协同多个线程
coord = tf.train.Coordinator()
# 声明创建5个线程
threads = [ threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5) ]
# 启动所有线程
for t in threads:
	t.start()
# 等待所有线程推出
coord.join(threads)

