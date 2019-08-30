#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date	: 2019-08-30 13:41:19
# Author: Pntehan

import tensorflow as tf

# # 创建TFRecord文件的帮助函数
# def _int64_feature(value):
# 	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# # 模拟海量数据情况下将数据写入不同的文件，num_shards定义了总共写入多少个文件
# # instances_per_shard定义了每个文件中有多少个数据
# num_shards = 2
# instances_per_shard = 2
# for i in range(num_shards):
# 	# 文件名格式0000n-of-0000m
# 	filename = ("data/path/to/data.tfrecords-%.5d-of-%.5d"%(i, num_shards))
# 	writer = tf.io.TFRecordWriter(filename)
# 	# 将数据封装成Example结构并写入TFRecord文件
# 	for j in range(instances_per_shard):
# 		example = tf.train.Example(features=tf.train.Features(feature={
# 			"i": _int64_feature(i),
# 			"j": _int64_feature(j)
# 			}))
# 		writer.write(example.SerializeToString())
# 	writer.close()

# 使用tf.train.match_filenames_once函数来获取文件列表
files = tf.train.match_filenames_once("data/path/to/data.tfrecords-*")
# 通过tf.train.string_input_producer函数创建输入队列
file_queue = tf.train.string_input_producer(files, shuffle=False)
# 读取并解析一个样本
reader = tf.TFRecordReader()
_, serializerd_example = reader.read(file_queue)
features = tf.parse_single_example(
	serializerd_example,
	features={
	"i": tf.FixedLenFeature([], tf.int64),
	"j": tf.FixedLenFeature([], tf.int64)
	})
with tf.Session() as sess:
	tf.local_variables_initializer().run()
	print(sess.run(files))
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	for i in range(8):
		print(sess.run([features["i"], features["j"]]))
	coord.request_stop()
	coord.join(threads)

