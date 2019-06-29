from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
# 载入手写字符体数据
mnist = input_data.read_data_sets("./data/mnist_data/", one_hot=True)
# 打印数据部分信息
print("Training data size: ", mnist.train.num_examples)
print("Validating data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)
# print("Example training data: ", mnist.train.images[0])
num = mnist.train.images[10].reshape(28, 28)
img = Image.fromarray(num*255)
img.show()
print("Example training data label: ", mnist.train.labels[10])




