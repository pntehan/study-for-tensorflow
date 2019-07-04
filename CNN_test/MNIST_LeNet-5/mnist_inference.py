import tensorflow as tf
# 定义神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CLANNELS = 1
NUM_LABELS = 10
# 定义卷积层参数
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    '''
    input_tensor:输入为28X28X1的原始MNIST图片像素
    train:区别训练和测试
    regularizer:是否使用正则化
    '''
    # 第一层：卷积层
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable(
            name="weight", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CLANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            name="bias", shape=[CONV1_DEEP], initializer=tf.truncated_normal_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(
            input=input_tensor, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME"
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # 第二层：池化层
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
    # 第三层：卷积层
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            name="weight", shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            name="bias", shape=[CONV2_DEEP], initializer=tf.truncated_normal_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(
            input=pool1, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME"
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # 第四层：池化层
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
    # 将图片拉直并且获取下一层输入的节点数
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])
    # 第五层：全连接层
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable(
            name="weight", shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 只有全连接层的权需要加入正则化
        if regularizer != None:
            tf.add_to_collection(name="losses", value=regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            name="biases", shape=[FC_SIZE], initializer=tf.truncated_normal_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    # 第六层：全连接层
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable(
            name="weight", shape=[FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            name="bias", shape=[NUM_LABELS], initializer=tf.truncated_normal_initializer(0.1)
        )
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    # 输出
    return logit














