import tensorflow as tf

# 定义神经网络结构的相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    '''
    通过tf.get_varibale函数来获取变量，在训练时会创建这些变量，测试时可以通过保存的模型加载这些变量
    而且加载时利用滑动平均变量重命名，测试时直接使用变量的滑动平均值
    '''
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    '''
    声明第一层神经网络的变量并完成前向传播过程
    '''
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2




