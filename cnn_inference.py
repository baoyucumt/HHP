import tensorflow as tf
INPUT_NODE = 500
OUTPUT_NODE = 2
IMAGE_SIZE1 = 1
IMAGE_SIZE2 = 500
NUM_CHANNELS = 1
NUM_LABELS = 2
CONV1_DEEP = 32
CONV1_SIZE1 = 1
CONV1_SIZE2 = 30
CONV2_DEEP = 64
CONV2_SIZE1 = 1
CONV2_SIZE2 = 30
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    input_3d = tf.expand_dims(input_tensor, 1)
    input_tensor = tf.expand_dims(input_3d, -1)
    #第一层，卷积层
    with tf.compat.v1.variable_scope('layer1-conv1'):   ##tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.compat.v1.get_variable(
            "weight", [CONV1_SIZE1, CONV1_SIZE2, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.5)
        )
        conv1_biases = tf.compat.v1.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
        )
        
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        '''with tf.variable_scope('visualization'):
            # scale weights to [0 1], type is still float
            x_min = tf.reduce_min(conv1_weights)
            x_max = tf.reduce_max(conv1_weights)
            kernel_0_to_1 = (conv1_weights - x_min) / (x_max - x_min)
            # to tf.image_summary format [batch_size, height, width, channels]
            kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])
            # this will display random 3 filters from the 64 in conv1
            tf.summary.image('conv1/filters', kernel_transposed, max_outputs=3)
            layer1_image1 = conv1[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3,1,2,0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=16)'''
            
        #第二层，池化层
    with tf.compat.v1.name_scope('layer2-pool1'):
        pool1 = tf.compat.v1.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )
        
        #第三层卷积层
    with tf.compat.v1.variable_scope('layer3-conv2'):
        conv2_weights = tf.compat.v1.get_variable(
            "weight", [CONV2_SIZE1, CONV2_SIZE2, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.5)
        )
        conv2_biases = tf.compat.v1.get_variable(
            "bias", [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        #第四层池化层
        pool2 = tf.compat.v1.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.compat.v1.variable_scope('layer5-fc1'):
        fc1_weights = tf.compat.v1.get_variable(
            "weight", [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.7)
        )
        if regularizer != None:
            tf.compat.v1.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.compat.v1.get_variable(
            "bias", [FC_SIZE], initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.compat.v1.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.compat.v1.nn.dropout(fc1, 0.8)

    with tf.compat.v1.variable_scope('layer6-fc2'):
        fc2_weights = tf.compat.v1.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.7)
        )
        if regularizer != None:
            tf.compat.v1.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.compat.v1.get_variable(
            "bias", [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
    #return [logit,conv2]
