
import os
#import tensorflow.python.framework.nn_ops as ttf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cnn_inference

DATA_SIZE = 650
BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.9

MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"
acc_print=[]
acc_printx=[]
acc_printy=[]

def train(trainx,trainy):
    x = tf.compat.v1.placeholder(tf.float32, [None, cnn_inference.IMAGE_SIZE2], name="x-input")
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, cnn_inference.OUTPUT_NODE], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #y = cnn_inference.inference(x, 0, None)
    '''正则化'''
    y = cnn_inference.inference(x, 1, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )   #滑动平均更新参数
    variable_averages_op = variable_averages.apply(
        tf.compat.v1.trainable_variables()
    )
    # cross_entropy_mean = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y)  #计算交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  #张量的平均值
    '''正则化'''
    loss = cross_entropy_mean + tf.compat.v1.add_n(tf.compat.v1.get_collection('losses'))       #tf.add_n()列表内元素相加，tf.get_collection()获得名称为losses集合的所有元素
    #loss = cross_entropy_mean

    learning_rate = tf.compat.v1.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DATA_SIZE / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )                               #tf.train.exponential()学习率衰减函数

    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)  #梯度下降法

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.compat.v1.train.Saver()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False

    with tf.compat.v1.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            start = (i * BATCH_SIZE) % DATA_SIZE
            end = min(start + BATCH_SIZE, DATA_SIZE)

            # 每次选取batch_size个样本进行训练
            # _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: trainx[start: end], y_: trainy[start: end]})
            _, step = sess.run([train_op, global_step], feed_dict={x: trainx[start: end], y_: trainy[start: end]})

            # 通过选取样本训练神经网络并更新参数
            #sess.run(train_step, feed_dict={x: trainx[start:end], y_: trainy[start:end]})
            # 每迭代1000次输出一次日志信息
            
            # 计算所有数据的交叉熵
           
            # train_accuracy = sess.run(accuracy, feed_dict={x: X, y_: Y})
            # 输出交叉熵之和
            #print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
            
            if i % 1000 == 0:
                total_cross_entropy = sess.run(loss, feed_dict={x: trainx, y_: trainy})
            #测试正确率
                correct_prediction_train = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                acc_show= tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))
                acc_train = sess.run(acc_show, feed_dict={x: trainx, y_: trainy})
                acc_print.append(acc_train)
                acc_printy.append(0.5)
                acc_printx.append(i)
            # total_mse = sess.run(loss, feed_dict={x: trainx, y_: trainy})
                print("After %d training step(s), loss on training "
                      "batch is %g,train accuracy = %g" % (i, total_cross_entropy,acc_train))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print("save the model")
        print(acc_print)
        plt.title("trend of accuracy")
        plt.plot(acc_printx,acc_print,color='skyblue')
        plt.plot(acc_printx,acc_printy,color='cyan')
        plt.show()
                
def convert_to_one_hot(Y, C):       #转为1位热码编码
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def main(argv=None):
    #tf.reset_default_graph()
    tf.compat.v1.reset_default_graph();
    # mnist = input_data.read_data_sets("path/to/mnist_data", one_hot=True)  
    string0=np.loadtxt('total_data.txt',dtype=np.float32)
    train_y=string0[:,0].reshape(-1,1).T  #1行
    train_x=string0[:,1:].reshape(650,-1)
    '''string1=np.loadtxt('test_data.txt',dtype=np.float32)
    test_y=string1[:,0].reshape(-1,1).T  #1行
    test_x=string1[:,1:].reshape(120,-1)  
    test_y = convert_to_one_hot(test_y.astype(int), 2).T #一行代表一个样本，1位热码编码'''
    train_y= convert_to_one_hot(train_y.astype(int), 2).T
    

    train(train_x,train_y)



if __name__ == '__main__':
    main()
