"""
X 841=29*29维特征的文本  Y label[1.,0.]  [0.,1.]
第一层：卷积层    输入的是29*29*1的文本特征
    过滤器尺寸 3*3 深度为5 不使用全零填充 步长为1
    输出为29-3+1=27*27 深度为5
    参数w = 3*3*1*5  b = 5
第二层：池化层   输入27*27*5的矩阵
    过滤器大小 3*3 步长为 3
    输出9*9*5
第三层：卷积层   输入9*9*5的矩阵
    过滤器尺寸 2*2 深度为12 不使用全零填充 步长为1
    参数w = 2*2*5*12 b = 12
    输出9-2+1=8*8*12
第四层：池化层   输入8*8*12
    过滤器大小 2*2 步长 2
    输出4*4*12
第五层：全连接层 输入4*4*12
    过滤器尺寸 4*4*80 不使用全零填充 步长为1
    参数w = 4*4*12*80 b = 80
    输出1*1*80
第六层：全连接层
    输入80
    w = 80*56 b = 56
    输出56
输出层：
    输入56
    w = 56*2 b=2
    输出2

"""
import tensorflow as tf
import numpy as np
import csv
import logging
#########配置log日志方便打印#############

LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)


num_input = 841
num_classes = 2
dropout = 0.5

learning_rate = 0.001
batch_size = 100
num_steps = 10000
display_step = 10

data_file = '../../data/roumorPCAdata.npy'

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


X_batch = tf.Variable(tf.constant(0.0), dtype=tf.float32)
Y_batch = tf.Variable(tf.constant(0.0), dtype=tf.float32)
#权重和偏向
weigths={
    "w1":tf.Variable(tf.random_normal([3, 3, 1, 5])),
    "w2":tf.Variable(tf.random_normal([2, 2, 5, 12])),
    "w3":tf.Variable(tf.random_normal([4*4*12,80])),
    "w4":tf.Variable(tf.random_normal([80,56])),
    "w5":tf.Variable(tf.random_normal([56,2]))
}
bias = {
    "b1":tf.Variable(tf.random_normal([5])),
    "b2":tf.Variable(tf.random_normal([12])),
    "b3":tf.Variable(tf.random_normal([80])),
    "b4":tf.Variable(tf.random_normal([56])),
    "b5":tf.Variable(tf.random_normal([2]))
}

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')

#定义操作
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 29, 29, 1])

    conv1 = conv2d(x,weights['w1'],biases['b1'])
    conv1 = maxpool2d(conv1,k=3)

    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    conv2 = maxpool2d(conv2, k=2)

    fc3 = tf.reshape(conv2,[-1,weights['w3'].get_shape().as_list()[0]])
    fc3 = tf.add(tf.matmul(fc3, weights['w3']), biases['b3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, dropout)

    fc4 = tf.add(tf.matmul(fc3, weights['w4']), biases['b4'])
    fc4 = tf.nn.relu(fc4)
    fc4 = tf.nn.dropout(fc4, dropout)

    fc5 = tf.add(tf.matmul(fc4, weights['w5']), biases['b5'])
    # fc5 = tf.nn.relu(fc5)

    return fc5

# Construct model
logits = conv_net(X, weigths, bias, dropout)

prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

#读取事先准备好的数据
data = np.load(data_file)

training_data = data[0]
training_label = data[1]
train_len = len(training_data)

validation_data = data[2]
validation_label = data[3]

testing_data = data[4]
testing_label = data[5]

count =0
with tf.Session() as sess:
    logger.info("-----------")

    sess.run(init)
    start = 0
    end = batch_size
    for step in range(1, num_steps + 1):
        sess.run(train_op, feed_dict={X: training_data[start:end], Y: training_label[start:end]})

        if step % 100 == 0 or step == 1:
            va_loss, va_acc = sess.run([loss_op, accuracy], feed_dict={X: validation_data, Y: validation_label})
            tra_loss,tra_acc = sess.run([loss_op,accuracy],feed_dict={X:training_data[start:end],Y:training_label[start:end]})
            print("Step " + str(step) + ", Minibatch validation Loss= " + "{:.4f}".format(va_loss) \
                +", Minibatch train Loss="+ "{:.4f}".format(tra_loss) + ", validation Accuracy= " + \
                  "{:.3f}".format(va_acc) + ",train Accuracy = "+"{:.3f}".format(tra_acc))
        start = end
        if start == train_len:
            start = 0
        end = start+batch_size
        if end > train_len:
            end = train_len
    acc = sess.run(accuracy, feed_dict={X: testing_data, Y: testing_label})
    print("Final test accuracy = %.lf%%" % (acc * 100))







