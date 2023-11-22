""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf

import math
import numpy as np
import quantize_weights

import cv2
from numpy import array


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Input Data
# fname = 'C:/Users/ybrot/PycharmProjects/image/dataset/a08.jpg'
# src = cv2.imread(fname, flags=cv2.IMREAD_GRAYSCALE)
# 흑백 반전
# dst = cv2.resize(src, dsize=(28, 28), interpolation=cv2.INTER_AREA)
# bw = cv2.bitwise_not(dst)
# input = array(bw).reshape(1, 784) / 255

# Training Parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 128
# batch_size = 256
display_step = 10

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.7 # Dropout, probability to keep units

##quantize weight
method = 'quantize'  # 'real' 'quantize' 'binary''radix''radix_ReLU''quantize''quantized_relu''binary'
nb = 3
H = 1
##


# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
max_avg = tf.placeholder(tf.float32) # number of layer

# Create some wrappers for simplicity
def conv2d(x, W, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    out = quantize_weights.quantize_weights(x, method='radix_ReLU', H=H, nb=nb)

    return out


def conv2d_max(x, W, fixed_max, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    out = quantize_weights.quantize_weights(x, method='radix_ReLU_fixed', H=H, nb=nb, fixed_max=fixed_max)

    return out

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def meanpool2d(x, k=2):
    # MaxPool2D wrapper

    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv1(x, w):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.bias_add(conv, b)
    conv = maxpool2d(conv, k=2)
    return conv


def conv2(x, w):
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.bias_add(conv, b)
    conv = maxpool2d(conv, k=2)
    return conv


def fully1(x, w):
    fc1 = tf.reshape(x, [-1, w.get_shape().as_list()[0]]) # w의 모양대로 reshape
    fc1 = tf.matmul(fc1, w)
    # fc1 = quantize_weights.quantize_weights(fc1, method='radix_ReLU', H=H, nb=nb)
    # fc1 = quantize_weights.quantize_weights(fc1, method='radix_ReLu_fixed', H=H, nb=nb, fixed_max=)
    return fc1


def out(x, w):
    out = tf.matmul(x, w)
    return out

'''
# Create model
def conv_net(x, weights, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # conv1 = meanpool2d(x, k=2)
    # print('===x===:', x)
    # Convolution Layer
    conv1 = conv2d(x,  quantize_weights.quantize_weights(weights['wc1'], method=method, H=H, nb=nb))
    # print('===conv1===:', conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # print('===conv1===2:', conv1)
    # Convolution Layer
    conv2 = conv2d(conv1, quantize_weights.quantize_weights(weights['wc2'], method=method, H=H, nb=nb))
    # print('===conv2===:', conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    # print('===conv2===2:', conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.matmul(fc1, quantize_weights.quantize_weights(weights['wd1'], method=method, H=H, nb=nb))
    # fc1 = tf.matmul(fc1, weights['wd1'])

    fc1 = quantize_weights.quantize_weights(fc1, method='radix_ReLU', H=H, nb=nb)
    # fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.matmul(fc1, quantize_weights.quantize_weights(weights['out'], method=method, H=H, nb=nb))
    # out = tf.matmul(fc1, weights['out'])
    return out
'''


# def conv_net_maxavg(x, weights, dropout, fixed_max1, fixed_max2, fixed_max3):
#     x = tf.reshape(x, shape=[-1, 28, 28, 1])
#     # conv1 = meanpool2d(x, k=2)
#     conv1 = conv2d_max(x, quantize_weights.quantize_weights(weights['wc1'], method=method, H=H, nb=nb), fixed_max1)
#     conv1 = maxpool2d(conv1, k=2)
#     conv2 = conv2d_max(conv1, quantize_weights.quantize_weights(weights['wc2'], method=method, H=H, nb=nb), fixed_max2)
#     conv2 = maxpool2d(conv2, k=2)
#
#     fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#     fc1 = tf.matmul(fc1, quantize_weights.quantize_weights(weights['wd1'], method=method, H=H, nb=nb))
#     fc1 = quantize_weights.quantize_weights(fc1, method='radix_ReLU_fixed', H=H, nb=nb, fixed_max=fixed_max3)
#     fc1 = tf.nn.dropout(fc1, dropout)
#     # Output, class prediction
#     out = tf.matmul(fc1, quantize_weights.quantize_weights(weights['out'], method=method, H=H, nb=nb))
#     return out


def conv_net_max(x, weights, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # conv1 = meanpool2d(x, k=2)
    conv1 = conv2d(x, quantize_weights.quantize_weights(weights['wc1'], method=method, H=H, nb=nb))
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, quantize_weights.quantize_weights(weights['wc2'], method=method, H=H, nb=nb))
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, quantize_weights.quantize_weights(weights['wd1'], method=method, H=H, nb=nb))
    max = tf.reduce_max(fc1)
    fc1 = quantize_weights.quantize_weights(fc1, method='radix_ReLU_fixed', H=H, nb=nb, fixed_max=max)
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.matmul(fc1, quantize_weights.quantize_weights(weights['out'], method=method, H=H, nb=nb))
    return out


# Store layers weight & bias
weights = {
     # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    # 'wd1': tf.Variable(tf.random_normal([14 * 14 * 1, 256])),
    # 1024 inputs, 10 outputs (class prediction)
    # 'out': tf.Variable(tf.random_normal([256, num_classes]))
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

# Construct model
# logits = conv_net(X, weights, keep_prob)
logits = conv_net_max(X, weights, keep_prob)
prediction = tf.nn.softmax(logits)
# pre_max = tf.argmax(prediction, 1)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))  # loss_op = cost function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # optimizer : cost function 가 최소가 되는 W와 b들을 찾아줌
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))  # tf.equal : Returns the truth value of (x == y) element-wise. #tf.argmax : Returns the index with the largest value across axes of a tensor.
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})

            # print("Step " + str(step) + ", Minibatch Loss= " +
            #       "{:.4f}".format(loss) + ", Training Accuracy= " +
            #       "{:.3f}".format(acc))

            print("{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))

    # model.save_weights('mnist_checkpoint')

    # Save weights
    # save_file = './train_model'
    # saver = tf.train.Saver()
    # saver.save(sess, save_file)
    # print('Trained Model Saved.')

    # Session을 닫지 않고 계속해서 숫자를 인식
    # while True:
    #     print('Enter 키를 누르면 종료')
    #     q = input()
    #     if q == '':
    #         break






'''
    sess.run(tf.assign(weights['wc1'], quantize_weights.quantize_weights(weights['wc1'], method=method, H=H, nb=nb)))
    sess.run(tf.assign(weights['wc2'], quantize_weights.quantize_weights(weights['wc2'], method=method, H=H, nb=nb)))
    sess.run(tf.assign(weights['wd1'], quantize_weights.quantize_weights(weights['wd1'], method=method, H=H, nb=nb)))
    sess.run(tf.assign(weights['out'], quantize_weights.quantize_weights(weights['out'], method=method, H=H, nb=nb)))


    # 최대값 분포
    conv1_out = np.zeros(num_steps)
    conv2_out = np.zeros(num_steps)
    fc1_out = np.zeros(num_steps)
    # out_out = np.zeros(num_steps)

    convolution1 = conv1(quantize_weights.quantize_weights(X, method='radix_ReLU', H=H, nb=nb), quantize_weights.quantize_weights(weights['wc1'], method=method, H=H, nb=nb))
    convolution2 = conv2(quantize_weights.quantize_weights(convolution1, method='radix_ReLU', H=H, nb=nb), quantize_weights.quantize_weights(weights['wc2'], method=method, H=H, nb=nb))
    fc1 = fully1(quantize_weights.quantize_weights(convolution2, method='radix_ReLU', H=H, nb=nb), quantize_weights.quantize_weights(weights['wd1'], method=method, H=H, nb=nb))
    # out = out(fc1, quantize_weights.quantize_weights(weights['out'], method=method, H=H, nb=nb))



    conv1max = tf.reduce_max(convolution1)
    conv2max = tf.reduce_max(convolution2)
    fc1max = tf.reduce_max(fc1)
    # outmax = tf.reduce_max(out)

    for i in range(num_steps):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        #sess.run(tf.global_variables_initializer())

        conv1_out[i] = sess.run(conv1max, feed_dict={X: batch_x})
        conv2_out[i] = sess.run(conv2max, feed_dict={X: batch_x})
        fc1_out[i] = sess.run(fc1max, feed_dict={X: batch_x})
        # print("covn1_out:", conv1_out[i])
        # print("covn2_out:", conv2_out[i])
        # print("fc1_out:", fc1_out[i])

        conv1_max_avg = 0
        conv2_max_avg = 0
        fc1_out_avg = 0

    for i in range(num_steps):
        conv1_max_avg += conv1_out[i]
        conv2_max_avg += conv2_out[i]
        fc1_out_avg += fc1_out[i]
    conv1_max_avg = conv1_max_avg / num_steps
    conv2_max_avg = conv2_max_avg / num_steps
    fc1_out_avg = fc1_out_avg / num_steps

    print("conv1_max_avg:", conv1_max_avg)
    print("conv2_max_avg:", conv2_max_avg)
    print("fc1_out_avg:", fc1_out_avg)

    print("Testing Accuracy with max_avg activation:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))

    


# 학습이 끝난 뒤 max_avg값을 사용
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # logits_max = conv_net(X, weights, max_avg)
    # logits_max = conv_net_maxavg(X, weights, max_avg, fc1_out_avg)
    logits_max = conv_net_maxavg(X, weights, max_avg, conv1_max_avg, conv2_max_avg, fc1_out_avg)
    prediction_max = tf.nn.softmax(logits_max)
    loss_op_max = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_max, labels=Y))
    optimizer_max = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op_max = optimizer.minimize(loss_op_max)
    correct_pred_max = tf.equal(tf.argmax(prediction_max, 1), tf.argmax(Y, 1))
    accuracy_max = tf.reduce_mean(tf.cast(correct_pred_max, tf.float32))

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_op_max, feed_dict={X: batch_x, Y: batch_y, max_avg: dropout})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op_max, accuracy_max], feed_dict={X: batch_x,
                                                                         Y: batch_y,
                                                                         max_avg: 1.0})

            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

            # print("{:.3f}".format(acc))

    print("conv1_max_avg:", conv1_max_avg)
    print("conv2_max_avg:", conv2_max_avg)
    print("fc1_out_avg:", fc1_out_avg)
    print("Testing Accuracy with max_avg activation:", \
          sess.run(accuracy_max, feed_dict={X: mnist.test.images[:256],
                                            Y: mnist.test.labels[:256],
                                            max_avg: 1.0}))


    '''


'''
    # Save weights
    save_file = './train_model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, save_file)
    print('Trained Model Saved.')
'''









'''
# 매번 학습마다 max값을 사용
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for i in range(num_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        fc1_out[i] = sess.run(fc1max, feed_dict={X: batch_x})

    # logits_max = conv_net(X, weights, max_avg)
    logits_max = conv_net_maxavg(X, weights, max_avg, fc1_out[0])
    prediction_max = tf.nn.softmax(logits_max)
    loss_op_max = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_max, labels=Y))
    optimizer_max = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op_max = optimizer.minimize(loss_op_max)
    correct_pred_max = tf.equal(tf.argmax(prediction_max, 1), tf.argmax(Y, 1))
    accuracy_max = tf.reduce_mean(tf.cast(correct_pred_max, tf.float32))

    for step in range(num_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # fc1_out = np.zeros(num_steps)
        # fc1max = tf.reduce_max(fc1)
        # fc1_out[step] = sess.run(fc1max, feed_dict={X: batch_x})

        # Run optimization op (backprop)
        sess.run(train_op_max, feed_dict={X: batch_x, Y: batch_y, max_avg: 1.0})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op_max, accuracy_max], feed_dict={X: batch_x,
                                                                         Y: batch_y,
                                                                         max_avg: 1.0})

            # print("Step " + str(step) + ", Minibatch Loss= " +
            #       "{:.4f}".format(loss) + ", Training Accuracy= " +
            #       "{:.3f}".format(acc))

            print("{:.3f}".format(acc))

    print("fc1_out_avg:" + format(fc1_out_avg))
    print("Testing Accuracy with max_avg activation:", \
          sess.run(accuracy_max, feed_dict={X: mnist.test.images[:256],
                                            Y: mnist.test.labels[:256],
                                            max_avg: 1.0}))
'''



# for index, pixel in enumerate(input[0]):
#
#     if index % 28 == 0:
#
#         print('\n')
#
#     else:
#
#         print("%10f" % pixel, end="")
