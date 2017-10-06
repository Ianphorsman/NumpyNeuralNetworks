import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )

with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("Layer1"):
    W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
    b_conv1 = bias_variable([32], "b_conv1")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("Layer2"):
    W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
    b_conv2 = bias_variable([64], "b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("Layer3"):
    W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
    b_fc1 = bias_variable([1024], "b_fc1")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope("Dropout"):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("Layer4"):
    W_fc2 = weight_variable([1024, 10], "W_fc2")
    b_fc2 = bias_variable([10], "b_fc2")

with tf.name_scope("Output"):
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope("Cost"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y_,
            logits=y_conv
        )
    )
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    tf.summary.scalar('Cross Entropy', cross_entropy)

with tf.name_scope("Acc"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("Accuracy", accuracy)

merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter('/tmp/mnist/1')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter('/tmp/mnist/2', graph=tf.get_default_graph())

    for i in range(10000):
        batch = mnist.train.next_batch(50)

        _, c, summary = sess.run(
            [optimizer, accuracy, merged_summary],
            feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 0.1
            }
        )
        if i % 250 == 0:
            print("Iteration: ", i, c)

        summary_writer.add_summary(summary, i)



