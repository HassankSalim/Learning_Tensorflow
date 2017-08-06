import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

Weights = {
    'conv1':tf.random_normal([5, 5, 1, 32])
    'conv2':tf.random_normal([5, 5, 32, 64])
    'w1'
}
