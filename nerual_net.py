import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


input_layer = 784
hidden_layer = 150
output_layer = 10
learning_rate = 0.1
epochs = 20
batch_size = 100

x = tf.placeholder(tf.float32, [None, input_layer])
y = tf.placeholder(tf.float32, [None, output_layer])

Weights = {
    'w1' : tf.Variable(tf.random_normal([input_layer, hidden_layer])),
    'w2' : tf.Variable(tf.random_normal([hidden_layer, output_layer]))
}
bias = {
    'b1' : tf.Variable(tf.random_normal([hidden_layer])),
    'b2' : tf.Variable(tf.random_normal([output_layer]))
}

layer_1 = tf.add(tf.matmul(x, Weights['w1']), bias['b1'])
layer_1_activated = tf.nn.sigmoid(layer_1)
layer_output = tf.add(tf.matmul(layer_1_activated, Weights['w2']), bias['b2'])
layer_output_activated = tf.nn.sigmoid(layer_output)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_output_activated, labels=y))
opt = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()

print 'Starting traing'

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        iteration = int(mnist.train.num_examples/batch_size)
        average_cost = 0.0
        for j in range(iteration):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, mCost = sess.run([opt, cost], feed_dict={x:x_batch, y:y_batch})

            average_cost += mCost
        average_cost /= iteration
        print('Iteration %02d'%i, 'Cost {:9}'.format(average_cost))

    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
    print('Traing Complete and model saved')

    correct_pred = tf.equal(tf.argmax(layer_output_activated, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print "Accuracy: {:.9f}".format(acc.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
