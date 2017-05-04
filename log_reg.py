import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

learning_rate = 0.0001
epochs = 35
batch_size = 100
display_time = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

cost = tf.reduce_mean(tf.reduce_sum(-y*tf.log(pred), reduction_indices=1))
opt = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoche in range(epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for j in range(total_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, c = sess.run([opt, cost], feed_dict={x:x_batch, y:y_batch})

            avg_cost += c

        avg_cost /= total_batch

        if not (epoche+1) % display_time:
            print 'Epoch: %04d '%(epoche+1), 'Cost {:.9f}'.format(avg_cost)
    print('Optimization Finished')

    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
    print('Saved')

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print "Accuracy: {:.9f}".format(accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))
