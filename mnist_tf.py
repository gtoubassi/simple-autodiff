import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist
import random
import skimage.transform

def resize_images(train_xs):
  # resize
  resized = np.zeros((train_xs.shape[0], 7, 7))
  for i in range(train_xs.shape[0]):
    fullim = train_xs[i].copy()
    fullim.resize((28,28))
    resized[i] = skimage.transform.downscale_local_mean(fullim, (4, 4))

  resized.resize(resized.shape[0], 7*7)
  return resized

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_xs = resize_images(mnist.train._images)
train_ys = mnist.train._labels
test_xs = resize_images(mnist.test._images)
test_ys = mnist.test._labels

x = tf.placeholder(tf.float32, [None, 49])

W = tf.Variable(tf.zeros([49, 10]))
y = tf.nn.softmax(tf.matmul(x, W))

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+1e-8), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(.5)
train_step = optimizer.minimize(cross_entropy)
grads_and_vars = optimizer.compute_gradients(cross_entropy, tf.trainable_variables())

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print('Training')

batch_size = 100
batch_indices = [i for i in range(train_xs.shape[0])]

print(sess.run(accuracy, feed_dict={x: train_xs, y_: train_ys}))

for epoch in range(100):
  random.shuffle(batch_indices)
  for i in range(0, len(batch_indices), batch_size):
    batch = batch_indices[i:(i + batch_size)]
    
    batch_xs = np.take(train_xs, batch, axis=0)
    batch_ys = np.take(train_ys, batch, axis=0)
  
    _, loss, grads = sess.run([train_step, cross_entropy, grads_and_vars], feed_dict={x: batch_xs, y_: batch_ys})
  if (epoch + 1) % 1 == 0:
    acc = sess.run(accuracy, feed_dict={x: train_xs, y_: train_ys})
    print("Epoch %d Accuracy %f" % (epoch, acc))
