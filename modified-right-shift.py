# modified-right-shift:
#     [0, 1, 1, 0] -> [0, 0, 1, 1]
#     [0, 1, 1] -> [1, 0, 1]
#     [0, 1, 0, 0, 0, 1, 0, 1, 1] -> [1, 0, 1, 0, 0, 0, 1, 0, 1]
#     etc...

# import important libraries
import numpy as np
import tensorflow as tf 
from random import getrandbits

# Adding these two lines because tensorflow wasn't compiled on this machine (used pip install)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# the source of all our inputs (X) and labels (Y)
def getRandomDataPair():
    random_binary = list(map(int, bin(getrandbits(NUM_BITS))[2:].zfill(NUM_BITS)))
    shifted = list(map(int, random_binary[-1:] + random_binary[:-1]))
    return [random_binary, shifted]

def getBatch(size):
    inputs = []
    labels = []
    for i in range(size):
        x_input, y_label = getRandomDataPair()
        inputs.append(x_input)
        labels.append(y_label)
    return [inputs, labels]

# hyperparameters and config
NUM_BITS = 40
learning_rate = 0.001
num_steps = 5000
batch_size = 10
display_step = 10
num_hidden_layer_neurons = NUM_BITS ** 2

# building the actual computation graph
X = tf.placeholder("float", [None, NUM_BITS])
Y = tf.placeholder("float", [None, NUM_BITS])

hidden_weights = tf.Variable(tf.random_normal([NUM_BITS, num_hidden_layer_neurons]))
hidden_biases = tf.Variable(tf.random_normal([num_hidden_layer_neurons]))
hidden_layer = tf.matmul(X, hidden_weights) + hidden_biases

output_weights = tf.Variable(tf.random_normal([num_hidden_layer_neurons, NUM_BITS]))
output_biases = tf.Variable(tf.random_normal([NUM_BITS]))
output_layer = tf.matmul(hidden_layer, output_weights) + output_biases

loss_op = tf.reduce_mean(tf.square(tf.subtract(Y, output_layer)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# GO!
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(1, num_steps + 1):

        # generate data
        batch_x, batch_y = getBatch(batch_size)

        # run the entire graph
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # occasionally print out progress
        if step % display_step == 0 or step == 1:
            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= {:.4f}".format(loss))

    # display a real life output
    test_input, test_label = getBatch(1)
    print("test input is: ", test_input)
    print("test label is: ", test_label)
    print("actual predicted result is: ", sess.run(output_layer, feed_dict={X: test_input}))

