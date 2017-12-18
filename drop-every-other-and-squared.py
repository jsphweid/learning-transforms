# drop-every-other-and-squared:
#     [5, 2, 7, 44] -> [25, 49]
#     [0, 1, 1, 0, 0, 1, 1, 0] -> [0, 1, 0, 1]
#     [2, 4, 22, 6, 4, 33] -> [4, 484, 16]
#     etc... (except pick one length)

# import important libraries
import numpy as np
import tensorflow as tf 
from random import randint

# hyperparameters and config
INPUT_SIZE = 6
LABEL_SIZE = int(INPUT_SIZE / 2)
learning_rate = 0.001
num_steps = 5000
batch_size = 50
display_step = 10
NUM_HIDDEN_LAYER_NEURONS = 1024

# the source of all our inputs (X) and labels (Y)
def getRandomDataPair():
    input_data = []
    label_data = []
    for i in range(INPUT_SIZE):
        randomInt = randint(0, 4)
        input_data.append(randomInt)
        if (i % 2 == 0):
            label_data.append(randomInt ** 2)
    return [input_data, label_data]

def getBatch(size):
    inputs = []
    labels = []
    for i in range(size):
        x_input, y_label = getRandomDataPair()
        inputs.append(x_input)
        labels.append(y_label)
    return [inputs, labels]

# building the actual computation graph
X = tf.placeholder("float", [None, INPUT_SIZE])
Y = tf.placeholder("float", [None, LABEL_SIZE])

hidden1_weights = tf.Variable(tf.random_normal([INPUT_SIZE, NUM_HIDDEN_LAYER_NEURONS]))
hidden1_biases = tf.Variable(tf.random_normal([NUM_HIDDEN_LAYER_NEURONS]))
hidden1_layer = tf.matmul(X, hidden1_weights) + hidden1_biases

hidden2_weights = tf.Variable(tf.random_normal([NUM_HIDDEN_LAYER_NEURONS, NUM_HIDDEN_LAYER_NEURONS]))
hidden2_biases = tf.Variable(tf.random_normal([NUM_HIDDEN_LAYER_NEURONS]))
hidden2_layer = tf.matmul(hidden1_layer, hidden2_weights) + hidden2_biases

output_weights = tf.Variable(tf.random_normal([NUM_HIDDEN_LAYER_NEURONS, LABEL_SIZE]))
output_biases = tf.Variable(tf.random_normal([LABEL_SIZE]))
output_layer = tf.matmul(hidden2_layer, output_weights) + output_biases

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

