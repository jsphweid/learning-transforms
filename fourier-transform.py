# can a NN learn the fourier transform...
# specifically, can it learn abs(fft(timeDomainSignal))
# if it learns from a "random signal", which is white noise, then the fft won't be anything specially, but still relevant?


# import important libraries
import numpy as np
import tensorflow as tf 
import random
from scipy.fftpack import fft

# hyperparameters and config
INPUT_SIZE = 32
FFT_SIZE = int(INPUT_SIZE / 2)
learning_rate = 0.01
num_steps = 50000
batch_size = 5
display_step = 10
NUM_HIDDEN_LAYER_NEURONS = 128
CONV_SIZE = 31

# the source of all our inputs (X) and labels (Y)
def getRandomDataPair():
    input_data = []
    label_data = []
    for i in range(INPUT_SIZE):
        input_data.append(random.uniform(-1, 1))
    label_data = abs(fft(input_data))[0:FFT_SIZE]
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
Y = tf.placeholder("float", [None, FFT_SIZE])

x_resized = tf.reshape(X, [-1, INPUT_SIZE, 1]) # -1 infers batch size, 2

with tf.name_scope('convolution-layer'):
    conv1_weights = tf.Variable(tf.random_normal([CONV_SIZE, 1, NUM_HIDDEN_LAYER_NEURONS]))
    conv1_biases = tf.Variable(tf.random_normal([NUM_HIDDEN_LAYER_NEURONS]))
    conv1 = tf.nn.conv1d(x_resized, conv1_weights, stride=1, padding='VALID')
    hidden1_conv1 = tf.nn.relu(conv1 + conv1_biases)

SIZE_AFTER_CONV = int(INPUT_SIZE - CONV_SIZE + 1)
NEXT_LAYER_SIZE = SIZE_AFTER_CONV * NUM_HIDDEN_LAYER_NEURONS

with tf.name_scope('flattened-after-convolution'):
    flattened = tf.reshape(hidden1_conv1, [-1, NEXT_LAYER_SIZE])

with tf.name_scope('output-layer'):
    output_weights = tf.Variable(tf.random_normal([NEXT_LAYER_SIZE, FFT_SIZE]))
    output_biases = tf.Variable(tf.random_normal([FFT_SIZE]))
    output_layer = tf.matmul(flattened, output_weights) + output_biases

loss_op = tf.reduce_mean(tf.square(tf.subtract(Y, output_layer)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

tf.summary.scalar('loss_op', loss_op)

# GO!
with tf.Session() as sess:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/', sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(1, num_steps + 1):

        # generate data
        batch_x, batch_y = getBatch(batch_size)

        # run the entire graph
        summary, _ = sess.run([merged, train_op], feed_dict={X: batch_x, Y: batch_y})
        train_writer.add_summary(summary, step)

        # occasionally print out progress
        if step % display_step == 0 or step == 1:
            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= {:.4f}".format(loss))

    # display a real life output
    # test_input, test_label = getBatch(1)
    # print("test input is: ", test_input)
    # print("test label is: ", test_label)
    # print("actual predicted result is: ", sess.run(output_layer, feed_dict={X: test_input}))

