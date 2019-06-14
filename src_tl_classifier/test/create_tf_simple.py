""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    from https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10

# Network Parameters
num_input = 28*28  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input], name='input')
Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
prob_keep = tf.placeholder(tf.float32, name='prob_keep')  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1, name=None):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W,
                     strides=[1, strides, strides, 1],
                     padding='SAME',
                     name=name+'_conv',
                     )
    x = tf.nn.bias_add(x, b,
                     name=name+'_bias_add',)
    x = tf.nn.relu(x, name=name+'_relu',)
    return x


def maxpool2d(x, k=2, name=None):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x,
                          ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',
                          name=name)


# Create model
def conv_net(x, weights, biases, prob_keep_dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['w_conv1'], biases['b_conv1'], name='conv1')
    conv1 = maxpool2d(conv1, k=2, name='conv1_pool')

    # Convolution Layer
    conv2 = conv2d(conv1, weights['w_conv2'], biases['b_conv2'], name='conv2')
    conv2 = maxpool2d(conv2, k=2, name='conv2_pool')

    # Flatten
    fc1 = tf.reshape(conv2, [-1, weights['w_fc1'].get_shape().as_list()[0]])

    # fully connected layer 1 including dropout
    fc1 = tf.add(tf.matmul(fc1, weights['w_fc1']), biases['b_fc1'], name='fc1')
    fc1 = tf.nn.relu(fc1, name='fc1_relu')
    fc1 = tf.nn.dropout(fc1, prob_keep_dropout, name='fc1_dropout')

    # fully connected layer 2
    fc2 = tf.add(tf.matmul(fc1, weights['w_fc2']), biases['b_fc2'], name='fc2')
    return fc2


# Store layers weight & bias
weights = {
    'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'w_fc1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'w_fc2': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'b_conv1': tf.Variable(tf.random_normal([32])),
    'b_conv2': tf.Variable(tf.random_normal([64])),
    'b_fc1': tf.Variable(tf.random_normal([1024])),
    'b_fc2': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, prob_keep)
prediction = tf.nn.softmax(logits, name='softmax')

# Define loss and optimizer
loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='crossentropy')
    , name='loss'
)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
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
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, prob_keep: 0.5})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 prob_keep: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    acc = sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        prob_keep: 1.0})
    print("Testing Accuracy:", acc)

    # Save graph as protobuf file
    path_folder_out = os.path.dirname(__file__)
    frozen_graph = freeze_session(sess, output_names=['softmax'])
    tf.train.write_graph(frozen_graph, path_folder_out, "model.pb", as_text=False)
    tf.train.write_graph(frozen_graph, path_folder_out, "model.pb.txt", as_text=True)
    print("=== Finished!")
