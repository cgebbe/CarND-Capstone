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
import numpy as np
import PIL.Image


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


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1, name=None):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W,
                     strides=[1, strides, strides, 1],
                     padding='SAME',
                     name=name + '_conv',
                     )
    x = tf.nn.bias_add(x, b,
                       name=name + '_bias_add', )
    x = tf.nn.relu(x, name=name + '_relu', )
    return x


def maxpool2d(x, k=2, name=None):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x,
                          ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',
                          name=name)


# Create model
def get_architecture(x, prob_keep_dropout, num_classes):
    # weights and bias
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

    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

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


def get_graph(X,
              prob_keep,
              num_classes,
              learning_rate=0.001
              ):
    # Based on model, define
    logits = get_architecture(X, prob_keep, num_classes)
    prediction = tf.nn.softmax(logits, name='softmax')
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                labels=Y,
                                                name='crossentropy')
        , name='loss'
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # return operation nodes
    return accuracy_op, loss_op, train_op


class Dataset:
    def __init__(self, X=None, y=None, flag_augment=False):
        self.X = X
        self.y = y
        self.idx_batch = 0
        self.flag_augment = flag_augment

    def get_next_batch(self, batch_size):
        num_batches = len(self.X) // batch_size
        while True:
            if self.idx_batch == num_batches:
                self.idx_batch = 0
                self.shuffle()
            batch_X = self.X[self.idx_batch * batch_size:(self.idx_batch + 1) * batch_size]
            batch_y = self.y[self.idx_batch * batch_size:(self.idx_batch + 1) * batch_size]
            self.idx_batch += 1
            if self.flag_augment:
                for idx in len(batch_X):
                    batch_X[idx,...] = self.augment_img(batch_X[idx, ...])
            return batch_X, batch_y

    def shuffle(self):
        indexes = np.arange(len(self.X))
        np.random.shuffle(indexes)
        self.X = self.X[indexes, ...]
        self.y = self.y[indexes, ...]

    def augment_img(self, img):
        # flip horizontally, i.e. across width axis
        if np.random.uniform(0,1) > 0.5:
            img = img[:, ::-1, :]

        # slight pixel shift
        pixels_height = np.random.randint(-5, 6)
        pixels_width = np.random.randint(-5, 6)
        img = np.roll(img, pixels_height, axis=0)
        img = np.roll(img, pixels_width, axis=1)

        # add gaussian noise
        img += np.random.uniform(-10,10,img.shape)
        img = np.clip(img,0,255).astype(np.uint8)

        # Potentially more: zoom, color shift, ... ?

        # return
        return img




def load_datasets_mnist():
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # define training subset
    train = Dataset()
    train.X = mnist.train.images
    train.X = train.X.reshape(-1,28,28,1)
    train.y = mnist.train.labels

    # define validation/test subset
    valid = Dataset()
    valid.X = mnist.test.images[:256]
    valid.X = valid.X.reshape(-1,28,28,1)
    valid.y = mnist.test.labels[:256]
    return train, valid


def load_datasets_own(width, height, recalc=False):
    def load_folder_as_numpy(root, width=width, height=height, recalc=recalc):
        path_X = os.path.join(root, "X.npy")
        path_y = os.path.join(root, "y.npy")

        if (recalc == False and os.path.exists(path_X) and os.path.exists(path_y)):
            print("Returning precalculated numpy arrays...")
            X = np.load(path_X)
            y = np.load(path_y)
            return X, y
        else:
            # find out number of files
            labels = os.listdir(root)
            num_files = 0
            for label in labels:
                files = os.listdir(os.path.join(root, label))
                files_png = [f for f in files if f.endswith(".png")]
                num_files += len(files_png)

            # allocate numpy arrays
            X = np.empty((num_files, height, width, 3), dtype=np.uint8)
            y = np.empty((num_files, 1), dtype=np.uint8)

            # store in numpy array
            cnt_img = 0
            for label in labels:
                files = os.listdir(os.path.join(root, label))
                files_png = [f for f in files if f.endswith(".png")]
                for file in files_png:
                    print("Loading image {}/{}".format(cnt_img, num_files))
                    path_png = os.path.join(root, label, file)
                    img = PIL.Image.open(path_png)
                    img = img.resize((width, height),resample=PIL.Image.BILINEAR)
                    X[cnt_img, ...] = np.asarray(img).astype(np.uint8)
                    y[cnt_img, ...] = np.asarray(label).astype(np.uint8)
                    cnt_img += 1

            # save as numpy array and return
            np.save(path_X, X)
            np.save(path_y, y)
            return X, y

    # params
    path_train = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/export/splits/train"
    path_valid = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/export/splits/valid"

    # define training and validation subset
    train = Dataset()
    valid = Dataset()
    train.X, train.y = load_folder_as_numpy(path_train)
    valid.X, valid.y = load_folder_as_numpy(path_valid)

    return train, valid


if __name__ == '__main__':
    # load datasets and set
    dataset_name = 'own' #''mnist' #
    if dataset_name == 'own':
        width = 320
        height = 240
        set_train, set_valid = load_datasets_own(width, height)
        set_train.shuffle()
        input_shape = [None,width, height, 3]# MNIST data input (img shape: 28*28)
        num_classes = 2  # for MNIST total classes (0-9 digits)
    elif dataset_name == 'mnist':
        set_train, set_valid = load_datasets_mnist()
        set_train.shuffle()
        input_shape = [None, 28,28,1]  # MNIST data input (img shape: 28*28)
        num_classes = 10  # for MNIST total classes (0-9 digits)
    else:
        raise NotImplementedError

    # Network parameters

    # Training parameters
    num_steps = 200
    batch_size = 128
    display_step = 10

    # DEFINE GRAPH
    X = tf.placeholder(tf.float32, input_shape, name='input')
    Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
    prob_keep = tf.placeholder(tf.float32, name='prob_keep')  # dropout (keep probability)
    accuracy_op, loss_op, train_op = get_graph(X, prob_keep, num_classes)

    # RUN GRAPH
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Train
        for step in range(1, num_steps + 1):
            batch_x, batch_y = set_train.get_next_batch(batch_size)

            # Run optimization op (backprop) at each step
            sess.run(train_op, feed_dict={X: batch_x,
                                          Y: batch_y,
                                          prob_keep: 0.5})

            # Print loss and accuracy
            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy_op], feed_dict={X: batch_x,
                                                                        Y: batch_y,
                                                                        prob_keep: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Evaluate on test dataset (256 MNIST test images)
        acc = sess.run(accuracy_op, feed_dict={X: set_valid.X,
                                               Y: set_valid.y,
                                               prob_keep: 1.0})
        print("Testing Accuracy:", acc)

        # Save graph as protobuf file
        path_folder_out = os.path.dirname(__file__)
        frozen_graph = freeze_session(sess, output_names=['softmax'])
        tf.train.write_graph(frozen_graph, path_folder_out, "model.pb", as_text=False)
        tf.train.write_graph(frozen_graph, path_folder_out, "model.pb.txt", as_text=True)
        print("=== Finished!")
