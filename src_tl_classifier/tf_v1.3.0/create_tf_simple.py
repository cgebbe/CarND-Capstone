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
                for idx in range(len(batch_X)):
                    batch_X[idx, ...] = self.augment_img(batch_X[idx, ...])
            batch_X = batch_X / 255.  # rescale to [0,1]
            return batch_X, batch_y

    def shuffle(self):
        indexes = np.arange(len(self.X))
        np.random.shuffle(indexes)
        self.X = self.X[indexes, ...]
        self.y = self.y[indexes, ...]

    def augment_img(self, img):
        # flip horizontally, i.e. across width axis
        if np.random.uniform(0, 1) > 0.5:
            img = img[:, ::-1, :]

        # slight pixel shift
        pixels_height = np.random.randint(-8, 9)
        pixels_width = np.random.randint(-8, 9)
        img = np.roll(img, pixels_height, axis=0)
        img = np.roll(img, pixels_width, axis=1)

        # add gaussian noise
        img = img.astype(np.int)
        img += np.random.randint(-10, 11, img.shape, dtype=np.int)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Potentially more: zoom, color shift, ... ?
        if False:
            img_pil = PIL.Image.fromarray(img)
            img_pil.show()

        # return
        return img


def load_datasets_mnist():
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # define training subset
    train = Dataset()
    train.X = mnist.train.images
    train.X = train.X.reshape(-1, 28, 28, 1)
    train.y = mnist.train.labels

    # define validation/test subset
    valid = Dataset()
    valid.X = mnist.test.images[:256]
    valid.X = valid.X.reshape(-1, 28, 28, 1)
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
            num_classes = len(labels)
            X = np.empty((num_files, height, width, 3), dtype=np.uint8)
            y = np.zeros((num_files, num_classes), dtype=np.uint8)

            # store in numpy array
            cnt_img = 0
            for label in labels:
                files = os.listdir(os.path.join(root, label))
                files_png = [f for f in files if f.endswith(".png")]
                for file in files_png:
                    print("Loading image {}/{}".format(cnt_img, num_files))
                    path_png = os.path.join(root, label, file)
                    img = PIL.Image.open(path_png)
                    img = img.resize((width, height), resample=PIL.Image.BILINEAR)
                    X[cnt_img, ...] = np.asarray(img).astype(np.uint8)
                    y[cnt_img, int(label)] = 1
                    cnt_img += 1

            # save as numpy array and return
            np.save(path_X, X)
            np.save(path_y, y)
            return X, y

    # params
    path_train = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/export/splits/train"
    path_valid = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/export/splits/valid"

    # define training and validation subset
    train = Dataset(flag_augment=True)
    valid = Dataset(flag_augment=False)
    train.X, train.y = load_folder_as_numpy(path_train)
    valid.X, valid.y = load_folder_as_numpy(path_valid)

    return train, valid


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


# Create model
def apply_graph(x, prob_keep_dropout, num_classes):
    def block(x, num_filters, name, kernel_size=(5, 5), prob_keep_dropout=1.0):
        x = tf.layers.conv2d(x, num_filters, kernel_size, padding='same',
                             activation=tf.nn.relu, name=name + '_conv')
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name=name + '_pool')
        x = tf.nn.dropout(x, prob_keep_dropout, name=name + '_dropout')
        return x

    # blocks of convolution + maxpool + dropout (potentially)
    x = block(x, 8, 'block1', kernel_size=(3, 3), prob_keep_dropout=1.0)
    x = block(x, 12, 'block2', kernel_size=(3, 3), prob_keep_dropout=prob_keep_dropout)
    x = block(x, 18, 'block3', kernel_size=(3, 3), prob_keep_dropout=1.0)
    x = block(x, 24, 'block4', kernel_size=(3, 3), prob_keep_dropout=prob_keep_dropout)

    # Another convolution and a maxpool
    x = tf.layers.conv2d(x, 36, (3, 3), padding='same',
                         activation=tf.nn.relu, name='block5_conv')
    shape = x._shape_as_list()
    x = tf.layers.max_pooling2d(x, (shape[1], shape[2]), (1, 1),
                                padding='valid', name='block5_pool')

    # Flatten
    shape = x._shape_as_list()
    num_elements = shape[1] * shape[2] * shape[3]
    x = tf.reshape(x, [-1, num_elements])

    # Dropout and fully connected
    x = tf.nn.dropout(x, prob_keep_dropout, name='fc1_dropout')
    x = tf.layers.dense(x, num_classes, activation=None, name='fc2')

    return x


def calc_loss(X, Y,
              prob_keep,
              num_classes
              ):
    # Calculate accuracy (simply for evaluation purposes...)
    logits = apply_graph(X, prob_keep, num_classes)
    op_pred = tf.nn.softmax(logits, name='softmax')
    op_pred_correct = tf.equal(tf.argmax(op_pred, 1), tf.argmax(Y, 1))
    op_acc = tf.reduce_mean(tf.cast(op_pred_correct, tf.float32))

    # Calcualte loss
    op_cross = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=Y,
                                                       name='crossentropy')
    op_loss = tf.reduce_mean(op_cross, name='loss')

    # return operation nodes
    return op_pred, op_acc, op_cross, op_loss


if __name__ == '__main__':
    # load datasets and set
    dataset_name = 'own'  # ''mnist' #
    if dataset_name == 'own':
        width = 320
        height = 240
        set_train, set_valid = load_datasets_own(width, height)
        set_train.shuffle()
        input_shape = [None, height, width, 3]  # MNIST data input (img shape: 28*28)
        num_classes = 2  # for MNIST total classes (0-9 digits)
    elif dataset_name == 'mnist':
        set_train, set_valid = load_datasets_mnist()
        set_train.shuffle()
        input_shape = [None, 28, 28, 1]  # MNIST data input (img shape: 28*28)
        num_classes = 10  # for MNIST total classes (0-9 digits)
    else:
        raise NotImplementedError

    # Training parameters
    num_steps = 1000
    batch_size = 64
    steps_per_epoch = len(set_train.y) // batch_size

    # DEFINE GRAPH
    X = tf.placeholder(tf.float32, input_shape, name='input')
    Y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
    prob_keep = tf.placeholder(tf.float32, name='prob_keep')  # dropout (keep probability)
    op_pred, op_acc, op_cross, op_loss = calc_loss(X, Y, prob_keep, num_classes)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    op_train = optimizer.minimize(op_loss)

    # RUN GRAPH
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Train
        for step in range(num_steps):
            batch_x, batch_y = set_train.get_next_batch(batch_size)

            # Run optimization op (backprop) at each step
            sess.run(op_train, feed_dict={X: batch_x,
                                          Y: batch_y,
                                          prob_keep: 0.8})

            # Print loss and accuracy after N steps
            if step % steps_per_epoch == 0:
                pred, acc, cross, loss = sess.run([op_pred, op_acc, op_cross, op_loss],
                                                  feed_dict={X: batch_x,
                                                             Y: batch_y,
                                                             prob_keep: 1.0})
                pred_val, acc_val, cross_val, loss_val = sess.run(
                    [op_pred, op_acc, op_cross, op_loss],
                    feed_dict={X: set_valid.X / 255.,
                               Y: set_valid.y,
                               prob_keep: 1.0}
                )
                print("Step " + str(step+1)
                      + ", train_loss={:.4f}".format(loss)
                      + ", train_acc={:.3f}".format(acc)
                      + ", val_loss={:.4f}".format(loss_val)
                      + ", val_acc={:.3f}".format(acc_val)
                      )

                # show falsely predicted images
                if False:
                    mask_correct = np.equal(np.argmax(pred_val, 1), np.argmax(set_valid.y, 1))
                    indexes_false = np.argwhere(mask_correct == False).flatten()
                    for idx in indexes_false:
                        if set_valid.y[idx, 1] == 1:
                            img_np = set_valid.X[idx, ...]
                            img_pil = PIL.Image.fromarray(img_np)
                            img_pil.show()

                # Save graph as protobuf file
                if False:
                    path_folder_out = os.path.dirname(__file__)
                    frozen_graph = freeze_session(sess, output_names=['softmax'])
                    tf.train.write_graph(frozen_graph, path_folder_out, "model.pb", as_text=False)
                    tf.train.write_graph(frozen_graph, path_folder_out, "model.pb.txt", as_text=True)

        print("=== Finished!")
