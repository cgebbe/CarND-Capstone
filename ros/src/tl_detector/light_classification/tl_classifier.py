import tensorflow as tf
from google.protobuf import text_format
import numpy as np
import os

class TLClassifier(object):
    def __init__(self, filename_pb="model.pb", flag_testrun=True):
        # load model
        folder_curr = os.path.dirname(__file__)
        path_pb = os.path.join(folder_curr, filename_pb)
        self.graph = self.load_graph_from_pb(path_pb, is_pb_binary=True)
        self.tensor_in_name = 'input:0'
        self.tensor_out_name = 'softmax:0'

        # perform a test run with random input
        if flag_testrun:
            test_out = self.predict()

    def load_graph_from_pb(self, path_pb, is_pb_binary):
        graph_def = tf.GraphDef()
        with open(path_pb, "rb") as f:
            if is_pb_binary:
                graph_def.ParseFromString(f.read())
            else:
                text_format.Merge(f.read(), graph_def)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    def predict(self, input_tensor = None):
        op_in = self.graph.get_tensor_by_name(self.tensor_in_name)
        op_out = self.graph.get_tensor_by_name(self.tensor_out_name)
        op_dropout_prob_keep = self.graph.get_tensor_by_name('prob_keep:0')

        # if input tensor is not specified, create random input (used for test run)
        if input_tensor is None:
            input_shape = op_in._shape_as_list()
            input_shape[0] = 1  # meaning batch_size=1
            input_tensor = np.random.uniform(0.0, 1.0, input_shape)

        # run graph
        with tf.Session(graph=self.graph) as sess:
            out = sess.run(op_out, feed_dict={op_in: input_tensor,
                                              op_dropout_prob_keep: 1.0})
            return out