import tensorflow as tf
from google.protobuf import text_format
import numpy as np

def load_graph_from_pb(path_b, is_pb_binary):
    graph_def = tf.GraphDef()
    with open(path_pb, "rb") as f:
        if is_pb_binary:
            graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), graph_def)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def run_graph(graph, tensor_in_name, tensor_out_name):
    tensor_in = graph.get_tensor_by_name(tensor_in_name)
    tensor_out = graph.get_tensor_by_name(tensor_out_name)
    tensor_dropout = graph.get_tensor_by_name('prob_keep:0')

    # create random input
    input_shape = tensor_in._shape_as_list()
    input_shape[0] = 1  # corresponds to batch size
    img = np.random.uniform(0.0, 1.0, input_shape)

    # run graph
    with tf.Session(graph=graph) as sess:
        out = sess.run(tensor_out, feed_dict={tensor_in: img,
                                              tensor_dropout: 1.0})
        return out

if __name__ == '__main__':
    # params
    #path_pb = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/src_tl_classifier/create/new_model.pb"
    #path_pb = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/src_tl_classifier/create/mobilenet_v1/model.pb"
    path_pb = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/src_tl_classifier/test/mnist/model.pb"
    #path_meta = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/src_tl_classifier/create/own_simple/model.meta"
    #path_ckpt = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/src_tl_classifier/create/own_simple/model.ckpt"
    is_pb = True
    is_pb_binary = True
    tensor_in_name = 'input:0' #'lambda_input:0'
    tensor_out_name = 'softmax:0' #'dense_1/Softmax:0' #'dense/Softmax:0'

    # load and run pb graph
    if is_pb:
        graph = load_graph_from_pb(path_pb, is_pb_binary)
        out = run_graph(graph, tensor_in_name, tensor_out_name)
        print(out)
    else:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path_meta)
            saver.restore(sess, path_ckpt)
            dummy = 0

    print("=== Finished")

