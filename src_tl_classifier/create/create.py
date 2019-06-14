"""
from https://www.tensorflow.org/tutorials/images/hub_with_keras
"""

from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
import numpy as np
import PIL.Image
import os


def create_new_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(6, (3, 3), padding='same', activation='relu', input_shape=(112, 112, 3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(6, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(6, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(6, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()
    return model


def get_pretrained_model(feature_extractor_url):
    # get classifier from tf hub without top layer
    NUM_CLASSES = 2

    def feature_extractor(x):
        feature_extractor_module = hub.Module(feature_extractor_url)
        return feature_extractor_module(x)

    IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
    features_extractor_layer = tf.keras.layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE + [3])
    features_extractor_layer.trainable = False

    # add classification layer to model
    model = tf.keras.Sequential([
        features_extractor_layer,
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.summary()

    # variables have to be manually initialized this time (?)
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)

    return model


def get_data(path_folder, input_size):
    # get dataset
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    image_data = image_generator.flow_from_directory(path_folder,
                                                     target_size=input_size)
    for image_batch, label_batch in image_data:
        print("Image batch shape: ", image_batch.shape)
        print("Labe batch shape: ", label_batch.shape)
        break
    return image_data


def train(model, image_data, path_folder_out):
    # save checkpoint
    path_out = os.path.join(path_folder_out, "model.ckpt")
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(path_out,
                                                       save_weights_only=False,
                                                       verbose=1)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    steps_per_epoch = image_data.samples // image_data.batch_size
    hist = model.fit((item for item in image_data),
                     epochs=1,
                     steps_per_epoch=steps_per_epoch,
                     callbacks=[cb_checkpoint],
                     )
    return hist


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


if __name__ == '__main__':
    # params
    # url_model = r"https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"
    url_model = r"https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/3"
    path_data = r"/mnt/sda1/projects/git/udacity_car_nanodegree/term2_new_syllabus/VM_capstone/shared/export/splits/train"
    path_folder_out = os.path.dirname(__file__)

    # train model
    # model = get_pretrained_model(url_model)
    model = create_new_model()
    data = get_data(path_data, model.input_shape[1:3])
    train(model, data, path_folder_out)

    # save graph (as metagraph) and weights as checkpoint
    graph = K.get_session().graph
    meta_graph_def = tf.train.export_meta_graph(filename=os.path.join(path_folder_out,'model.meta'),
                                                graph=graph)

    # save model as protobuf file
    K.set_learning_phase(0)  # prunes graph such that only inference part is kept
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs],
                                  )
    tf.train.write_graph(frozen_graph, path_folder_out, "model.pb", as_text=False)
    tf.train.write_graph(frozen_graph, path_folder_out, "model.pb.txt", as_text=True)

    print("=== Finished")
