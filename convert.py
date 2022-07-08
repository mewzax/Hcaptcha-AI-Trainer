"""
This script converts a .h5 Keras model into a Tensorflow .pb file.
Attribution: This script was adapted from https://github.com/amir-abdi/keras_to_tensorflow
MIT License
Copyright (c) 2017 bitbionic
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import os.path as osp

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import load_model
from keras import backend as K


def convertGraph(modelPath, outdir, numoutputs, prefix, name):
    # NOTE: If using Python > 3.2, this could be replaced with os.makedirs( name, exist_ok=True )
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    K.set_learning_phase(0)

    net_model = load_model(modelPath)

    # Alias the outputs in the model - this sometimes makes them easier to access in TF
    pred = [None] * numoutputs
    pred_node_names = [None] * numoutputs
    for i in range(numoutputs):
        pred_node_names[i] = prefix + "_" + str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    print("Output nodes names are: ", pred_node_names)

    sess = K.get_session()

    # Write the graph in binary .pb file
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io

    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), pred_node_names
    )
    graph_io.write_graph(constant_graph, outdir, name, as_text=False)
    print("Saved the constant graph (ready for inference) at: ", osp.join(outdir, name))


if __name__ == "__main__":
    convertGraph(
        "output/keras/model_1.h5", "./output/tensorflow", 1, "k2tfout", "model_1.pb"
    )
