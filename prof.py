import os
import argparse
import importlib

import tensorflow as tf
from tensorflow.contrib import slim


def main(args):

    model = importlib.import_module('models.{}'.format(args.model))
    shape = [1, 3, 112, 96] if getattr(model, 'data_format', None) == 'NCHW' else [1, 112, 96, 3]
    # shape = [1, 3, 224, 224] if getattr(model, 'data_format', None) == 'NCHW' else [1, 224, 224, 3]
    images = tf.placeholder(tf.float32, shape=shape)

    with slim.arg_scope(model.build_arg_scope(weight_decay=0.)):
        feats = model.build_net(images, is_training=False)

    graph = tf.get_default_graph()

    if args.type == "compution":
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
    else:
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    param_stats = tf.profiler.profile(graph, options=opts)

    if args.write_graph:
        if not os.path.exists(args.write_graph):
            os.mkdir(args.write_graph)
        writer = tf.summary.FileWriter(args.write_graph, graph=tf.get_default_graph())
        writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="profile model")
    parser.add_argument("model", help="the model to profile")
    parser.add_argument("--type", default="compution",
                        choices=["compution", "parameter"],
                        help="type to profile")
    parser.add_argument("--write_graph", default="",
                        help="dir to write the graph")
    args = parser.parse_args()
    main(args)
