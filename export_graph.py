from argparse import ArgumentParser
import importlib
import tensorflow as tf
from tensorflow.contrib import slim


def main(args):

    orig_graph = tf.Graph()

    with orig_graph.as_default():

        model = importlib.import_module('models.{}'.format(args.model))
        with slim.arg_scope(model.build_arg_scope()):

            images = tf.placeholder(
                tf.float32, shape=[None, 3, 112, 96], name='input')
            images = tf.reverse(images, axis=[1])
            images = (images - 127.5) / 128.
            # images = tf.transpose(images, [0, 2, 3, 1])
            with slim.arg_scope(model.build_arg_scope()):
                features = model.build_net(images, is_training=False, scope='')
            features = tf.identity(features, name='output')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=orig_graph, config=config)

        tf.train.init_from_checkpoint(args.cpt_name, {'/': '/'})
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

    orig_graph_def = orig_graph.as_graph_def()
    dst_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, orig_graph_def, [features.op.name])

    with open(args.output, 'wb') as f:
        f.write(dst_graph_def.SerializeToString())

    if args.config:
        with open(args.config, 'wb') as f:
            f.write(config.SerializeToString())


if __name__ == '__main__':
    parser = ArgumentParser(description='export GraphDef')
    parser.add_argument('model', help='model name')
    parser.add_argument('cpt_name', help='checkpoint file name')
    parser.add_argument('output', help='output file name')
    parser.add_argument('--config', '-c', default='',
                        help='output config file name')
    args = parser.parse_args()
    main(args)
