import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import time
import importlib
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


def preprocess(height, width):

    def _preprocess(filename):

        raw_str = tf.read_file(filename)
        image = tf.image.decode_image(raw_str, 3)
        image.set_shape([None, None, 3])
        image = tf.image.resize_images(image, [height, width])
        image = (tf.to_float(image) - 127.5) / 128.0

        return filename, image

    return _preprocess


def main(args):

    with open(args.imglist) as f:
        imglist = [i.strip() for i in f]

    with tf.device('/cpu:0'):
        dataset = (tf.data.Dataset
            .from_tensor_slices(imglist)
            .map(preprocess(args.height, args.width), args.n_threads)
            .batch(args.batch_size)
            .prefetch(5))

        iterator = dataset.make_one_shot_iterator()
        filenames, images = iterator.get_next()


    model = importlib.import_module('models.{}'.format(args.model))
    with slim.arg_scope(model.build_arg_scope()):

        if getattr(model, 'data_format', None) == 'NCHW':
            images = tf.transpose(images, [0, 3, 1, 2])

        features = model.build_net(images, is_training=False)


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, args.cpt_name)

    if not tf.gfile.Exists(args.outdir):
        tf.gfile.MkDir(args.outdir)

    feats_file = open(args.outdir + '/reps.csv', 'wb')
    labels_file = open(args.outdir + '/labels.csv', 'wb')

    count = 0
    try:
        while True:

            start = time.time()
            fns, feats = sess.run([filenames, features])
            duration = time.time() - start

            np.savetxt(feats_file, feats, fmt='%.6f', delimiter=args.sep)
            labels_file.writelines([b'%s\n' % i for i in fns])

            count += len(feats)
            print('Finish: %d images, Speed: %.4fi/s.' % (count, len(feats)/duration))

    except tf.errors.OutOfRangeError:
        print('-' * 48)
        print('Done')

    labels_file.close()
    feats_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract features')
    parser.add_argument('imglist', help='image list')
    parser.add_argument('outdir', help='output dir')
    parser.add_argument('model', help='model name')
    parser.add_argument('cpt_name', help='checkpoint name')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='batch size')
    parser.add_argument('--width', '-w', type=int, default=96, help='image width')
    parser.add_argument('--height', '-i', type=int, default=112, help='image height')
    parser.add_argument('--n_threads', '-n', type=int, default=4, help='n threads')
    parser.add_argument('--sep', '-s', default=',', help='sep')
    parser.add_argument('--n_gpu', type=int, default=1, help='n gpus')

    args = parser.parse_args()
    main(args)
