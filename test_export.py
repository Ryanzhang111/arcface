import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import time
import importlib
import argparse

import numpy as np
import tensorflow as tf


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

    batch_size = args.batch_size * args.n_gpu

    with tf.device('/cpu:0'):
        dataset = (tf.data.Dataset
            .from_tensor_slices(imglist)
            .map(preprocess(args.height, args.width), args.n_threads)
            .batch(batch_size)
            .prefetch(5))

        iterator = dataset.make_one_shot_iterator()
        filenames, images = iterator.get_next()

        # pad for the last batch
        n_pad = batch_size - tf.shape(images)[0]
        images = tf.cond(tf.equal(n_pad, 0), lambda : images, lambda : tf.pad(images, [(0, n_pad), (0, 0), (0, 0), (0, 0)]))

        # split for each gpu
        image_batches = tf.split(images, args.n_gpu, num=batch_size)


    features_list = []
    for i in range(args.n_gpu):

        with tf.name_scope('GPU_%d' % i), tf.device('/gpu:%d' % i):

            images = image_batches[i]
            images.set_shape([None, args.height, args.width, 3])

            if args.data_format == 'NCHW':
                images = tf.transpose(images, [0, 3, 1, 2])

            with open(args.model, 'rb') as f:
                graph_def = tf.GraphDef.FromString(f.read())

            features, = tf.import_graph_def(
                graph_def, {'input:0': images},
                ['output:0'], name='')

            features_list.append(features)


    with tf.device('/cpu:0'):
        features = tf.concat(features_list, axis=0)
        features = tf.cond(tf.equal(n_pad, 0), lambda : features, lambda : features[0:batch_size - n_pad])


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(init_op)

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
    parser.add_argument('--data_format', default='NCHW', help='model name')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='batch size')
    parser.add_argument('--width', '-w', type=int, default=96, help='image width')
    parser.add_argument('--height', '-i', type=int, default=112, help='image height')
    parser.add_argument('--n_threads', '-n', type=int, default=8, help='n threads')
    parser.add_argument('--sep', '-s', default=',', help='sep')
    parser.add_argument('--n_gpu', type=int, default=1, help='n gpus')

    args = parser.parse_args()
    main(args)
