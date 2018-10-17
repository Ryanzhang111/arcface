from __future__ import print_function, division

import os
import argparse
from multiprocessing import Pool
from random import shuffle

import tensorflow as tf


Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example
BytesList = tf.train.BytesList
Int64List = tf.train.Int64List


def to_record(args):

    data, fname = args

    if not data:
        return

    writer = tf.python_io.TFRecordWriter(fname)

    for d in data:
        i_path, label = d.split()

        with open(i_path, 'rb') as f:
            img_str = f.read()
        
        example = Example(features=Features(feature={
            'image_raw': Feature(bytes_list=BytesList(value=[img_str])),
            'label': Feature(int64_list=Int64List(value=[int(label)]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main(args):
    imglist = args.imglist
    dst = args.dst
    n = args.n
    n_threads = args.n_threads

    with open(imglist) as f:
        datalist = f.readlines()

    step_size = len(datalist) // n + 1

    shuffle(datalist)

    pool = Pool(n_threads)
    args = [(datalist[i*step_size:(i+1)*step_size],
             '%s_%03d.tfrecord' % (dst, i))
             for i in range(n)]
    pool.map(to_record, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make tfrecord for dataset')
    parser.add_argument('imglist', help='image list file, each row: img_path<space>class_id')
    parser.add_argument('dst', help='dst file name')
    parser.add_argument('n', type=int, help='n split file')
    parser.add_argument('-t', '--n_threads', type=int, default=1, help='n threads: 1')
    args = parser.parse_args()
    main(args)
