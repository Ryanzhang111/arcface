import os
# set tensorflow cpp log level
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import importlib
import time

import tensorflow as tf
from tensorflow.contrib import slim

from data_utils import preprocess_for_train, preprocess_for_eval
import custom_ops


FLAGS = tf.app.flags.FLAGS

# flags for the dataset
tf.flags.DEFINE_string('dataset_root', './data/webface-lcnn-40-new', 'dataset root')
tf.flags.DEFINE_integer('num_class', 10575, 'num of class')
tf.flags.DEFINE_string('image_size', '112,96', 'height,width')

# flags for log
tf.flags.DEFINE_string('logdir', 'log/resnet_18_half', 'log directory')
tf.flags.DEFINE_integer('log_every', 100, 'display and log frequency')
tf.flags.DEFINE_string('restore', '', 'snapshot path')

# flats for hyper parameters
tf.flags.DEFINE_integer('start_epoch', 0, 'start epoch')
tf.flags.DEFINE_integer('end_epoch', 30, 'number of epoch')

tf.flags.DEFINE_string('model', 'resnet_152_half', 'model name')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')

# flags for learning rate
tf.flags.DEFINE_string('boundaries', '10,20,26', 'learning rate boundaries')
tf.flags.DEFINE_string('values', '1e-1,1e-2,1e-3,1e-4', 'learning rate values')
tf.flags.DEFINE_float('lr_scale', 1., 'learning rate scale')
tf.flags.DEFINE_float('s', 32, 'aam scale')
tf.flags.DEFINE_float('m', 0.5, 'aam margin')
tf.flags.DEFINE_integer('n_epoch_softmax', 5, 'n epoch train with softmax')

# flags for finetune
tf.flags.DEFINE_string('finetune', '', 'finetune snapshot path')
tf.flags.DEFINE_bool('train_model', True, 'whether to train the model')
tf.flags.DEFINE_float('pre_lr_scale', 1., 'lr scale for layers bellow classify layers')
tf.flags.DEFINE_bool('load_fc', False, 'whether to train the model')

# flags for gpus
tf.flags.DEFINE_integer('n_gpu', 1, 'number of gpus')


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, v in grad_and_vars]
        grad = tf.add_n(grads)
        grad = grad * (1. / len(grads))

        v = grad_and_vars[0][1]
        average_grads.append((grad, v))

    return average_grads


class LRManager:

    def __init__(self, boundaries, values):
        self.boundaries = boundaries
        self.values = values
        self.i = 0

    def get(self, epoch):
        for b, v in zip(self.boundaries, self.values):
            if epoch < b:
                # N = 5000
                # lr_start = 0.05
                # if self.i < N:
                #     v_m = lr_start + (self.i / N) * (v - lr_start)
                #     v = v_m
                #     self.i += 1
                return v
        return self.values[-1]


class TimeMeter:

    def __init__(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.duration += time.perf_counter() - self.start_time
        self.counter += 1

    def get(self):
        return self.duration / self.counter

    def reset(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.


def main(_):

    with tf.device('/cpu:0'), tf.name_scope('input') as scope:

        image_size = list(map(int, FLAGS.image_size.split(',')))

        # build dataset for training
        train_dataset = (tf.data.Dataset
            .list_files(FLAGS.dataset_root + '/train_*.tfrecord')
            .shuffle(100, seed=0)
            .interleave(lambda f: tf.data.TFRecordDataset(f), 8, 4)
            .shuffle(10000)
            .map(preprocess_for_train(*image_size), 8)
            .batch(FLAGS.batch_size)
            .prefetch(FLAGS.n_gpu))

        # build dataset for val
        val_dataset = (tf.data
            .TFRecordDataset(tf.gfile.Glob(FLAGS.dataset_root + '/val_*.tfrecord'))
            .map(preprocess_for_eval(*image_size), 8)
            .batch(FLAGS.batch_size))

        # construct iterator
        data_iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes)

        train_data_init = data_iterator.make_initializer(train_dataset)
        val_data_init = data_iterator.make_initializer(val_dataset)

        # get images and labels for gpus
        datas = [data_iterator.get_next() for i in range(FLAGS.n_gpu)]
        tf.summary.image('images', datas[0][0])


    # define useful scalars
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    tf.summary.scalar('lr', learning_rate)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    m_value = tf.placeholder(tf.float32, shape=(), name='m')


    # define optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, 0.9, use_nesterov=True, name='optimizer')

    variable_reuse = False
    grads_all = []
    for i in reversed(range(FLAGS.n_gpu)):

        device_setter = tf.device(tf.train.replica_device_setter(
            worker_device='/gpu:%d' % i,
            ps_device='/gpu:0' if FLAGS.n_gpu == 1 else '/cpu:0',
            ps_tasks=1
        ))

        with tf.name_scope('GPU_%d' % i) as scope, device_setter:

            images, labels = datas[i]

            # build the net
            model = importlib.import_module('models.{}'.format(FLAGS.model))
            with slim.arg_scope(model.build_arg_scope(FLAGS.weight_decay)):

                if getattr(model, 'data_format', None) == 'NCHW':
                    images = tf.transpose(images, [0, 3, 1, 2])

                features = model.build_net(images, reuse=variable_reuse, is_training=is_training if FLAGS.train_model else False)

            with tf.name_scope('losses'):

                # compute loss
                loss, logits = custom_ops.additive_angular_margin(
                    features, labels, FLAGS.num_class,
                    regularizer=slim.l2_regularizer(FLAGS.weight_decay),
                    s=FLAGS.s, m=m_value,
                    reuse=variable_reuse,
                    scope='AAM')

                # compute l2 regularization
                l2_reg = tf.losses.get_regularization_loss()

            variable_reuse = True

            # compute grads for this gpu
            grads = optimizer.compute_gradients(loss + l2_reg)
            grads_all.append(grads)


            with tf.name_scope('metrics') as scope:

                mean_loss, mean_loss_update_op = tf.metrics.mean(
                    loss, name='mean_loss')

                prediction = tf.argmax(logits, axis=1)
                accuracy, accuracy_update_op = tf.metrics.accuracy(
                    labels, prediction, name='accuracy')

                reset_metrics = tf.variables_initializer(
                    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope))
                metrics_update_op = tf.group(mean_loss_update_op, accuracy_update_op)

                # collect metric summary alone, because it need to
                # summary after metrics update
                metric_summary = [
                    tf.summary.scalar('loss', mean_loss, collections=[]),
                    tf.summary.scalar('accuracy', accuracy, collections=[])]


    for w in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
        tf.summary.histogram(w.op.name, w)

    with tf.name_scope('update_ops'):
        average_grads = average_gradients(grads_all)
        # summary grads
        # for g, v in average_grads:
        #     tf.summary.histogram(v.name + '/grad', g)

        train_ops = []
        if abs(FLAGS.pre_lr_scale - 1) > 0.0001 or not FLAGS.train_model:

            pre_grad = []
            cls_grad = []
            for g, v in average_grads:
                if 'AAM' in v.op.name:
                    cls_grad.append((g, v))
                else:
                    pre_grad.append((g, v))

            assert cls_grad

            average_grads = cls_grad

            if FLAGS.train_model:
                pre_optimizer = tf.train.MomentumOptimizer(
                    learning_rate*FLAGS.pre_lr_scale, 0.9,
                    use_nesterov=True, name='optimizer_for_train')

                train_ops.append(pre_optimizer.apply_gradients(pre_grad))
                print(len(pre_grad))

        print(len(average_grads))
        train_ops.append(
            optimizer.apply_gradients(average_grads, global_step=global_step))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(*train_ops, *update_ops)


    # build summary
    train_summary_str = tf.summary.merge_all()
    metric_summary_str = tf.summary.merge(metric_summary)

    # init op
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # prepare for the logdir
    if not tf.gfile.Exists(FLAGS.logdir):
    	tf.gfile.MakeDirs(FLAGS.logdir)
    	
    # saver
    saver = tf.train.Saver(max_to_keep=None, name='checkpoint_saver')

    # writer
    train_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'train'),
        tf.get_default_graph())
    val_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'val'),
        tf.get_default_graph())

    # session
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        intra_op_parallelism_threads=20, inter_op_parallelism_threads=32)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # do initialization
    sess.run(init_op)

    # restore
    if FLAGS.restore:
        saver.restore(sess, FLAGS.restore)
    elif FLAGS.finetune:
        print('finetune:', FLAGS.finetune)
        vars_to_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        if not FLAGS.load_fc:
            vars_to_restore = [v for v in vars_to_restore if 'AAM' not in v.op.name]
        saver_ft = tf.train.Saver(vars_to_restore, name='finetune_saver')
        saver_ft.restore(sess, FLAGS.finetune)

    lr_boundaries = list(map(int, FLAGS.boundaries.split(',')))
    lr_values = list(map(float, FLAGS.values.split(',')))
    lr_values = [r * FLAGS.lr_scale for r in lr_values]
    lr_manager = LRManager(lr_boundaries, lr_values)
    time_meter = TimeMeter()

    # start to train
    for e in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('-' * 40)
        print('Epoch: {:d}'.format(e))

        # training loop
        try:
            i = 0
            sess.run([train_data_init, reset_metrics])
            while True:

                m = 0 if e < FLAGS.n_epoch_softmax else FLAGS.m
                lr = lr_manager.get(e)
                fetch = [train_summary_str] if i % FLAGS.log_every == 0 else []

                time_meter.start()
                result = sess.run(
                    [train_op, metrics_update_op] + fetch,
                    {learning_rate: lr, is_training: True, m_value: m})
                time_meter.stop()

                if i % FLAGS.log_every == 0:
                    t_summary = result[-1]
                    t_metric_summary = sess.run(metric_summary_str)

                    t_loss, t_acc = sess.run([mean_loss, accuracy])
                    sess.run(reset_metrics)

                    spd = FLAGS.batch_size / time_meter.get()
                    time_meter.reset()

                    print('Iter: {:d}, LR: {:g}, Loss: {:.4f}, Acc: {:.2%}, Spd: {:.2f} i/s'.format(
                            i, lr, t_loss, t_acc, spd))

                    train_writer.add_summary(
                        t_summary, global_step=sess.run(global_step))
                    train_writer.add_summary(
                        t_metric_summary, global_step=sess.run(global_step))

                i += 1
        except tf.errors.OutOfRangeError:
            pass

        # save snapshot
        saver.save(
            sess,
            '{}/{}_{:d}'.format(FLAGS.logdir, FLAGS.model, e),
            global_step=tf.train.global_step(sess, global_step))
        
        # val loop
        try:
            sess.run([val_data_init, reset_metrics])
            while True:
                sess.run(
                    [metrics_update_op], {is_training: False, m_value: m})
        except tf.errors.OutOfRangeError:
            pass

        v_loss, v_acc = sess.run([mean_loss, accuracy])
        print('[VAL]Loss: {:.4f}, Acc: {:.2%}'
              .format(v_loss, v_acc))

        val_writer.add_summary(sess.run(metric_summary_str),
                               global_step=sess.run(global_step))

    print('-' * 40)
    print('Done!')

if __name__ == '__main__':
    tf.app.run(main)
