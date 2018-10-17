import tensorflow as tf
from tensorflow.contrib import slim
from custom_ops import conv2d_same
import math

data_format='NHWC'

def build_arg_scope(weight_decay=0.0001):
    conv2d_scope = slim.arg_scope(
        [slim.separable_conv2d, slim.conv2d],
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm)
    fc_scope = slim.arg_scope(
        [slim.fully_connected],
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm)
    bn_scope = slim.arg_scope(
        [slim.batch_norm],
        fused=True,
        scale=True)
    df_scope =  slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, slim.batch_norm, conv2d_same],
        data_format=data_format)
    wd_scope = slim.arg_scope(
        [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer())

    with conv2d_scope, fc_scope, bn_scope, df_scope, wd_scope as arg_scope:
        return arg_scope


def shuffle_block(inputs, stride, scope=None):

    with tf.variable_scope(scope, default_name='shuffle_block'):
        if stride > 1:
            # when stride == 2
            left_inputs, right_inputs = inputs, inputs
        else:
            left_inputs, right_inputs = tf.split(inputs, 2, axis=3)
        
        # right branch
        right_outputs = slim.conv2d(
            right_inputs, right_inputs.shape[3], 1, stride=1,
            data_format='NHWC', 
            scope='point_conv1')

        kernel_size = 3
        if stride > 1:
            pad_wh = math.floor(kernel_size / 2)
            right_outputs = tf.pad(right_outputs, [[0, 0], [pad_wh, pad_wh], [pad_wh, pad_wh], [0, 0]])
            right_outputs = slim.separable_conv2d(
                right_outputs, None, kernel_size, 1,
                activation_fn=None,
                stride=stride, padding='VALID', scope='depth_conv')
        else:
            right_outputs = slim.separable_conv2d(
                right_outputs, None, kernel_size, 1,
                activation_fn=None,
                stride=stride, padding='SAME', scope='depth_conv')

        right_outputs = slim.conv2d(
            right_outputs, right_outputs.shape[3], 1, stride=1,
            data_format='NHWC', 
            scope='point_conv2')

        # left branch
        if stride > 1:
            pad_wh = math.floor(kernel_size / 2)
            left_outputs = tf.pad(left_inputs, [[0, 0], [pad_wh, pad_wh], [pad_wh, pad_wh], [0, 0]])
            left_outputs = slim.separable_conv2d(
                left_outputs, None, kernel_size, 1,
                activation_fn=None,
                stride=stride, padding='VALID', scope='depth_conv_left')

            left_outputs = slim.conv2d(
                left_outputs, right_outputs.shape[3], 1, stride=1,
                data_format='NHWC', 
                scope='point_conv_left')
        else:
            left_outputs = left_inputs

        # shuffle
        outputs = tf.stack([left_outputs, right_outputs], axis=4, name='output')
        output_shape = outputs.shape
        outputs = tf.reshape(outputs, [-1, output_shape[1], output_shape[2], 2 * output_shape[3].value])
        return outputs


def build_net(images, n_class=None, is_training=True, reuse=False, alpha=1, scope='mobile_id'):

    with tf.variable_scope(scope, reuse=reuse) as scope:

        with slim.arg_scope([slim.batch_norm], is_training=is_training):

            net = conv2d_same(images, 58, 3, stride=2, scope='conv1')
            # net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')

            net = shuffle_block(net, 2, 'stage2_block1')
            for i in range(3):
                net = shuffle_block(net, 1, 'stage2_block%d' % (i+2))

            net = shuffle_block(net, 2, 'stage3_block1')
            for i in range(7):
                net = shuffle_block(net, 1, 'stage3_block%d' % (i+2))

            net = shuffle_block(net, 2, 'stage4_block1')
            for i in range(3):
                net = shuffle_block(net, 1, 'stage4_block%d' % (i+2))

            net = conv2d_same(net, 1024, 1, stride=1, scope='conv5')

            # net = slim.avg_pool2d(net, 7, stride=1, padding='VALID', scope='global_pool')

            net = slim.flatten(net)
            net = slim.fully_connected(
                net, 128,
                # weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                normalizer_fn=slim.batch_norm,
                # normalizer_params={'param_initializers': {'gamma': tf.constant_initializer(0.1)}},
                activation_fn=None,
                scope='fc1') 

            if isinstance(n_class, int):
                net = slim.flatten(net)
                net = slim.fully_connected(net, n_class, activation_fn=None, scope='logits')

            return net 
