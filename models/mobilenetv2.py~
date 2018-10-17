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


def mobile_block(inputs, num_output, kernel_size, stride=1, t=6, scope=None):
    with tf.variable_scope(scope, default_name='mobile_block'):

        k = inputs.shape[-1].value
        outputs = slim.conv2d(
            inputs, t*k, 1, stride=1, data_format='NHWC', scope='expand_conv')

        if k != num_output:
            shortcut = slim.conv2d(
                inputs, num_output, 1, stride=stride,
                activation_fn=None, scope='shortcut')
        else:
            shortcut = inputs

        if stride > 1:
            pad_wh = math.floor(kernel_size / 2)
            outputs = tf.pad(outputs, [[0, 0], [pad_wh, pad_wh], [pad_wh, pad_wh], [0, 0]])
            outputs = slim.separable_conv2d(
                outputs, None, kernel_size, 1,
                stride=stride, padding='VALID', scope='depth_conv')
        else:
            outputs = slim.separable_conv2d(
                outputs, None, kernel_size, 1,
                stride=stride, padding='SAME', scope='depth_conv')

        outputs = slim.conv2d(
            outputs, num_output, 1, stride=1,
            data_format='NHWC', activation_fn=None,
            normalizer_params={'param_initializers': {'gamma': tf.constant_initializer(0.)}},
            scope='point_conv')

        return outputs + shortcut


def build_net(images, n_class=None, is_training=True, reuse=False, alpha=1, scope='mobile_id'):

    with tf.variable_scope(scope, reuse=reuse) as scope:

        with slim.arg_scope([slim.batch_norm], is_training=is_training):

            net = conv2d_same(images, int(32*alpha), 3, stride=2, scope='conv1')

            net = mobile_block(net, int(16*alpha), 3, stride=1, t=1, scope='mblock1')

            net = mobile_block(net, int(24*alpha), 3, stride=2, scope='mblock2_1')
            net = mobile_block(net, int(24*alpha), 3, stride=1, scope='mblock2_2')

            net = mobile_block(net, int(32*alpha), 3, stride=2, scope='mblock3_1')
            net = mobile_block(net, int(32*alpha), 3, stride=1, scope='mblock3_2')
            net = mobile_block(net, int(32*alpha), 3, stride=1, scope='mblock3_3')

            net = mobile_block(net, int(64*alpha), 3, stride=2, scope='mblock4_1')
            for i in range(3):
                net = mobile_block(net, int(64*alpha), 3, stride=1, scope='mblock4_%d' % (i+2))
            for i in range(3):
                net = mobile_block(net, int(96*alpha), 3, stride=1, scope='mblock5_%d' % (i+1))

            net = mobile_block(net, int(160*alpha), 3, stride=2, scope='mblock6_1')
            for i in range(2):
                net = mobile_block(net, int(160*alpha), 3, stride=1, scope='mblock6_%d' % (i+2))

            net = mobile_block(net, int(320*alpha), 3, stride=1, scope='mblock7_1')

            net = slim.conv2d(net, 1280, 1, stride=1, scope='final_point_conv')
            # net = slim.conv2d(net, 512, 1, stride=1, scope='final_point_conv')

            # net = slim.separable_conv2d(
            #     net, None, [4, 3], 1,
            #     stride=1, padding='VALID',
            #     scope='GDConv')
            # print(net.shape)

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

            # add summarys
            slim.summarize_collection(tf.GraphKeys.MODEL_VARIABLES)
            
            return net 
