import tensorflow as tf
from tensorflow.contrib import slim
from custom_ops import pre_basic_block, conv2d_same

data_format='NCHW'


def build_arg_scope(weight_decay=0.0001):
    conv2d_scope = slim.arg_scope(
        [slim.conv2d],
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm
    )
    bn_scope = slim.arg_scope(
        [slim.batch_norm],
        fused=True,
        scale=True
    )
    df_scope =  slim.arg_scope(
        [slim.conv2d, slim.batch_norm, slim.avg_pool2d, conv2d_same],
        data_format='NCHW'
    )
    wd_scope = slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer()
    )

    with conv2d_scope, bn_scope, df_scope, wd_scope as arg_scope:
        return arg_scope


def build_net(images, n_class=None, is_training=True, reuse=False, scope='resnet_152_half'):

    with tf.variable_scope(scope, reuse=reuse):

        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

            net_spec = [
                [[i, i] for i in [32, 64, 128, 256]],
                [2, 2, 2, 2]
            ]

            net = conv2d_same(images, 32, 3, stride=1, scope='conv1')

            for i, spec in enumerate(zip(*net_spec)):

                stride = 2 if i != 0 else 1

                block_spec, n_block = spec 
                for j in range(n_block): 
                    net = pre_basic_block(
                        net, block_spec,
                        stride=stride,
                        scope='res_%d_%d' % (i+1, j+1))
                    stride = 1

            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            net = slim.flatten(net)
            # net = slim.dropout(net, 0.4)
            net = slim.fully_connected(
                net, 128,
                # weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                normalizer_fn=slim.batch_norm,
                activation_fn=None, scope='fc5') 

            if isinstance(n_class, int):
                net = slim.flatten(net)
                net = slim.fully_connected(net, n_class, activation_fn=None, scope='logits')

            return net

