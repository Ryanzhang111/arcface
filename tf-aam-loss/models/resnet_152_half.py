import tensorflow as tf
from tensorflow.contrib import slim
from custom_ops import bottleneck_block, conv2d_same

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

        with tf.arg_scope([slim.batch_norm], is_training=is_training):

            net_spec = [
                [[i, i, i*4] for i in [32, 64, 128, 256]],
                [3, 8, 36, 3]
            ]

            net = conv2d_same(images, 32, 5, stride=2, scope='conv1')

            for i, spec in enumerate(zip(*net_spec)):

                stride = 2 if i != 0 else 1

                block_spec, n_block = spec 
                for j in range(n_block): 
                    net = bottleneck_block(
                        net, block_spec,
                        stride=stride,
                        scope='res_%d_%d' % (i+1, j+1)
                    )
                    stride = 1

            net = slim.conv2d(net, 256, 1, stride=1, scope='last_conv')
            net = slim.flatten(net)
            net = slim.fully_connected(
                net, 512,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                activation_fn=None, scope='fc5'
            ) 

            if isinstance(n_class, int):
                net = slim.fully_connected(net, n_class, activation_fn=None, scope='logits')

            # add summarys
            slim.summarize_collection(tf.GraphKeys.MODEL_VARIABLES)
            
            return net

