import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import slim

data_format='NCHW'

def build_arg_scope(weight_decay=0.0001):

    noact_scope =  slim.arg_scope(
        [layers.conv2d, layers.batch_norm, layers.fully_connected],
        activation_fn=None
    )
    bn_scope = slim.arg_scope(
        [slim.batch_norm],
        fused=True,
        scale=True
    )
    df_scope =  slim.arg_scope(
        [slim.conv2d, slim.batch_norm, slim.avg_pool2d],
        data_format='NCHW'
    )
    wd_scope = slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer()
    )

    with noact_scope, bn_scope, df_scope, wd_scope as arg_scope:
        return arg_scope


def build_net(images, n_class=None, is_training=True, reuse=False, scope='resnet-456-half'):

    net_def = [(6, 16, 72, 6), ([32, 32, 32*4], [64, 64, 64*4], [128, 128, 128*4], [256, 256, 256*4])]


    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

        with slim.arg_scope([layers.batch_norm],
                fused=True, scale=True, is_training=is_training):

            net = layers.conv2d(images, 32, 3, stride=1, scope='conv0')
                    
            block = pre_bottleneck_block
            for k, (n, nOutChannels) in enumerate(zip(*net_def)):
                stride = 2
                both_preact = True
                for i in range(0, n):
                    net = block(
                        net, nOutChannels, stride=stride,
                        both_preact=both_preact,
                        scope='res%d_%d' % (k, i)
                    )
                    stride = 1
                    both_preact = False

            net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='bn1')

            net = layers.flatten(net)
            net = layers.fully_connected(net, 128, scope='fc5')
            net = layers.batch_norm(
                net, activation_fn=prelu,
                param_initializers={'gamma': tf.initializers.constant(0.1)},
                scope='bn2')

            if isinstance(n_class, int):
                logits = layers.fully_connected(
                    net, n_class, biases_initializer=None, scope='fc6')

            return net


@slim.add_arg_scope
def prelu(in_, trainable=True, scope=None):
    with tf.variable_scope(scope, 'PRelu'):
        in_shape = in_.get_shape().as_list()
        alpha_shape = [1 for i in range(len(in_shape))]
        alpha_shape[1] = in_shape[1]
        alpha = tf.get_variable(
            'weights', initializer=tf.constant_initializer(0.25),
            shape=alpha_shape, trainable=trainable
        )
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, alpha)

        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        with jit_scope():
            return tf.where(in_ > 0., in_, alpha * in_)


def pre_bottleneck_block(in_, nOutChannels, both_preact=False, activation=tf.nn.relu, stride=1, scope=None):

    assert len(nOutChannels) == 3

    scope_conv = slim.arg_scope(
        [layers.conv2d], activation_fn=None,
        biases_initializer=None,
    )

    scope_bn = slim.arg_scope(
        [layers.batch_norm], scale=True,
        center=True,        
    )

    with scope_conv, scope_bn, tf.variable_scope(scope):

        nInChannels = in_.get_shape()[1].value

        pre_act = layers.batch_norm(in_, scope='bn0')
        pre_act = activation(pre_act)

        pre_shortcut = pre_act if both_preact else in_

        if nInChannels == nOutChannels[-1]:
            if stride == 1:
                shortcut = pre_shortcut
            else:
                shortcut = layers.max_pool2d(pre_shortcut, 1, stride=stride, scope='shortcut')
        else:
            shortcut = layers.conv2d(pre_shortcut, nOutChannels[2], 1, stride=stride, scope='shortcut')

        out = layers.conv2d(pre_act, nOutChannels[0], 1, stride=stride, scope='conv0')
        out = layers.batch_norm(out, scope='bn1')
        out = activation(out)
        out = layers.conv2d(out, nOutChannels[1], 3, stride=1, scope='conv1')
        out = layers.batch_norm(
            out, scope='bn2',
            param_initializers={'gamma': tf.initializers.constant(0.1)})
        out = activation(out)
        out = layers.conv2d(out, nOutChannels[2], 1, stride=1, scope='conv2')
        out = out + shortcut
        return out

