import math

import tensorflow as tf
from tensorflow.contrib import slim


@slim.add_arg_scope
def conv2d_same(inputs, num_outputs, kernel_size, stride, data_format='NHWC', scope=None):

    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           data_format=data_format, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        if data_format == 'NCHW':
            inputs = tf.pad(inputs,
                            [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           data_format=data_format, padding='VALID', scope=scope)


@slim.add_arg_scope
def basic_block(inputs, nOutChannels, stride=1, activation_fn=tf.nn.relu, scope=None):

    assert len(nOutChannels) == 2

    with tf.variable_scope(scope):

        nInChannels = inputs.get_shape()[1].value

        if nInChannels == nOutChannels[-1]:
            if stride == 1:
                shortcut = inputs
            else:
                shortcut = slim.max_pool2d(inputs, 1, stride=stride, scope='shortcut')
        else:
            shortcut = slim.conv2d(inputs, nOutChannels[1], [1, 1], stride=stride, activation_fn=None, scope='shortcut')

        residual = conv2d_same(inputs, nOutChannels[0], 3, stride=stride, scope='conv1')
        residual = slim.conv2d(residual, nOutChannels[1], 3, stride=1, scope='conv2')
        outputs = residual + shortcut
        return activation_fn(outputs)


@slim.add_arg_scope
def bottleneck_block(inputs, nOutChannels, stride=1, activation_fn=tf.nn.relu, scope=None):

    assert len(nOutChannels) == 3

    with tf.variable_scope(scope):

        nInChannels = inputs.get_shape()[1].value

        if nInChannels == nOutChannels[-1]:
            if stride == 1:
                shortcut = inputs
            else:
                shortcut = slim.max_pool2d(inputs, 1, stride=stride, scope='shortcut')
        else:
            shortcut = slim.conv2d(inputs, nOutChannels[2], [1, 1], stride=stride, activation_fn=None, scope='shortcut')

        residual = slim.conv2d(inputs, nOutChannels[0], 1, stride=1, scope='conv1')
        residual = conv2d_same(residual, nOutChannels[1], 3, stride=stride, scope='conv2')
        residual = slim.conv2d(residual, nOutChannels[2], 1, stride=1, scope='conv3')
        outputs = residual + shortcut
        return activation_fn(outputs)


@slim.add_arg_scope
def pre_basic_block(inputs, nOutChannels, stride=1, scope=None):

    assert len(nOutChannels) == 2

    with tf.variable_scope(scope):

        nInChannels = inputs.get_shape()[1].value
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if nInChannels == nOutChannels[-1]:
            if stride == 1:
                shortcut = inputs
            else:
                shortcut = slim.max_pool2d(inputs, 1, stride=stride, scope='shortcut')
        else:
            shortcut = slim.conv2d(
                preact, nOutChannels[1], [1, 1], stride=stride,
                activation_fn=None, normalizer_fn=None,
                scope='shortcut')

        residual = conv2d_same(preact, nOutChannels[0], 3, stride=stride, scope='conv1')
        residual = slim.conv2d(
            residual, nOutChannels[1], 3, stride=1,
            normalizer_fn=None, activation_fn=None,
            scope='conv2')
        outputs = residual + shortcut
        return outputs


@slim.add_arg_scope
def pre_bottleneck_block(inputs, nOutChannels, stride=1, activation_fn=tf.nn.relu, scope=None):

    assert len(nOutChannels) == 3

    with tf.variable_scope(scope):

        nInChannels = inputs.get_shape()[1].value
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if nInChannels == nOutChannels[-1]:
            if stride == 1:
                shortcut = inputs
            else:
                shortcut = slim.max_pool2d(inputs, 1, stride=stride, scope='shortcut')
        else:
            shortcut = slim.conv2d(
                preact, nOutChannels[2], [1, 1], stride=stride,
                normalizer_fn=None, activation_fn=None,
                scope='shortcut'
            )

        residual = slim.conv2d(preact, nOutChannels[0], 1, stride=1, scope='conv1')
        residual = conv2d_same(residual, nOutChannels[1], 3, stride=stride, scope='conv2')
        residual = slim.conv2d(
            residual, nOutChannels[2], 1, stride=1,
            activation_fn=None, normalizer_fn=None,
            scope='conv3'
        )
        outputs = residual + shortcut
        return outputs


def a_softmax(
        x, labels, n_class, global_step, m=4,
        decay_step=50, decay_rate=0.999,
        alpha_max=0.05, alpha_init=0., reuse=False,
        scope='ASoftmax'):
    """a implement of a-softmax.
    Args:
        x: input features. float32 tensor.
        labels: class labels. int32 tensor.
        scope: name scope.
        m: the m of a-softmax
        decay_step: decay step
        decay_rate: decay rate
    
    Returns:
        loss value. float32 scalar.
    """
    with tf.variable_scope(scope, reuse=reuse):

        PI = 3.141592653589793

        # create W
        W = tf.get_variable(
            'weights', [x.shape[-1], n_class],
            initializer=tf.truncated_normal_initializer(stddev=0.00005),
            dtype=tf.float32)

        W_normed = tf.nn.l2_normalize(W, 0)

        tf.add_to_collection('loss_weights', W)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, W)

        # compute output
        score = tf.matmul(x, W_normed)

        s_shape = tf.shape(score)

        # compute indices
        first_indices = tf.range(0, s_shape[0], 1)
        sec_indices = labels
        indices = tf.stack([first_indices, sec_indices], axis=1)

        # gather values
        x_gather = tf.gather_nd(score, indices)

        # compute (-1)^k*cos(m*theta) - 2*k
        x_norm = tf.norm(x, ord=2, axis=1, keep_dims=False) + 1e-8
        x_cos = x_gather / x_norm

        # compute k
        x_theta = tf.acos(x_cos)
        k = tf.floor(x_theta / (PI / m))
        k = tf.stop_gradient(k)

        # compute output
        if m == 4:
            out_cosm = 1. - 8*(tf.pow(x_cos, 2) - tf.pow(x_cos, 4))
        elif m == 3:
            out_cosm = 4*tf.pow(x_cos, 3) - 3*x_cos
        elif m == 2:
            out_cosm = 2*tf.pow(x_cos, 2) - 1.
        elif m == 1:
            out_cosm = x_cos
        else:
            raise Exception('Unsuported m: {:d}'.format(m))

        out_gather = tf.pow(-1., k)*out_cosm - 2*k
        out_gather = out_gather * x_norm

        # scatter values
        alpha = 1. - tf.train.exponential_decay(1. - alpha_init, global_step, decay_step, decay_rate)
        alpha = tf.minimum(alpha, alpha_max)
        out_scatter = tf.scatter_nd(indices, alpha * (out_gather - x_gather), s_shape) 
        score_modified = score + out_scatter

        # compute softmax loss
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=score,
                name='softmax_loss'))

        # compute a_softmax loss
        if m == 1:
            aloss = loss
        else:
            aloss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=score_modified,
                    name='softmax_loss'))


        return score, aloss, loss, alpha, tf.reduce_mean(x_theta)


def additive_angular_margin(
        x, labels, n_class, s=32, m=0.5,
        regularizer=None, reuse=False,
        scope='AAM'):
    """a implement of a-softmax.
    Args:
        x: input features. float32 tensor.
        labels: class labels. int32 tensor.
        scope: name scope.
        m: the m of a-softmax
        decay_step: decay step
        decay_rate: decay rate
    
    Returns:
        loss value. float32 scalar.
    """
    with tf.variable_scope(scope, reuse=reuse):

        # create W
        W = tf.get_variable(
            'weights', [x.shape[-1], n_class],
            initializer=tf.truncated_normal_initializer(stddev=0.001),
            regularizer=regularizer,
            dtype=tf.float32)

        W_normed = tf.nn.l2_normalize(W, 0)

        if W not in tf.get_collection('loss_weights'):
            tf.add_to_collection('loss_weights', W)
        if W not in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, W)

        # norm x
        x_normed = tf.nn.l2_normalize(x, 1)

        # compute output
        cos_thetha = tf.matmul(x_normed, W_normed)

        shape = tf.shape(cos_thetha)

        # compute indices
        first_indices = tf.range(0, shape[0], 1)
        sec_indices = labels
        indices = tf.stack([first_indices, sec_indices], axis=1)

        # gather values
        cos_thetha_yi = tf.gather_nd(cos_thetha, indices)
        sin_thetha_yi = tf.sqrt(1. - tf.square(cos_thetha_yi))
        cos_thetha_yi_m = cos_thetha_yi*math.cos(m) - sin_thetha_yi*math.sin(m)

        # theta_yi = tf.acos(cos_thetha_yi)
        # cos_thetha_yi_m = tf.cos(theta_yi + m)

        # scatter back
        cos_thetha_yi_m_scatter = tf.scatter_nd(indices, cos_thetha_yi_m - cos_thetha_yi, shape) 
        cos_thetha_m = cos_thetha + cos_thetha_yi_m_scatter

        # scale
        logits = cos_thetha_m * s

        # compute softmax
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits,
                name='additive_angular_margin'))

        return loss, cos_thetha
