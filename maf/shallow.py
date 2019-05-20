import tensorflow as tf
import numpy as np
# import horovod.tensorflow as hvd
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from maf.inverses.shallow_cython import Inverse


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """
    Function to get autoregressive convolution
    """
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[l, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


def get_conv_weight_np(filter_shape, scale):
    weight_np = np.random.randn(*filter_shape) * scale
    weight_np = weight_np.astype('float32')
    return weight_np


def maf_shallow_emerging(
        name, z, logdet, ksize=3, reverse=False,
        checkpoint_fn=None):
    with tf.variable_scope(name):
        if not reverse:
            z, logdet = maf_ar_single_layer(
                'upper', z, logdet, is_upper=True, ksize=ksize)

            if checkpoint_fn is not None:
                z, logdet = checkpoint_fn(z, logdet)

            z, logdet = maf_ar_single_layer(
                'lower', z, logdet, is_upper=False, ksize=ksize)

            if checkpoint_fn is not None:
                z, logdet = checkpoint_fn(z, logdet)

            return z, logdet

        else:
            z, logdet = maf_ar_single_layer(
                'lower', z, logdet, is_upper=False, ksize=ksize, reverse=True)

            z, logdet = maf_ar_single_layer(
                'upper', z, logdet, is_upper=True, ksize=ksize, reverse=True)

            return z, logdet


@add_arg_scope
def get_weights(ksize, n_channels, scale, unit_testing=False):
    if unit_testing:
        scale = 0.05

    mask_np = get_conv_ar_mask(
            ksize, ksize, n_channels, 2*n_channels, zerodiagonal=True)

    mask = tf.constant(mask_np)

    filter_shape = [ksize, ksize, n_channels, 2*n_channels]

    weight_np = get_conv_weight_np(filter_shape, scale)

    w = tf.get_variable(
        'W', dtype=tf.float32, initializer=weight_np)

    if unit_testing:
        b = tf.get_variable(
            'b', [2*n_channels],
            initializer=tf.initializers.random_normal(stddev=0.1))
    else:
        b = tf.get_variable(
            'b', [2*n_channels],
            initializer=tf.zeros_initializer())
    b = tf.reshape(b, [1, 1, 1, -1])

    w = mask * w

    return w, b


def maf_ar_single_layer(name, z, logdet, is_upper, ksize=3, reverse=False):
    shape = int_shape(z)
    n_channels = shape[3]

    with tf.variable_scope(name):
        if is_upper:
            z = z[:, ::-1, ::-1, ::-1]

        w, b = get_weights(ksize, n_channels, scale=0.0001)

        def f(z):
            r = tf.nn.conv2d(
                z, w, strides=[1, 1, 1, 1],
                padding='SAME', data_format='NHWC') + b
            mu = r[:, :, :, ::2]
            loga = r[:, :, :, 1::2]
            scale = tf.nn.sigmoid(loga + 2.)
            return mu, scale

        if not reverse:
            mu, scale = f(z)

            z = z * scale + mu

            logdet += tf.reduce_sum(tf.log(scale), axis=(1, 2, 3))

        else:
            x = tf.py_func(
                Inverse(),
                inp=[z, w, b],
                Tout=tf.float32,
                stateful=True,
                name='conv2dinverse',)
            x.set_shape(z.get_shape())

            mu, scale = f(x)
            logdet -= tf.reduce_sum(tf.log(scale), axis=(1, 2, 3))

            z = x

        if is_upper:
            z = z[:, ::-1, ::-1, ::-1]

    return z, logdet


def test_performance(layer, layer_kwargs):
    import time

    shape = [128, 32, 32, 12]

    x = tf.placeholder(
        tf.float32, shape, name='image')

    logdet = tf.zeros_like(x)[:, 0, 0, 0]

    print('layer', layer)
    sess = tf.Session()

    s = time.time()

    N_iterations = 1000

    with tf.variable_scope('test'):
        i = tf.Variable(tf.constant(0))

        logdet = tf.zeros_like(x)[:, 0, 0, 0]

        def c(i, z, logdet):
            return tf.less(i, N_iterations)

        def body(i, z, logdet):
            with tf.variable_scope('body'):

                for l in range(10):
                    z, logdet = layer(
                        'layer{}'.format(l), z, logdet, reverse=False,
                        **layer_kwargs)

                i = tf.add(i, 1)

                z = z / tf.reduce_max(tf.abs(z))

                return i, z, logdet

        r = tf.while_loop(
            c,
            body,
            [i, x, logdet],
            parallel_iterations=1,
            name='loop')

    x_np = np.random.randn(*shape).astype('float32')
    sess.run(tf.global_variables_initializer())
    r_np = sess.run(r, feed_dict={x: x_np})

    # print(r_np)
    # for i in range(100):
    #     x_np = np.random.randn(*shape).astype('float32')

    #     z_np, logdet_np = sess.run(
    #         [z, logdet], feed_dict={x: x_np})

    print(layer_kwargs)
    print('Took {} seconds'.format(time.time() - s))
    print()

    tf.reset_default_graph()


def test_layer(layer, layer_kwargs):
    shape = [128, 32, 32, 6]

    x = tf.placeholder(
        tf.float32, shape, name='image')

    logdet = tf.zeros_like(x)[:, 0, 0, 0]
    with arg_scope([get_weights], unit_testing=True):
        with tf.variable_scope('test'):
            z, logdet = layer(
                'layer', x, logdet, reverse=False, **layer_kwargs)

        with tf.variable_scope('test', reuse=True):
            recon, logdet_out = layer(
                'layer', z, logdet, reverse=True, **layer_kwargs)

    print('layer', layer)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    x_np = np.random.randn(*shape).astype('float32')

    z_np, recon_np, logdet_np = sess.run(
        [z, recon, logdet_out], feed_dict={x: x_np})

    # # print(x_np[0, :, :, 0] - z_np[0, :, :, 0])
    # print()

    z_recon_np = sess.run([z], feed_dict={x: recon_np})

    def rmse(a, b):
        return np.sqrt(np.mean(np.power(a - b, 2)))

    print('RMSE on x:\t', rmse(x_np, recon_np))

    print(
        'RMSE on conv(x):\t', rmse(z_np, z_recon_np))
    print('log det: \t', rmse(logdet_np, 0))
    print('')

    tf.reset_default_graph()


if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # test_performance(maf_square_single_layer, {'is_upper': False})
    # test_performance(maf_ar_single_layer, {'is_upper': False})
    # test_performance(maf_square_single_layer, {'is_upper': False})
    # test_performance(maf_ar_single_layer, {'is_upper': False})
    # test_derivative()

    for layer, kwargs in [(maf_ar_single_layer, {'is_upper': False}),
                          (maf_ar_single_layer, {'is_upper': True}),
                          (maf_shallow_emerging, {}),
                          ]:
        test_layer(layer, kwargs)
