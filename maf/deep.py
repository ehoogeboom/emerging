import tensorflow as tf
import numpy as np
# import horovod.tensorflow as hvd
from maf.inverses.deep_cython import Inverse


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


def get_conv_weights(filter_shape, scale, zerodiagonal):
    k_y, k_x, n_in, n_out = filter_shape

    mask_np = get_conv_ar_mask(
        k_y, k_x, n_in, n_out, zerodiagonal=zerodiagonal)
    mask = tf.constant(mask_np)

    weight_np = get_conv_weight_np(filter_shape, scale)

    w = tf.get_variable(
            'W', dtype=tf.float32, initializer=weight_np)
    b = tf.get_variable(
            'b', [n_out], initializer=tf.zeros_initializer())
    b = tf.reshape(b, [1, 1, 1, -1])

    w = w * mask
    return w, b


def f_weights(n_channels, depth, ksize, n_layers, scale):
    assert n_layers > 2

    weights = []
    with tf.variable_scope('0'):
        filter_shape = [ksize, ksize, n_channels, depth]
        weights.append(
            get_conv_weights(filter_shape, scale, zerodiagonal=True))

    for i in range(1, n_layers-1):
        with tf.variable_scope('{}'.format(i)):
            filter_shape = [ksize, ksize, depth, depth]
            weights.append(
                get_conv_weights(filter_shape, scale, zerodiagonal=False))

    filter_shape = [ksize, ksize, depth, 2 * n_channels]
    with tf.variable_scope('{}'.format(n_layers-1)):
        weights.append(get_conv_weights(
            filter_shape, scale=0.15, zerodiagonal=False))

    ws = [item[0] for item in weights]
    bs = [item[1] for item in weights]

    return ws, bs


def maf_deep(
        name, z, logdet, depth=96, ksize=3, dilation=1, reverse=False,
        checkpoint_fn=None, n_layers=5, is_upper=False):
    if is_upper:
        z = z[:, ::-1, ::-1, ::-1]

    batchsize, height, width, n_channels = int_shape(z)

    assert depth > n_channels

    with tf.variable_scope(name):
        ws, bs = f_weights(n_channels, depth, ksize, n_layers, scale=0.05)

        def f(h):
            for l in range(n_layers - 1):
                h = tf.nn.conv2d(
                    h, ws[l], strides=[1, 1, 1, 1],
                    padding='SAME', data_format='NHWC') + bs[l]
                h = tf.nn.relu(h)

            r = tf.nn.conv2d(
                    h, ws[-1], strides=[1, 1, 1, 1],
                    padding='SAME', data_format='NHWC') + bs[-1]

            mu = r[:, :, :, ::2]
            loga = r[:, :, :, 1::2]

            scale = tf.nn.sigmoid(loga + 2.)

            return mu, scale

        if not reverse:
            mu, scale = f(z)

            z = z * scale + mu

            logdet += tf.reduce_sum(tf.log(scale), axis=(1, 2, 3))

        else:
            ws_tensor = tf.concat([
                tf.expand_dims(w, 0) for w in ws[1:-1]], axis=0)
            bs_tensor = tf.concat([
                tf.expand_dims(b, 0) for b in bs[1:-1]], axis=0)

            x = tf.py_func(
                Inverse(),
                inp=[z,
                     ws[0],
                     bs[0],
                     ws_tensor,
                     bs_tensor,
                     ws[-1],
                     bs[-1]
                     ],
                Tout=tf.float32,
                stateful=True,
                name='inverse',)

            x.set_shape(z.get_shape())

            mu, scale = f(x)

            logdet -= tf.reduce_sum(tf.log(scale), axis=(1, 2, 3))

            z = x

    if is_upper:
        z = z[:, ::-1, ::-1, ::-1]

    return z, logdet


def test_layer(layer, layer_kwargs):
    shape = [13, 6, 6, 2]

    x = tf.placeholder(
        tf.float32, shape, name='image')

    logdet = tf.zeros_like(x)[:, 0, 0, 0]

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

    z_recon_np = sess.run([z], feed_dict={x: recon_np})

    # print(x_np[0].transpose(2, 0, 1) - recon_np[0].transpose(2, 0, 1))

    # print('z')
    # print(z_np[0].transpose(2, 0, 1))

    def rmse(a, b):
        return np.sqrt(np.mean(np.power(a - b, 2)))

    print('RMSE on x:\t', rmse(x_np, recon_np))

    print(
        'RMSE on conv(x):\t', rmse(z_np, z_recon_np))
    # print('log det: \t', rmse(logdet_np, 0))
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

    for layer, kwargs in [(maf_deep, {'is_upper': False}),
                          (maf_deep, {'is_upper': True})
                          ]:
        test_layer(layer, kwargs)


    # print(get_conv_ar_mask(3, 3, 2, 2, zerodiagonal=True).transpose(3, 2, 0, 1))
