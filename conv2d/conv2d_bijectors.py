import tensorflow as tf
import numpy as np
import tfops as Z
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import tensorflow_probability as tfp
tfb = tfp.bijectors


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


def get_conv_square_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """
    Function to get autoregressive convolution with square shape.
    """
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[:, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


@add_arg_scope
def get_conv_weight_np(filter_shape, stable_init=True, unit_testing=False):
    weight_np = np.random.randn(*filter_shape) * 0.02
    kcent = (filter_shape[0] - 1) // 2
    if stable_init or unit_testing:
        weight_np[kcent, kcent, :, :] += 1. * np.eye(filter_shape[3])
    weight_np = weight_np.astype('float32')
    return weight_np


def invertible_conv2D_emerging(
        name, z, logdet, ksize=3, dilation=1, reverse=False,
        checkpoint_fn=None):
    batchsize, height, width, n_channels = Z.int_shape(z)

    assert (ksize - 1) % 2 == 0

    kcent = (ksize - 1) // 2

    with tf.variable_scope(name):
        mask_np = get_conv_square_ar_mask(
            ksize, ksize, n_channels, n_channels,
            zerodiagonal=True)[::-1, ::-1, ::-1, ::-1].copy()
        mask = tf.constant(mask_np)

        print(mask_np.transpose(3, 2, 0, 1))

        filter_shape = [ksize, ksize, n_channels, n_channels]

        w1_np = get_conv_weight_np(filter_shape)
        w2_np = get_conv_weight_np(filter_shape)
        w1 = tf.get_variable('W1', dtype=tf.float32, initializer=w1_np)
        w2 = tf.get_variable('W2', dtype=tf.float32, initializer=w2_np)
        b = tf.get_variable('b', [n_channels],
                            initializer=tf.zeros_initializer())
        b = tf.reshape(b, [1, 1, 1, -1])

        w1 = w1 * mask
        w2 = w2 * mask

        s_np = (1 + np.random.randn(n_channels) * 0.02).astype('float32')
        s = tf.get_variable('scale', dtype=tf.float32, initializer=s_np)
        s = tf.reshape(s, [1, 1, 1, n_channels])

        def flat(z):
            return tf.reshape(z, [batchsize, height * width * n_channels])

        def unflat(z):
            return tf.reshape(z, [batchsize, height, width, n_channels])

        def shift_and_log_scale_fn_volume_preserving_1(z_flat):
            z = unflat(z_flat)

            shift = tf.nn.conv2d(
                z, w1, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='SAME', data_format='NHWC')

            shift_flat = flat(shift)

            return shift_flat, tf.zeros_like(shift_flat)

        def shift_and_log_scale_fn_volume_preserving_2(z_flat):
            z = unflat(z_flat)

            shift = tf.nn.conv2d(
                z, w2, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='SAME', data_format='NHWC')

            shift_flat = flat(shift)

            return shift_flat, tf.zeros_like(shift_flat)

        flow1 = tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn_volume_preserving_1
            )

        flow2 = tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn_volume_preserving_2
            )

        def flip(z_flat):
            z = unflat(z_flat)
            z = z[:, ::-1, ::-1, ::-1]
            z = flat(z)
            return z

        def forward(z, logdet):
            z = z * s
            logdet += tf.reduce_sum(tf.log(tf.abs(s))) * (height * width)

            z_flat = flat(z)

            z_flat = flow1.forward(z_flat)

            z_flat = flip(z_flat)
            z_flat = flow2.forward(z_flat)
            z_flat = flip(z_flat)

            z = unflat(z_flat)

            z = z + b
            return z, logdet

        def inverse(z, logdet):
            z = z - b

            z_flat = flat(z)

            z_flat = flip(z_flat)
            z_flat = flow2.inverse(z_flat)
            z_flat = flip(z_flat)

            z_flat = flow1.inverse(z_flat)

            z = unflat(z_flat)

            z = z / s
            logdet -= tf.reduce_sum(tf.log(tf.abs(s))) * (height * width)

            z = unflat(z)

            return z, logdet

        if not reverse:
            x, logdet = forward(z, logdet)

            return x, logdet

        else:
            x, logdet = inverse(z, logdet)

            return x, logdet


def test_layer(layer, layer_kwargs):
    shape = [128, 32, 32, 3]

    x = tf.placeholder(
        tf.float32, shape, name='image')

    logdet = tf.zeros_like(x)[:, 0, 0, 0]

    with arg_scope([get_conv_weight_np], unit_testing=True):
        with tf.variable_scope('test'):
            z, logdet = layer(
                'layer', x, logdet, reverse=False, **kwargs)

        with tf.variable_scope('test', reuse=True):
            recon, logdet_out = layer(
                'layer', z, logdet, reverse=True, **kwargs)

    print('layer', layer)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    x_np = np.random.randn(*shape).astype('float32')

    z_np, recon_np, logdet_np = sess.run(
        [z, recon, logdet_out], feed_dict={x: x_np})

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

    # test_derivative()

    for layer, kwargs in [(invertible_conv2D_emerging, {'dilation': 1}),
                          # (invertible_conv2D_emerging, {'dilation': 2})
                          ]:
        test_layer(layer, kwargs)
