import tensorflow as tf
import numpy as np
# import horovod.tensorflow as hvd
import tfops as Z
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope


def get_conv_weight_np(filter_shape, stable_init=True):
    weight_np = np.random.randn(*filter_shape) * 0.02
    kcent = (filter_shape[0] - 1) // 2

    # Sample a random orthogonal matrix:
    w_ortho = np.linalg.qr(np.random.randn(
        *filter_shape[2:]))[0].astype('float32')

    if stable_init:
        weight_np[kcent, kcent, :, :] += w_ortho

    # weight_np[kcent, kcent, :, :] += 2 * np.eye(filter_shape[3])

    weight_np = weight_np.astype('float32')
    return weight_np


def compute_logdet(w_fft, width):
    """
    w_fft: a tensor indexed as: [c_out, c_in, v, u]

    """
    _, log_abs_determinant = \
        tf.linalg.slogdet(tf.transpose(w_fft, [2, 3, 0, 1]))

    # log_abs_determinant = \
    #     tf.log(tf.abs(tf.linalg.det(tf.transpose(w_fft, [2, 3, 0, 1]))))

    # # Take real part, imaginary part is zero.
    dlogdet = tf.real(log_abs_determinant)

    # dlogdet = tf.log(
    #     tf.abs(tf.linalg.det(tf.transpose(w_fft, [2, 3, 0, 1]))))

    # dlogdet = tf.reduce_sum(dlogdet)

    # Using det(A*) = det(A). Depending on whether the last spatial
    # dimension is even or odd, w_fft is repeated differently.
    s = slice(1, -1) if width % 2 == 0 else slice(1, None)
    dlogdet = tf.reduce_sum(dlogdet) + tf.reduce_sum(dlogdet[:, s])
    # dlogdet = tf.reduce_sum(tf.real(w_fft)) + tf.reduce_sum(tf.imag(w_fft))

    return dlogdet


def reindex(z, reverse=False):
    shift = 1

    shift = shift if not reverse else -shift
    # s_a = slice(shift, None)
    # s_b = slice(0, shift)

    z = tf.concat(
        (z[:, shift:], z[:, :shift]), axis=1)

    z = tf.concat(
        (z[:, :, shift:], z[:, :, :shift]), axis=2)

    return z


def fourier_conv(
        name, z, logdet, ksize=3, reverse=False,
        checkpoint_fn=None, use_fourier_forward=False):
    batchsize, height, width, n_channels = Z.int_shape(z)

    assert (ksize - 1) % 2 == 0

    with tf.variable_scope(name):
        filter_shape = [ksize, ksize, n_channels, n_channels]

        w_np = get_conv_weight_np(filter_shape)
        w = tf.get_variable('W', dtype=tf.float32, initializer=w_np)
        b = tf.get_variable('b', [n_channels],
                            initializer=tf.zeros_initializer())
        b = tf.reshape(b, [1, 1, 1, -1])

        f_shape = [height, width]

        def forward(z, w, logdet):
            padsize = (ksize - 1) // 2
            # Circular padding.
            z = tf.concat(
                (z[:, -padsize:, :], z, z[:, :padsize, :]),
                axis=1)

            z = tf.concat(
                (z[:, :, -padsize:], z, z[:, :, :padsize]),
                axis=2)

            # Circular convolution (due to padding.)
            z = tf.nn.conv2d(
                z, w, [1, 1, 1, 1],
                padding='VALID', data_format='NHWC')

            # Fourier transform for log determinant.
            w_fft = tf.spectral.rfft2d(
                tf.transpose(w, [3, 2, 0, 1])[:, :, ::-1, ::-1],
                fft_length=f_shape,
                name=None
            )
            dlogdet = compute_logdet(w_fft, width)

            logdet += dlogdet

            z = z + b

            return z, logdet

        def forward_fourier(x, w, logdet):
            # Dimension [b, c, v, u]
            x_fft = tf.spectral.rfft2d(
                tf.transpose(x, [0, 3, 1, 2]),
                fft_length=f_shape,
                name=None
            )

            # Dimension [b, 1, c_in, v, u]
            x_fft = tf.expand_dims(x_fft, 1)

            # Dimension [c_out, c_in, v, u]
            w_fft = tf.spectral.rfft2d(
                tf.transpose(w, [3, 2, 0, 1])[:, :, ::-1, ::-1],
                fft_length=f_shape,
                name=None
            )

            logdet += compute_logdet(w_fft, width)

            # Dimension [1, c_out, c_in, v, u]
            w_fft = tf.expand_dims(w_fft, 0)

            z_fft = tf.reduce_sum(
                tf.multiply(x_fft, w_fft), axis=2)

            z = tf.spectral.irfft2d(
                z_fft,
                fft_length=f_shape,
            )

            z = tf.transpose(z, [0, 2, 3, 1])

            z = reindex(z)

            z = z + b
            return z, logdet

        def inverse(z, logdet):
            z = z - b

            z = reindex(z, reverse=True)

            # Dimension [b, c_out, v, u]
            z_fft = tf.spectral.rfft2d(
                tf.transpose(z, [0, 3, 1, 2]),
                fft_length=f_shape,
                name=None
            )

            # Dimension [b, 1, c_out, v, u]
            z_fft = tf.expand_dims(z_fft, 1)

            # Dimension [c_out, c_in, v, u]
            w_fft = tf.spectral.rfft2d(
                tf.transpose(w, [3, 2, 0, 1])[:, :, ::-1, ::-1],
                fft_length=f_shape,
                name=None
            )

            dlogdet = compute_logdet(w_fft, width)

            # z_fft = tf.Print(
            #     z_fft, data=[dlogdet / height / width], message='dlogdet:')

            logdet -= dlogdet

            # Dimension [v, u, c_in, c_out], channels switched because of
            # inverse.
            w_fft_inv = tf.linalg.inv(
                tf.transpose(w_fft, [2, 3, 0, 1]),
                )
            # Dimension [c_in, c_out, v, u]
            w_fft_inv = tf.transpose(w_fft_inv, [2, 3, 0, 1])

            # Dimension [1, c_in, c_out, v, u]
            w_fft_inv = tf.expand_dims(w_fft_inv, 0)

            x_fft = tf.reduce_sum(
                tf.multiply(z_fft, w_fft_inv), axis=2)

            x = tf.spectral.irfft2d(
                x_fft,
                fft_length=f_shape,
            )

            x = tf.transpose(x, [0, 2, 3, 1])

            return x, logdet

        if not reverse:
            x = z

            if use_fourier_forward:
                z, logdet = forward_fourier(x, w, logdet)
            else:
                z, logdet = forward(x, w, logdet)

            return z, logdet

        else:
            z, logdet = inverse(z, logdet)

            return z, logdet


def rmse(a, b):
    return np.sqrt(np.mean(np.power(a - b, 2)))


def test_performance(layer, layer_kwargs):
    import time

    shape = [128, 32, 32, 48]
    N_iterations = 10

    x = tf.placeholder(
        tf.float32, shape, name='image')
    logdet = tf.zeros_like(x)[:, 0, 0, 0]

    print('layer', layer, layer_kwargs)

    with tf.variable_scope('test'):
        i = tf.Variable(tf.constant(0))

        logdet = tf.zeros_like(x)[:, 0, 0, 0]

        def condition(i, z, logdet):
            return tf.less(i, N_iterations)

        def body(i, z, logdet):
            with tf.variable_scope('body'):

                for l in range(32):
                    z, logdet = layer(
                        'layer{}'.format(l), z, logdet, reverse=False,
                        **layer_kwargs)

                i = tf.add(i, 1)

                z = z / tf.reduce_max(tf.abs(z))

                return i, z, logdet

        r = tf.while_loop(
            condition,
            body,
            [i, x, logdet],
            parallel_iterations=1,
            name='loop')

    x_np = np.random.randn(*shape).astype('float32')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    s = time.time()
    r_np = sess.run(r, feed_dict={x: x_np})

    print('Took {} seconds'.format(time.time() - s))
    # print(r_np[1][0, :, :, 0])

    tf.reset_default_graph()


def test_layer(layer, layer_kwargs):
    shape = [2, 7, 7, 3]

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

    z_np, recon_np, logdet_np, logdet_out_np = sess.run(
        [z, recon, logdet, logdet_out], feed_dict={x: x_np})

    z_np, recon_np, logdet_np, logdet_out_np = sess.run(
        [z, recon, logdet, logdet_out], feed_dict={x: x_np})

    z_recon_np = sess.run([z], feed_dict={x: recon_np})

    print('RMSE on x:\t', rmse(x_np, recon_np))

    print(
        'RMSE on conv(x):\t', rmse(z_np, z_recon_np))
    print('log det / dim: \t', rmse(logdet_np / np.prod(shape[1:]), 0))

    tf.reset_default_graph()


def test_derivative_fourier_conv():
    print('Testing gradients')
    shape = [128, 32, 32, 3]

    x = tf.placeholder(
        tf.float32, shape, name='image')
    x_np = np.random.randn(*shape).astype('float32')

    logdet = tf.zeros_like(x)[:, 0, 0, 0]

    with tf.variable_scope('test'):
        z = x
        z, logdet = fourier_conv(
            'layer', z, logdet, reverse=False)

    with tf.variable_scope('test', reuse=True):
        w = tf.get_variable('layer/W')

    f = tf.reduce_sum(logdet)

    grad = tf.gradients(f, w)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        w_np = sess.run(w)

        grad_np = sess.run(grad, feed_dict={x: x_np})

        delta = 0.0001

        v = np.random.randn(*Z.int_shape(w))

        finite_diff = (sess.run(f, feed_dict={x: x_np, w:w_np+delta*v}) - sess.run(f, feed_dict={x: x_np, w:w_np-delta*v})) / 2 / delta

        other_side = np.sum(grad_np * v)

        print(finite_diff, other_side, finite_diff - other_side)

        print(finite_diff / other_side)

    # tf.gradients(tf.reduce_sum(logdet), w)

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # test_performance(invertible_conv2D_fourier, {})
    # test_performance(invertible_conv2D_fourier, {})
    # test_performance(invertible_conv2D_fourier, {})
    for layer, kwargs in [(fourier_conv_stable, {}),
                          (fourier_conv, {})]:
        test_layer(layer, kwargs)


    # test_derivative()


    # test_performance(invertible_conv2D_fourier, {'use_fourier_forward':False})
    # test_performance(invertible_conv2D_fourier, {'use_fourier_forward':True})

