import tensorflow as tf
import numpy as np
# import horovod.tensorflow as hvd
import tfops as Z
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
from conv2d.inverses.inverse_cython import Inverse


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


def invertible_conv2D_emerging_1x1(
        name, z, logdet, ksize=3, dilation=1, reverse=False,
        checkpoint_fn=None, decomposition=None, unit_testing=False):
    shape = Z.int_shape(z)
    batchsize, height, width, n_channels = shape

    assert (ksize - 1) % 2 == 0

    kcent = (ksize - 1) // 2

    with tf.variable_scope(name):
        if decomposition is None or decomposition == '':
            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                shape[3], shape[3]))[0].astype('float32')
            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)
            dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]
            w_inv = tf.matrix_inverse(w)

        elif decomposition == 'PLU' or decomposition == 'LU':
            # LU-decomposed version
            dtype = 'float64'

            # Random orthogonal matrix:
            import scipy
            np_w = scipy.linalg.qr(np.random.randn(shape[3], shape[3]))[
                0].astype('float32')

            np_p, np_l, np_u = scipy.linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p = tf.get_variable("P", initializer=np_p, trainable=False)
            l = tf.get_variable("L", initializer=np_l)
            sign_s = tf.get_variable(
                "sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, 'float64')
            l = tf.cast(l, 'float64')
            sign_s = tf.cast(sign_s, 'float64')
            log_s = tf.cast(log_s, 'float64')
            u = tf.cast(u, 'float64')

            l_mask = np.tril(np.ones([shape[3], shape[3]], dtype=dtype), -1)
            l = l * l_mask + tf.eye(shape[3], dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            u_inv = tf.matrix_inverse(u)
            l_inv = tf.matrix_inverse(l)
            p_inv = tf.matrix_inverse(p)
            w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            dlogdet = tf.reduce_sum(log_s) * (shape[1]*shape[2])

        elif decomposition == 'QR':
            np_s = np.ones(shape[3], dtype='float32')
            np_u = np.zeros((shape[3], shape[3]), dtype='float32')

            if unit_testing:
                np_s = 1 + 0.02 * np.random.randn(shape[3]).astype('float32')
                np_u = np.random.randn(shape[3], shape[3]).astype('float32')

            np_u = np.triu(np_u, k=1).astype('float32')
            u_mask = np.triu(np.ones([shape[3], shape[3]], dtype='float32'), 1)

            s = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            log_s = tf.log(tf.abs(s))

            r = u * u_mask + tf.diag(s)

            # Householder transformations
            I = tf.eye(shape[3])
            q = I
            for i in range(shape[3]):
                v_np = np.random.randn(shape[3], 1).astype('float32')
                v = tf.get_variable(
                    "v_{}".format(i), initializer=v_np)
                vT = tf.transpose(v)
                q_i = I - 2 * tf.matmul(v, vT) / tf.matmul(vT, v)

                q = tf.matmul(q, q_i)

            # Modified Gram–Schmidt process
            # def inner(a, b):
            #     return tf.reduce_sum(a * b)

            # def proj(v, u):
            #     return u * inner(v, u) / inner(u, u)

            # q = []
            # for i in range(shape[3]):
            #     v_np = np.random.randn(shape[3], 1).astype('float32')
            #     v = tf.get_variable("v_{}".format(i), initializer=v_np)
            #     for j in range(i):
            #         p = proj(v, q[j])
            #         v = v - proj(v, q[j])
            #     q.append(v)
            # q = tf.concat(q, axis=1)
            # q = q / tf.norm(q, axis=0, keepdims=True)

            q_inv = tf.transpose(q)
            r_inv = tf.matrix_inverse(r)

            w = tf.matmul(q, r)
            w_inv = tf.matmul(r_inv, q_inv)

            dlogdet = tf.reduce_sum(log_s) * (shape[1]*shape[2])
        else:
            raise ValueError('Unknown decomposition: {}'.format(decomposition))

        mask_np = get_conv_square_ar_mask(ksize, ksize, n_channels, n_channels)

        mask_upsidedown_np = mask_np[::-1, ::-1, ::-1, ::-1].copy()

        mask = tf.constant(mask_np)
        mask_upsidedown = tf.constant(mask_upsidedown_np)

        filter_shape = [ksize, ksize, n_channels, n_channels]

        w1_np = get_conv_weight_np(filter_shape)
        w2_np = get_conv_weight_np(filter_shape)
        w1 = tf.get_variable('W1', dtype=tf.float32, initializer=w1_np)
        w2 = tf.get_variable('W2', dtype=tf.float32, initializer=w2_np)
        b = tf.get_variable('b', [n_channels],
                            initializer=tf.zeros_initializer())
        b = tf.reshape(b, [1, 1, 1, -1])

        w1 = w1 * mask
        w2 = w2 * mask_upsidedown

        def log_abs_diagonal(w):
            return tf.log(tf.abs(tf.diag_part(w[kcent, kcent])))

        def forward(z, logdet):
            w_ = tf.reshape(w, [1, 1] + [shape[3], shape[3]])
            z = tf.nn.conv2d(z, w_, [1, 1, 1, 1],
                             'SAME', data_format='NHWC')

            logdet += dlogdet

            z = tf.nn.conv2d(
                z, w1, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='SAME', data_format='NHWC')
            logdet += tf.reduce_sum(log_abs_diagonal(w1)) * (height * width)

            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = tf.nn.conv2d(
                z, w2, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='SAME', data_format='NHWC')
            logdet += tf.reduce_sum(log_abs_diagonal(w2)) * (height * width)

            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = z + b
            return z, logdet

        def forward_fast(z, logdet):
            """
            Convolution with [(k+1) // 2]^2 filters.
            """
            # Smaller versions of w1, w2.
            w1_s = w1[kcent:, kcent:, :, :]
            w2_s = w2[:-kcent, :-kcent, :, :]

            pad = kcent * dilation

            # standard filter shape: [v, u, c_in, c_out]
            # standard fmap shape: [b, h, w, c]

            w_ = tf.transpose(
                tf.reshape(w, [1, 1] + [shape[3], shape[3]]), (0, 1, 3, 2))
            w_equiv = tf.nn.conv2d(
                tf.transpose(w1_s, (3, 0, 1, 2)), w_, [1, 1, 1, 1],
                padding='SAME')

            w_equiv = tf.transpose(w_equiv, (1, 2, 3, 0))

            z = tf.pad(z, [[0, 0], [0, pad], [0, pad], [0, 0]], 'CONSTANT')
            z = tf.nn.conv2d(
                z, w_equiv, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='VALID',
                data_format='NHWC')

            logdet += tf.reduce_sum(log_abs_diagonal(w1)) * (height * width)
            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = tf.pad(z, [[0, 0], [pad, 0], [pad, 0], [0, 0]], 'CONSTANT')

            z = tf.nn.conv2d(
                z, w2_s, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='VALID',
                data_format='NHWC')

            logdet += tf.reduce_sum(log_abs_diagonal(w2)) * (height * width)
            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = z + b
            return z, logdet

        if not reverse:
            x, logdet = forward_fast(z, logdet)
            # x_, _ = forward(z, logdet)

            # x = tf.Print(
            #     x, data=[tf.reduce_mean(tf.square(x - x_))], message='diff')

            return x, logdet

        else:
            logdet -= dlogdet
            logdet -= tf.reduce_sum(log_abs_diagonal(w2)) * (height * width)

            x = tf.py_func(
                Inverse(is_upper=1, dilation=dilation),
                inp=[z, w2, b],
                Tout=tf.float32,
                stateful=True,
                name='conv2dinverse2',)

            logdet -= tf.reduce_sum(log_abs_diagonal(w1)) * (height * width)

            x = tf.py_func(
                Inverse(is_upper=0, dilation=dilation),
                inp=[x, w1, tf.zeros_like(b)],
                Tout=tf.float32,
                stateful=True,
                name='conv2dinverse1',)

            x.set_shape(z.get_shape())

            z_recon, _ = forward_fast(x, tf.zeros_like(logdet))

            w_inv = tf.reshape(w_inv, [1, 1] + [shape[3], shape[3]])
            x = tf.nn.conv2d(
                x, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
            logdet -= dlogdet

            # mse = tf.sqrt(tf.reduce_mean(tf.pow(z_recon - z, 2)))

            # x = tf.Print(
            #     x,
            #     data=[mse],
            #     message='RMSE of inverse',
            # )

            return x, logdet


def invertible_conv2D_emerging(
        name, z, logdet, ksize=3, dilation=1, reverse=False,
        checkpoint_fn=None):
    batchsize, height, width, n_channels = Z.int_shape(z)

    assert (ksize - 1) % 2 == 0

    kcent = (ksize - 1) // 2

    with tf.variable_scope(name):
        mask_np = get_conv_square_ar_mask(ksize, ksize, n_channels, n_channels)

        mask_upsidedown_np = mask_np[::-1, ::-1, ::-1, ::-1].copy()

        mask = tf.constant(mask_np)
        mask_upsidedown = tf.constant(mask_upsidedown_np)

        filter_shape = [ksize, ksize, n_channels, n_channels]

        w1_np = get_conv_weight_np(filter_shape)
        w2_np = get_conv_weight_np(filter_shape)
        w1 = tf.get_variable('W1', dtype=tf.float32, initializer=w1_np)
        w2 = tf.get_variable('W2', dtype=tf.float32, initializer=w2_np)
        b = tf.get_variable('b', [n_channels],
                            initializer=tf.zeros_initializer())
        b = tf.reshape(b, [1, 1, 1, -1])

        w1 = w1 * mask
        w2 = w2 * mask_upsidedown

        def log_abs_diagonal(w):
            return tf.log(tf.abs(tf.diag_part(w[kcent, kcent])))

        def forward(z, logdet):
            z = tf.nn.conv2d(
                z, w1, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='SAME', data_format='NHWC')
            logdet += tf.reduce_sum(log_abs_diagonal(w1)) * (height * width)

            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = tf.nn.conv2d(
                z, w2, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='SAME', data_format='NHWC')
            logdet += tf.reduce_sum(log_abs_diagonal(w2)) * (height * width)

            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = z + b
            return z, logdet

        def forward_fast(z, logdet):
            """
            Convolution with [(k+1) // 2]^2 filters.
            """
            # Smaller versions of w1, w2.
            w1_s = w1[kcent:, kcent:, :, :]
            w2_s = w2[:-kcent, :-kcent, :, :]

            pad = kcent * dilation

            z = tf.pad(z, [[0, 0], [0, pad], [0, pad], [0, 0]], 'CONSTANT')
            z = tf.nn.conv2d(
                z, w1_s, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='VALID',
                data_format='NHWC')

            logdet += tf.reduce_sum(log_abs_diagonal(w1)) * (height * width)
            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = tf.pad(z, [[0, 0], [pad, 0], [pad, 0], [0, 0]], 'CONSTANT')

            z = tf.nn.conv2d(
                z, w2_s, [1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='VALID',
                data_format='NHWC')

            logdet += tf.reduce_sum(log_abs_diagonal(w2)) * (height * width)
            if checkpoint_fn is not None:
                checkpoint_fn(z, logdet)

            z = z + b
            return z, logdet

        if not reverse:
            x, logdet = forward_fast(z, logdet)

            return x, logdet

        else:
            logdet -= tf.reduce_sum(log_abs_diagonal(w2)) * (height * width)

            x = tf.py_func(
                Inverse(is_upper=1, dilation=dilation),
                inp=[z, w2, b],
                Tout=tf.float32,
                stateful=True,
                name='conv2dinverse2',)

            logdet -= tf.reduce_sum(log_abs_diagonal(w1)) * (height * width)

            x = tf.py_func(
                Inverse(is_upper=0, dilation=dilation),
                inp=[x, w1, tf.zeros_like(b)],
                Tout=tf.float32,
                stateful=True,
                name='conv2dinverse1',)

            x.set_shape(z.get_shape())

            z_recon, _ = forward_fast(x, tf.zeros_like(logdet))

            # mse = tf.sqrt(tf.reduce_mean(tf.pow(z_recon - z, 2)))

            # x = tf.Print(
            #     x,
            #     data=[mse],
            #     message='RMSE of inverse',
            # )

            return x, logdet


def invertible_ar_conv2D(
        name, z, logdet, is_upper, ksize=3, dilation=1, reverse=False, ):
    shape = Z.int_shape(z)
    n_channels = shape[3]
    kcent = (ksize - 1) // 2

    with tf.variable_scope(name):
        mask_np = get_conv_ar_mask(ksize, ksize, n_channels, n_channels)
        if is_upper:
            mask_np = mask_np[::-1, ::-1, ::-1, ::-1].copy()

        mask = tf.constant(mask_np)

        filter_shape = [ksize, ksize, n_channels, n_channels]

        weight_np = get_conv_weight_np(filter_shape)

        w = tf.get_variable('W', dtype=tf.float32, initializer=weight_np)
        b = tf.get_variable('b', [n_channels],
                            initializer=tf.zeros_initializer())
        b = tf.reshape(b, [1, 1, 1, -1])

        w = mask * w

        log_abs_diagonal = tf.log(tf.abs(tf.diag_part(w[kcent, kcent])))

        if not reverse:
            z = tf.nn.conv2d(
                z, w, strides=[1, 1, 1, 1],
                dilations=[1, dilation, dilation, 1],
                padding='SAME', data_format='NHWC') + b
            logdet += tf.reduce_sum(log_abs_diagonal) * (shape[1]*shape[2])

            return z, logdet
        else:
            logdet -= tf.reduce_sum(log_abs_diagonal) * (shape[1]*shape[2])

            x = tf.py_func(
                Inverse(is_upper=is_upper, dilation=dilation),
                inp=[z, w, b],
                Tout=tf.float32,
                stateful=True,
                name='conv2dinverse',)

            z_recon = tf.nn.conv2d(
                x, w, [1, 1, 1, 1], padding='SAME', data_format='NHWC') + b

            mse = tf.sqrt(tf.reduce_mean(tf.pow(z_recon - z, 2)))

            x = tf.Print(
                x,
                data=[mse],
                message='RMSE of inverse',
            )

            x.set_shape(z.get_shape())

            return x, logdet


def test_derivative():
    shape = [128, 32, 32, 3]

    x = tf.placeholder(
        tf.float32, shape, name='image')
    x_np = np.random.randn(*shape).astype('float32')

    logdet = tf.zeros_like(x)[:, 0, 0, 0]

    with tf.variable_scope('test'):
        z = x
        z, logdet = invertible_ar_conv2D(
            'layer', z, logdet, is_upper=False, reverse=False)

    with tf.variable_scope('test', reuse=True):
        w = tf.get_variable('layer/W')

    f = tf.reduce_sum(-tf.square(z))# + tf.reduce_sum(logdet)

    grad = tf.gradients(f, w)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        w_np = sess.run(w)

        grad_np = sess.run(grad, feed_dict={x: x_np})

        delta = 0.01

        v = np.random.randn(*Z.int_shape(w))

        finite_diff = (sess.run(f, feed_dict={x: x_np, w:w_np+delta*v}) - sess.run(f, feed_dict={x: x_np, w:w_np-delta*v})) / 2 / delta

        other_side = np.sum(grad_np * v)

        print(finite_diff, other_side, finite_diff - other_side)

        # results = tf.test.compute_gradient_error(
        #     w,
        #     Z.int_shape(w),
        #     f,
        #     Z.int_shape(f),
        #     x_init_value=None,
        #     delta=0.001,
        #     init_targets=None,
        #     extra_feed_dict={x: x_np}
        # )
        # print(results)
    tf.reset_default_graph()


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

    for layer, kwargs in [(invertible_ar_conv2D, {'is_upper': False}),
                          (invertible_ar_conv2D, {'is_upper': True}),
                          (invertible_conv2D_emerging, {'dilation': 1}),
                          (invertible_conv2D_emerging, {'dilation': 2}),
                          (invertible_conv2D_emerging_1x1, {})]:
        test_layer(layer, kwargs)
