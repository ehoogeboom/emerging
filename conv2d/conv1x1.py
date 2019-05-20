import tensorflow as tf
import numpy as np
# import horovod.tensorflow as hvd
import tfops as Z
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope


# Invertible 1x1 conv
@add_arg_scope
def invertible_1x1_conv(name, z, logdet, decomposition=None, reverse=False, unit_testing=False):
    shape = Z.int_shape(z)
    w_shape = [shape[3], shape[3]]

    if decomposition is None or decomposition == '':
        with tf.variable_scope(name):
            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')

            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)

            # dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
            dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]

            if not reverse:

                _w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += dlogdet

                return z, logdet
            else:
                # z = tf.Print(
                #     z,
                #     data=[dlogdet / shape[1] / shape[2]],
                #     message='logdet invconv foreach spatial location: ')

                _w = tf.matrix_inverse(w)
                _w = tf.reshape(_w, [1, 1]+w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet -= dlogdet

                return z, logdet

    elif decomposition == 'PLU' or decomposition == 'LU':
        # LU-decomposed version
        shape = Z.int_shape(z)
        with tf.variable_scope(name):

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
            # S = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, dtype)
            l = tf.cast(l, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u = tf.cast(u, dtype)

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            if True:
                u_inv = tf.matrix_inverse(u)
                l_inv = tf.matrix_inverse(l)
                p_inv = tf.matrix_inverse(p)
                w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            else:
                w_inv = tf.matrix_inverse(w)

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            if not reverse:

                w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet
            else:

                w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
                z = tf.nn.conv2d(
                    z, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet

    elif decomposition == 'QR':
        with tf.variable_scope(name):
            np_s = np.ones(shape[3], dtype='float32')
            np_u = np.zeros((shape[3], shape[3]), dtype='float32')

            if unit_testing:
                np_s = 1 + 0.02 * np.random.randn(shape[3]).astype('float32')
                np_u = np.random.randn(shape[3], shape[3]).astype('float32')

            np_u = np.triu(np_u, k=1).astype('float32')
            u_mask = np.triu(np.ones(w_shape, dtype='float32'), 1)

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

            if not reverse:
                w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet
            else:

                w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
                z = tf.nn.conv2d(
                    z, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet

    else:
        raise ValueError('Unkown decomposition: {}'.format(decomposition))


def test_performance(layer, layer_kwargs):
    import time

    shape = [128, 32, 32, 200]
    N_iterations = 10
    depth = 16

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

                for l in range(depth):
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

    # First run is usually slow.
    r_np = sess.run(r, feed_dict={x: x_np})

    s = time.time()
    r_np = sess.run(r, feed_dict={x: x_np})

    print('Took {} seconds'.format(time.time() - s))
    # print(r_np[1][0, :, :, 0])

    tf.reset_default_graph()


def test_layer(layer, layer_kwargs):
    shape = [128, 32, 32, 2]

    x = tf.placeholder(
        tf.float32, shape, name='image')

    logdet = tf.zeros_like(x)[:, 0, 0, 0]

    with arg_scope([invertible_1x1_conv], unit_testing=True):
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
        'RMSE on layer(x):\t', rmse(z_np, z_recon_np))
    print('log det: \t', rmse(logdet_np, 0))
    print('')

    tf.reset_default_graph()


if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # test_derivative()

    for layer, kwargs in [(invertible_1x1_conv, {'decomposition': None}),
                          (invertible_1x1_conv, {'decomposition': 'LU'}),
                          (invertible_1x1_conv, {'decomposition': 'QR'}),
                          ]:
        # test_layer(layer, kwargs)

        test_performance(layer, kwargs)
