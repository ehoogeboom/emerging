import numpy as np
# import time
# from Cython.Build import cythonize
import pyximport
pyximport.install(
    inplace=True,
    )

import conv2d.inverses.inverse_op_cython as inverse_op_cython


class Inverse():
    def __init__(self, is_upper, dilation):
        self.is_upper = is_upper
        self.dilation = dilation

    def __call__(self, z, w, b):
        if np.isnan(z).any():
            return z

        # start = time.time()

        z = z - b

        z_np = np.array(z, dtype='float64')
        w_np = np.array(w, dtype='float64')

        ksize = w_np.shape[0]
        kcent = (ksize - 1) // 2

        diagonal = np.diag(w_np[kcent, kcent, :, :])

        alpha = 1. / np.min(np.abs(diagonal))
        alpha = max(1., alpha)

        w_np *= alpha

        x_np = inverse_op_cython.inverse_conv(
            z_np, w_np, int(self.is_upper), self.dilation)

        x_np *= alpha

        # print('Inverse \t alpha {} \t compute time: {:.2f} seconds'.format(
        #                                         alpha, time.time() - start))
        return x_np.astype('float32')
