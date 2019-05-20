import numpy as np
import time
import pyximport
pyximport.install(
    inplace=True,
    )

import maf.inverses.deep_cython_op as inverse_op_cython


class Inverse():
    def __init__(self):
        np.set_printoptions(
            precision=2, threshold=None, edgeitems=None, linewidth=None,
            suppress=True, nanstr=None, infstr=None, formatter=None)

    def __call__(self, z, w_f, b_f, ws, bs, w_l, b_l):
        if np.isnan(z).any():
            return z

        start = time.time()

        x = inverse_op_cython.inverse(
            z, w_f, b_f, ws, bs, w_l, b_l)

        print('Inverse MAF compute time: {:.2f} seconds'.format(
                                                time.time() - start))
        return x
