import numpy as np
import pyximport
pyximport.install(
    inplace=True,
    )

import maf.inverses.three_cython_op as inverse_op_cython


class Inverse():
    def __init__(self):
        np.set_printoptions(
            precision=2, threshold=None, edgeitems=None, linewidth=None,
            suppress=True, nanstr=None, infstr=None, formatter=None)

    def __call__(self, z, w_0, b_0, w_1, b_1, w_2, b_2):
        if np.isnan(z).any():
            return z

        x = inverse_op_cython.inverse(
            z, w_0, b_0, w_1, b_1, w_2, b_2)

        return x
