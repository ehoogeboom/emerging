import numpy as np
import time


class Inverse():
    def __init__(self, is_upper, dilation):
        self.is_upper = is_upper
        self.dilation = dilation

    def __call__(self, z, w, b):
        if np.isnan(z).any():
            return z

        print('Inverse called')
        start = time.time()

        zs = z.shape
        batchsize, height, width, n_channels = zs
        ksize = w.shape[0]
        kcenter = (ksize-1) // 2

        # Subtract bias term.
        z = z - b

        diagonal = np.diag(w[kcenter, kcenter, :, :])
        # print(diagonal[np.argsort(diagonal)])
        factor = 1./np.min(diagonal)
        factor = max(1, factor)
        factor = 1.

        print('factor is', factor)
        # print('w is', w.transpose(3, 2, 0, 1))

        x_np = np.zeros(zs)
        z_np = np.array(z, dtype='float64')
        w_np = np.array(w, dtype='float64')

        w_np *= factor

        def filter2image(j, i, m, k):
            m_ = (m - kcenter) * self.dilation
            k_ = (k - kcenter) * self.dilation
            return j+k_, i+m_

        def in_bound(idx, lower, upper):
            return (idx >= lower) and (idx < upper)

        def reverse_range(n, reverse):
            if reverse:
                return range(n)
            else:
                return reversed(range(n))

        for b in range(batchsize):
            for j in reverse_range(height, self.is_upper):
                for i in reverse_range(width, self.is_upper):
                    for c_out in reverse_range(n_channels, not self.is_upper):
                        for c_in in range(n_channels):
                            for k in range(ksize):
                                for m in range(ksize):
                                    if k == kcenter and m == kcenter and \
                                            c_in == c_out:
                                        continue

                                    j_, i_ = filter2image(j, i, m, k)

                                    if not in_bound(j_, 0, height):
                                        continue

                                    if not in_bound(i_, 0, width):
                                        continue

                                    x_np[b, j, i, c_out] -= \
                                        w_np[k, m, c_in, c_out] \
                                        * x_np[b, j_, i_, c_in]

                        # Compute value for x
                        x_np[b, j, i, c_out] += z_np[b, j, i, c_out]
                        x_np[b, j, i, c_out] /= \
                            w_np[kcenter, kcenter, c_out, c_out]

        x_np = x_np * factor

        print('Total time to compute inverse {:.2f} seconds'.format(
                                                        time.time() - start))
        return x_np.astype('float32')
