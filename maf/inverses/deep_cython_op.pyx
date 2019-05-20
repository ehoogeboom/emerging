import numpy as np
cimport numpy as np
cimport cython
import cython.parallel as parallel
from libc.stdio cimport printf
from libc.math cimport exp


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def inverse(
        np.ndarray[DTYPE_t, ndim=4] z,
        np.ndarray[DTYPE_t, ndim=4] w_f,
        np.ndarray[DTYPE_t, ndim=4] b_f,
        np.ndarray[DTYPE_t, ndim=5] ws,
        np.ndarray[DTYPE_t, ndim=5] bs,
        np.ndarray[DTYPE_t, ndim=4] w_l,
        np.ndarray[DTYPE_t, ndim=4] b_l):

    cdef int batchsize = z.shape[0]
    cdef int height = z.shape[1]
    cdef int width = z.shape[2]
    cdef int n_channels = z.shape[3]
    cdef int ksize = w_f.shape[0]

    cdef int n_layers = ws.shape[0]
    cdef int depth = ws.shape[4]

    cdef int offset = (ksize - 1) // 2

    cdef np.ndarray[DTYPE_t, ndim=5] h = np.zeros([batchsize, n_layers+1, height, width, depth], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] x = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] mu = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] loga = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] scale = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)

    cdef int widening = depth // n_channels

    cdef int b, j, i, c_outer, c_inner, k, m, j_, i_, _j, _i, l, w, w_outer

    # # Single threaded
    # for b in range(batchsize):

    # Multi-threaded
    for b in parallel.prange(batchsize, nogil=True):

        # Debug multi-threaded
        # cdef int thread_id = parallel.threadid()
        # printf("Thread ID: %d\n", thread_id)

        for _j in range(height):
            j = height - _j - 1

            for _i in range(width):
                i = width - _i - 1

                for c_outer in range(n_channels):
                    # First convolution
                    for w in range(widening):
                        w_outer = c_outer * widening + w

                        for c_inner in range(n_channels):
                            for k in range(ksize):
                                for m in range(ksize):
                                    j_ = j + (k - offset)
                                    i_ = i + (m - offset)

                                    if not ((j_ >= 0) and (j_ < height)):
                                        continue

                                    if not ((i_ >= 0) and (i_ < width)):
                                        continue

                                    h[b, 0, j, i, w_outer] += w_f[k, m, c_inner, w_outer] * x[b, j_, i_, c_inner]

                        # Add bias.
                        h[b, 0, j, i, w_outer] += b_f[0, 0, 0, w_outer]
                        # ReLU.
                        h[b, 0, j, i, w_outer] = max(h[b, 0, j, i, w_outer], 0)

                        # All other convolutions
                        for l in range(n_layers):

                            # Note: c_inner runs over depth!!
                            for c_inner in range(depth):
                                for k in range(ksize):
                                    for m in range(ksize):
                                        j_ = j + (k - offset)
                                        i_ = i + (m - offset)

                                        if not ((j_ >= 0) and (j_ < height)):
                                            continue

                                        if not ((i_ >= 0) and (i_ < width)):
                                            continue

                                        h[b, l+1, j, i, w_outer] += ws[l, k, m, c_inner, w_outer] * h[b, l, j_, i_, c_inner]

                            # Add bias term.
                            h[b, l+1, j, i, w_outer] += bs[l, 0, 0, 0, w_outer]
                            # ReLU.
                            h[b, l+1, j, i, w_outer] = max(h[b, l+1, j, i, w_outer], 0)

                    # Last convolution
                    for c_inner in range(depth):
                        for k in range(ksize):
                            for m in range(ksize):
                                j_ = j + (k - offset)
                                i_ = i + (m - offset)

                                if not ((j_ >= 0) and (j_ < height)):
                                    continue

                                if not ((i_ >= 0) and (i_ < width)):
                                    continue

                                mu[b, j, i, c_outer] += w_l[k, m, c_inner, c_outer*2] * h[b, n_layers, j_, i_, c_inner]
                                loga[b, j, i, c_outer] += w_l[k, m, c_inner, c_outer*2+1] * h[b, n_layers, j_, i_, c_inner]

                    mu[b, j, i, c_outer] += b_l[0, 0, 0, c_outer*2]
                    loga[b, j, i, c_outer] += b_l[0, 0, 0, c_outer*2+1] + 2.

                    scale[b, j, i, c_outer] = 1. / (1. + exp(-loga[b, j, i, c_outer]))

                    x[b, j, i, c_outer] = z[b, j, i, c_outer] - mu[b, j, i, c_outer]
                    x[b, j, i, c_outer] /= scale[b, j, i, c_outer]

    return x
