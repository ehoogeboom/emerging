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
        np.ndarray[DTYPE_t, ndim=4] w0,
        np.ndarray[DTYPE_t, ndim=4] b0,
        np.ndarray[DTYPE_t, ndim=4] w1,
        np.ndarray[DTYPE_t, ndim=4] b1,
        np.ndarray[DTYPE_t, ndim=4] w2,
        np.ndarray[DTYPE_t, ndim=4] b2):
    assert w0.shape[3] == w1.shape[3]

    cdef int batchsize = z.shape[0]
    cdef int height = z.shape[1]
    cdef int width = z.shape[2]
    cdef int n_channels = z.shape[3]

    cdef int depth = w0.shape[3]

    cdef int ksize_0 = w0.shape[0]
    cdef int ksize_1 = w1.shape[0]
    cdef int ksize_2 = w2.shape[0]
    cdef int offset_0 = (ksize_0 - 1) // 2
    cdef int offset_1 = (ksize_1 - 1) // 2
    cdef int offset_2 = (ksize_2 - 1) // 2

    cdef np.ndarray[DTYPE_t, ndim=4] h0 = np.zeros([batchsize, height, width, depth], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] h1 = np.zeros([batchsize, height, width, depth], dtype=DTYPE)

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
                            for k in range(ksize_0):
                                for m in range(ksize_0):
                                    j_ = j + (k - offset_0)
                                    i_ = i + (m - offset_0)

                                    if not ((j_ >= 0) and (j_ < height)):
                                        continue

                                    if not ((i_ >= 0) and (i_ < width)):
                                        continue

                                    h0[b, j, i, w_outer] += w0[k, m, c_inner, w_outer] * x[b, j_, i_, c_inner]

                        # Add bias.
                        h0[b, j, i, w_outer] += b0[0, 0, 0, w_outer]
                        # ReLU.
                        h0[b, j, i, w_outer] = max(h0[b, j, i, w_outer], 0)

                        # Second convolution
                        for c_inner in range(depth):
                            for k in range(ksize_1):
                                for m in range(ksize_1):
                                    j_ = j + (k - offset_1)
                                    i_ = i + (m - offset_1)

                                    if not ((j_ >= 0) and (j_ < height)):
                                        continue

                                    if not ((i_ >= 0) and (i_ < width)):
                                        continue

                                    h1[b, j, i, w_outer] += w1[k, m, c_inner, w_outer] * h0[b, j_, i_, c_inner]

                        # Add bias term.
                        h1[b, j, i, w_outer] += b1[0, 0, 0, w_outer]
                        # ReLU.
                        h1[b, j, i, w_outer] = max(h1[b, j, i, w_outer], 0)

                    # Last convolution
                    for c_inner in range(depth):
                        for k in range(ksize_2):
                            for m in range(ksize_2):
                                j_ = j + (k - offset_2)
                                i_ = i + (m - offset_2)

                                if not ((j_ >= 0) and (j_ < height)):
                                    continue

                                if not ((i_ >= 0) and (i_ < width)):
                                    continue

                                mu[b, j, i, c_outer] += w2[k, m, c_inner, c_outer*2] * h1[b, j_, i_, c_inner]
                                loga[b, j, i, c_outer] += w2[k, m, c_inner, c_outer*2+1] * h1[b, j_, i_, c_inner]

                    mu[b, j, i, c_outer] += b2[0, 0, 0, c_outer*2]
                    loga[b, j, i, c_outer] += b2[0, 0, 0, c_outer*2+1] + 2.

                    scale[b, j, i, c_outer] = 1. / (1 + exp(-loga[b, j, i, c_outer]))

                    x[b, j, i, c_outer] = z[b, j, i, c_outer] - mu[b, j, i, c_outer]
                    x[b, j, i, c_outer] /= scale[b, j, i, c_outer]

    return x
