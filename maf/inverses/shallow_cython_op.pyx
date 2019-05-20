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
        np.ndarray[DTYPE_t, ndim=4] w,
        np.ndarray[DTYPE_t, ndim=4] bias):
    assert z.dtype == DTYPE and w.dtype == DTYPE

    cdef int batchsize = z.shape[0]
    cdef int height = z.shape[1]
    cdef int width = z.shape[2]
    cdef int n_channels = z.shape[3]
    cdef int ksize = w.shape[0]
    cdef int offset = (ksize - 1) // 2

    cdef np.ndarray[DTYPE_t, ndim=4] x = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] mu = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] loga = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=4] scale = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)

    cdef int b, j, i, c_out, c_in, k, m, j_, i_, _j, _i

    # Single threaded
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

                for c_out in range(n_channels):
                    for c_in in range(n_channels):
                        for k in range(ksize):
                            for m in range(ksize):
                                j_ = j + (k - offset)
                                i_ = i + (m - offset)

                                if not ((j_ >= 0) and (j_ < height)):
                                    continue

                                if not ((i_ >= 0) and (i_ < width)):
                                    continue

                                # mu computation
                                mu[b, j, i, c_out] += w[k, m, c_in, c_out*2] * x[b, j_, i_, c_in]

                                # loga computation
                                loga[b, j, i, c_out] += w[k, m, c_in, c_out*2+1] * x[b, j_, i_, c_in]

                    # mu computation
                    mu[b, j, i, c_out] += bias[0, 0, 0, c_out*2]

                    # loga computation
                    loga[b, j, i, c_out] += bias[0, 0, 0, c_out*2+1] + 2.

                    # Compute value for x
                    scale[b, j, i, c_out] = 1. / (1. + exp(-loga[b, j, i, c_out]))

                    x[b, j, i, c_out] = z[b, j, i, c_out] - mu[b, j, i, c_out]
                    x[b, j, i, c_out] /= scale[b, j, i, c_out]

    return x
