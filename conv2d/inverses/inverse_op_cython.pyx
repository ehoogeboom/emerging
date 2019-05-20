import numpy as np
cimport numpy as np
cimport cython
import cython.parallel as parallel
from libc.stdio cimport printf


DTYPE = np.float64


ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def inverse_conv(np.ndarray[DTYPE_t, ndim=4] z_np, np.ndarray[DTYPE_t, ndim=4] w_np, int is_upper, int dilation):
    assert z_np.dtype == DTYPE and w_np.dtype == DTYPE

    cdef int batchsize = z_np.shape[0]
    cdef int height = z_np.shape[1]
    cdef int width = z_np.shape[2]
    cdef int n_channels = z_np.shape[3]
    cdef int ksize = w_np.shape[0]
    cdef int kcenter = (ksize - 1) // 2

    cdef np.ndarray[DTYPE_t, ndim=4] x_np = np.zeros([batchsize, height, width, n_channels], dtype=DTYPE)

    cdef int b, j, i, c_out, c_in, k, m, j_, i_, _j, _i, _c_out

    # Single threaded
    # for b in range(batchsize):

    # Multi-threaded. Set max number of threads to avoid mem crash.
    for b in parallel.prange(batchsize, nogil=True, num_threads=8):

        # Debug multi-threaded
        # cdef int thread_id = parallel.threadid()
        # printf("Thread ID: %d\n", thread_id)

        for _j in range(height):
            j = _j if is_upper else height - _j - 1

            for _i in range(width):
                i = _i if is_upper else width - _i - 1

                for _c_out in range(n_channels):
                    c_out = n_channels - _c_out - 1 if is_upper else _c_out

                    for c_in in range(n_channels):
                        for k in range(ksize):
                            for m in range(ksize):
                                if k == kcenter and m == kcenter and \
                                        c_in == c_out:
                                    continue

                                j_ = j + (k - kcenter) * dilation
                                i_ = i + (m - kcenter) * dilation

                                if not ((j_ >= 0) and (j_ < height)):
                                    continue

                                if not ((i_ >= 0) and (i_ < width)):
                                    continue

                                x_np[b, j, i, c_out] -= w_np[k, m, c_in, c_out] * x_np[b, j_, i_, c_in]

                    # Compute value for x
                    x_np[b, j, i, c_out] += z_np[b, j, i, c_out]
                    x_np[b, j, i, c_out] /= \
                        w_np[kcenter, kcenter, c_out, c_out]

    return x_np
