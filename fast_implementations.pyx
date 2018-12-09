import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# def compute_hazard(float time, count_0, count_1, t, total):
#     cdef float h
#     h = 0
#     survivors = total
#     for ti in t:
#         if ti <= time:
#             h += count_1[ti] / survivors
#         survivors = survivors - count_1[ti] - count_0[ti]
#     return h

DTYPE = np.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def compute_hazard(float time, np.ndarray[DTYPE_t, ndim=1] count_0, np.ndarray[DTYPE_t, ndim=1] count_1, np.ndarray[DTYPE_t, ndim=1] t, int total):
    cdef float h
    cdef float ti
    h = 0
    cdef float survivors = total
    cdef float cnt_1
    N = len(t)
    for i in range(N): # ti in t:
        ti = t[i]
        if ti <= time:
            # h += count_1[i] / survivors
            cnt_1 = count_1[i]
            h += cnt_1 / survivors
        survivors = survivors - count_1[i] - count_0[i]
    return h

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def compute_logrank_sum(int N, int nb_inf, int nb_sup,
                        np.ndarray[DTYPE_t, ndim=2] count_inf,
                        np.ndarray[DTYPE_t, ndim=2] count_sup,
                        np.ndarray[DTYPE_t, ndim=2] y,
                        np.ndarray[DTYPE_t, ndim=2] d):
    cdef DTYPE_t int_nb_inf = nb_inf
    cdef DTYPE_t int_nb_sup = nb_sup
    for i in range(N):
        y[1, i] = int_nb_inf
        y[2, i] = int_nb_sup
        y[0, i] = y[1, i] + y[2, i]
        d[0, i] = d[1, i] + d[2, i]
        int_nb_inf = int_nb_inf - count_inf[i, 0]
        int_nb_sup = int_nb_sup - count_sup[i, 0]
    cdef float num = 0
    cdef float den = 0
    for i in range(N):
        if y[0, i] > 0:
            num += d[1, i] - y[1, i] * d[0, i] / float(y[0, i])
        if y[0, i] > 1:
            den += (y[1, i] / float(y[0, i])) * y[2, i] * ((y[0, i] - d[0, i]) / (y[0, i] - 1)) * d[0, i]
    if den == 0:
        return 0
    cdef float L = num / sqrt(den)
    return L