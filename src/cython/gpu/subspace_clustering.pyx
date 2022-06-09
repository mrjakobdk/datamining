import numpy as np
cimport numpy as np
cimport cython

# np.import_array()

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT


cdef extern from "../../cpp_wrappers/subspace_clustering.cpp":
    void GPU_FAST_PROCLUS_cpp(int *h_C, int *h_D, float *h_data, int n, int d, int k, int l, float a, float b,float min_deviation, int termination_rounds)
    void PROCLUS_cpp(int *h_C, int *h_D, float *h_data, int n, int d, int k, int l, float a, float b,float min_deviation, int termination_rounds)

### GPU-FAST-PROCLUS

cdef GPU_FAST_PROCLUS_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[int, ndim=2, mode='c'] D,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, int k, int l, int A, int B, float min_deviation, int termination_rounds
):
    GPU_FAST_PROCLUS_cpp(&C[0], &D[0,0], &data[0,0], n, d, k, l, A, B, min_deviation, termination_rounds)

def GPU_FAST_PROCLUS(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int k, int l, int A, int B, float min_deviation, int termination_rounds
):
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    cdef np.ndarray[DTYPE_INT, ndim=2, mode='c'] D = np.zeros((k, d), dtype=np.int32, order='c')
    GPU_FAST_PROCLUS_cython(C, D, data, n, d, k, l, A, B, min_deviation, termination_rounds)
    return np.asarray(C), np.asarray(D)

### GPU-FAST-PROCLUS

cdef PROCLUS_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[int, ndim=2, mode='c'] D,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, int k, int l, int A, int B, float min_deviation, int termination_rounds
):
    PROCLUS_cpp(&C[0], &D[0,0], &data[0,0], n, d, k, l, A, B, min_deviation, termination_rounds)

def PROCLUS(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int k, int l, int A, int B, float min_deviation, int termination_rounds
):
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    cdef np.ndarray[DTYPE_INT, ndim=2, mode='c'] D = np.zeros((k, d), dtype=np.int32, order='c')
    PROCLUS_cython(C, D, data, n, d, k, l, A, B, min_deviation, termination_rounds)
    return np.asarray(C), np.asarray(D)