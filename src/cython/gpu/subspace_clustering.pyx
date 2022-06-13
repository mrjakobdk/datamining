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

@cython.embedsignature(True)
def GPU_FAST_PROCLUS(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int k, int l, int A, int B, float min_deviation, int termination_rounds
):
    """
    Implementation of the GPU-FAST-PROCLUS algorithm [1].

    Parameters:
        **data** (NumPy float array of shape=(n_points, n_dims)) - The data

        **k** (int) - The number of cluster to find

        **l** (int) - The average number of dimensions in the subspace of each clustering

        **A** (int) - Constant for sampling the A*k data points

        **B** (int) - Constant for sampling the B*k medoids

        **min_deviation** (float) - The miminum deviation from the average cluster size

        **termination_rounds** (int) - The number of rounds without improvements before terminating

    Result:
        **C** (NumPy int array of shape=(n_points,)) - The cluster label for each data point

        **D** (NumPy int array of shape=(k, n_dims)) - Marks for each cluster which dimensions are used in the subspace


    References:
        [1] `Jakob Rødsgaard Jørgensen, Katrine Scheel, Ira Assent, Ajeet Ram Pathak, Anne C. Elster - GPU-FAST-PROCLUS: A Fast GPU-parallelized Approach to Projected Clustering <https://www.researchgate.net/profile/Jakob-Jorgensen-6/publication/359617765_GPU-FAST-PROCLUS_A_Fast_GPU-parallelized_Approach_to_Projected_Clustering/links/6245693b21077329f2e3df6b/GPU-FAST-PROCLUS-A-Fast-GPU-parallelized-Approach-to-Projected-Clustering.pdf>`_
    """
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

@cython.embedsignature(True)
def PROCLUS(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int k, int l, int A, int B, float min_deviation, int termination_rounds
):
    """
    Implementation of the PROCLUS algorithm [1].

    Parameters:
        **data** (NumPy float array of shape=(n_points, n_dims)) - The data

        **k** (int) - The number of cluster to find

        **l** (int) - The average number of dimensions in the subspace of each clustering

        **A** (int) - Constant for sampling the A*k data points

        **B** (int) - Constant for sampling the B*k medoids

        **min_deviation** (float) - The miminum deviation from the average cluster size

        **termination_rounds** (int) - The number of rounds without improvements before terminating

    Result:
        **C** (NumPy int array of shape=(n_points,)) - The cluster label for each data point

        **D** (NumPy int array of shape=(k, n_dims)) - Marks for each cluster which dimensions are used in the subspace

    References:
        [1] `Charu C. Aggarwal, Joel L. Wolf, Philip S. Yu, Cecilia Procopiuc, Jong Soo Park - Fast algorithms for projected clustering <https://dl.acm.org/doi/pdf/10.1145/304181.304188>`_
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    cdef np.ndarray[DTYPE_INT, ndim=2, mode='c'] D = np.zeros((k, d), dtype=np.int32, order='c')
    PROCLUS_cython(C, D, data, n, d, k, l, A, B, min_deviation, termination_rounds)
    return np.asarray(C), np.asarray(D)