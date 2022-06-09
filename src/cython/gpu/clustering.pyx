import numpy as np
cimport numpy as np
cimport cython

# np.import_array()

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT


cdef extern from "../../cpp_wrappers/clustering.cpp":
    void GPU_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts)
    void G_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts)
    void EGG_SynC_cpp(int *h_C, float *h_data, int n, int d, float eps)
    void SynC_cpp(int *h_C, float *h_data, int n, int d, float eps, float lam)

### GPU-DBSCAN
cdef GPU_DBSCAN_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, float eps, int minPts
):
    GPU_DBSCAN_cpp(&C[0], &data[0,0], n, d, eps, minPts)

@cython.embedsignature(True)
def GPU_DBSCAN(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    float eps, int minPts
):
    """
    Implementation of the DBSCAN in the GPU-INSCY algorithm [1].


    Parameters:
        **data** (NumPy float array of shape=(n_points, n_dims)) - The data

        **eps** (float) - The neighborhood radius

        **minPts** (float) - Minimum number of points in the neighborhood


    Result:
        **C** (NumPy int array of shape=(n_points,)) - The cluster label for each data point


    References:
        [1] `Jakob Rødsgaard Jørgensen, Katrine Scheel, Ira Assent - GPU-INSCY: A GPU-Parallel Algorithm and Tree Structure for Efficient Density-based Subspace Clustering <https://www.researchgate.net/profile/Jakob-Jorgensen-6/publication/353142592_GPU-INSCY_A_GPU-Parallel_Algorithm_and_Tree_Structure_for_Efficient_Density-based_Subspace_Clustering/links/60e94fc41c28af34585991c0/GPU-INSCY-A-GPU-Parallel-Algorithm-and-Tree-Structure-for-Efficient-Density-based-Subspace-Clustering.pdf>`_
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    GPU_DBSCAN_cython(C, data, n, d, eps, minPts)
    return np.asarray(C)


### G-DBSCAN

cdef G_DBSCAN_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, float eps, int minPts
):
    G_DBSCAN_cpp(&C[0], &data[0,0], n, d, eps, minPts)

@cython.embedsignature(True)
def G_DBSCAN(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    float eps, int minPts
):
    """
    Implementation of the G-DBSCAN algorithm [1].


    Parameters:
        **data** (NumPy float array of shape=(n_points, n_dims)) - The data

        **eps** (float) - The neighborhood radius

        **minPts** (float) - Minimum number of points in the neighborhood


    Result:
        **C** (NumPy int array of shape=(n_points,)) - The cluster label for each data point


    References:
        [1] `Guilherme Andrade, Gabriel Ramos, Daniel Madeira, Rafael Sachetto, Renato Ferreira, Leonardo Rocha - G-DBSCAN: A GPU accelerated algorithm for density-based clustering <https://www.sciencedirect.com/science/article/pii/S1877050913003438>`_
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    G_DBSCAN_cython(C, data, n, d, eps, minPts)
    return np.asarray(C)

### EGG-SynC
cdef EGG_SynC_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, float eps
):
    EGG_SynC_cpp(&C[0], &data[0,0], n, d, eps)

@cython.embedsignature(True)
def EGG_SynC(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    float eps
):
    """
    Implementation of the EGG-SynC algorithm [1].


    Parameters:
        **data** (NumPy float array of shape=(n_points, n_dims)) - The data

        **eps** (float) - The neighborhood radius


    Result:
        **C** (NumPy int array of shape=(n_points,)) - The cluster label for each data point


    References:
        [1] `Jakob Rødsgaard Jørgensen, Ira Assent - ?? <??>`_
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    EGG_SynC_cython(C, data, n, d, eps)
    return np.asarray(C)

#SynC, SynC_parallel, FSynC
cdef SynC_cython(
    np.ndarray[int, ndim=1, mode='c'] C,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    int n, int d, float eps, float lam
):
    SynC_cpp(&C[0], &data[0,0], n, d, eps, lam)

@cython.embedsignature(True)
def SynC(
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] data,
    float eps, float lam
):
    """
    Implementation of the SynC algorithm [1].


    Parameters:
        **data** (NumPy float array of shape=(n_points, n_dims)) - The data

        **eps** (float) - The neighborhood radius

        **lam** (float) - Accuracy parameter


    Result:
        **C** (NumPy int array of shape=(n_points,)) - The cluster label for each data point


    References:
        [1] `Christian Böhm, Claudia Plant, Junming Shao, Qinli Yang - Clustering by synchronization <https://dl.acm.org/doi/pdf/10.1145/1835804.1835879>`_
    """
    cdef int n = data.shape[0]
    cdef int d = data.shape[1]
    cdef np.ndarray[DTYPE_INT, ndim=1, mode='c'] C = np.zeros(n, dtype=np.int32, order='c')
    SynC_cython(C, data, n, d, eps, lam)
    return np.asarray(C)