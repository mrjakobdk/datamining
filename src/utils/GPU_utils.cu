//
// Created by mrjak on 20-07-2021.
//

#include "GPU_utils.cuh"
#include <cstdio>
#include <map>

#define SECTION_SIZE 128
#define BLOCK_SIZE 512

__global__ void scan_kernel_eff(int *x, int *y, int n) {
    /**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }
}

__global__ void scan_kernel_eff(unsigned int *x, unsigned int *y, int n) {
    /**
 * from the cuda book
 */
    __shared__ unsigned int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }
}

__global__ void scan_kernel_eff_large1(int *x, int *y, int *S, int n) {
    /**
 * from the cuda book
 */
    __shared__ int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        S[blockIdx.x] = XY[SECTION_SIZE - 1];
    }
}

__global__ void scan_kernel_eff_large1(unsigned int *x, unsigned int *y, unsigned int *S, int n) {
    /**
 * from the cuda book
 */
    __shared__ unsigned int XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        XY[threadIdx.x] = x[i];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            XY[index] += XY[index - stride];
        }
    }

    for (int stride = SECTION_SIZE; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    if (i < n) {
        y[i] = XY[threadIdx.x];
    }

    if (threadIdx.x == 0) {
        S[blockIdx.x] = XY[SECTION_SIZE - 1];
    }
}

__global__ void scan_kernel_eff_large3(int *y, int *S, int n) {
    /**
 * from the cuda book
 */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n) {
        y[i] += S[blockIdx.x - 1];
    }
}

__global__ void scan_kernel_eff_large3(unsigned int *y, unsigned int *S, int n) {
    /**
 * from the cuda book
 */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < n) {
        y[i] += S[blockIdx.x - 1];
    }
}

size_t total_memory_usage = 0;
size_t max_memory_usage = 0;
std::map<void*, size_t> memory_usage_map;

void gpu_malloc(void **devPtr, size_t size) {
    cudaMalloc(devPtr, size);
    total_memory_usage += size;
    memory_usage_map[devPtr[0]] = size;
    if (total_memory_usage > max_memory_usage) {
        max_memory_usage = total_memory_usage;
    }
}

void gpu_free(void *devPtr){
    total_memory_usage -= memory_usage_map[devPtr];
    cudaFree(devPtr);
}

size_t gpu_max_memory_usage(){
    return max_memory_usage;
}

void gpu_reset_max_memory_usage(){
    max_memory_usage = 0;
}

size_t gpu_total_memory_usage(){
    return total_memory_usage;
}

void inclusive_scan(int *source, int *result, int n) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE)
        numBlocks++;

    if (n > SECTION_SIZE) {
        int *S;
        gpu_malloc((void **) &S, numBlocks * sizeof(int));
        scan_kernel_eff_large1<<<numBlocks, SECTION_SIZE>>>(source, result, S, n);
        inclusive_scan(S, S, numBlocks);
        scan_kernel_eff_large3<<<numBlocks, SECTION_SIZE>>>(result, S, n);
        gpu_free(S);
    } else {
        scan_kernel_eff<<<numBlocks, SECTION_SIZE>>>(source, result, n);
    }
}

void inclusive_scan(unsigned int *source, unsigned int *result, int n) {
    int numBlocks = n / SECTION_SIZE;
    if (n % SECTION_SIZE)
        numBlocks++;

    if (n > SECTION_SIZE) {
        unsigned int *S;
        gpu_malloc((void **) &S, numBlocks * sizeof(unsigned int));
        scan_kernel_eff_large1<<<numBlocks, SECTION_SIZE>>>(source, result, S, n);
        inclusive_scan(S, S, numBlocks);
        scan_kernel_eff_large3<<<numBlocks, SECTION_SIZE>>>(result, S, n);
        gpu_free(S);
    } else {
        scan_kernel_eff<<<numBlocks, SECTION_SIZE>>>(source, result, n);
    }
}

int *gpu_malloc_int(int n) {
    if (n <= 0)
        return nullptr;
    int *tmp;
    gpu_malloc((void**)&tmp, n * sizeof(int));
    return tmp;
}

unsigned int *gpu_malloc_unsigned_int(int n) {
    if (n <= 0)
        return nullptr;
    unsigned int *tmp;
    gpu_malloc((void**)&tmp, n * sizeof(unsigned int));
    return tmp;
}

float *gpu_malloc_float(int n) {
    if (n <= 0)
        return nullptr;
    float *tmp;
    gpu_malloc((void**)&tmp, n * sizeof(float));
    return tmp;
}

bool *gpu_malloc_bool(int n) {
    if (n <= 0)
        return nullptr;
    bool *tmp;
    gpu_malloc((void**)&tmp, n * sizeof(bool));
    return tmp;
}

int *gpu_malloc_int_zero(int n) {
    if (n <= 0)
        return nullptr;
    int *tmp;
    gpu_malloc((void**)&tmp, n * sizeof(int));
    cudaMemset(tmp, 0, n * sizeof(int));
    return tmp;
}

float *gpu_malloc_float_zero(int n) {
    if (n <= 0)
        return nullptr;
    float *tmp;
    gpu_malloc((void**)&tmp, n * sizeof(float));
    cudaMemset(tmp, 0, n * sizeof(float));
    return tmp;
}

bool *gpu_malloc_bool_false(int n) {
    if (n <= 0)
        return nullptr;
    bool *tmp;
    gpu_malloc((void**)&tmp, n * sizeof(bool));
    cudaMemset(tmp, 0, n * sizeof(bool));
    return tmp;
}

void copy_D_to_H(int *h_out, int *d_in, int n) {
    cudaMemcpy(h_out, d_in, n * sizeof(int), cudaMemcpyDeviceToHost);
}

void copy_D_to_H(unsigned int *h_out, unsigned int *d_in, int n) {
    cudaMemcpy(h_out, d_in, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

void copy_D_to_H(float *h_out, float *d_in, int n) {
    cudaMemcpy(h_out, d_in, n * sizeof(float), cudaMemcpyDeviceToHost);
}

void copy_D_to_H(bool *h_out, bool *d_in, int n) {
    cudaMemcpy(h_out, d_in, n * sizeof(bool), cudaMemcpyDeviceToHost);
}

int *copy_D_to_H(int *d_array, int n) {
    int *h_array = new int[n];
    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    return h_array;
}

float *copy_D_to_H(float *d_array, int n) {
    float *h_array = new float[n];
    cudaMemcpy(h_array, d_array, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h_array;
}

bool *copy_D_to_H(bool *d_array, int n) {
    bool *h_array = new bool[n];
    cudaMemcpy(h_array, d_array, n * sizeof(bool), cudaMemcpyDeviceToHost);
    return h_array;
}

int *copy_H_to_D(int *h_array, int n) {
    int *d_array = gpu_malloc_int(n);
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    return d_array;
}

float *copy_H_to_D(float *h_array, int n) {
    float *d_array = gpu_malloc_float(n);
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
    return d_array;
}

bool *copy_H_to_D(bool *h_array, int n) {
    bool *d_array = gpu_malloc_bool(n);
    cudaMemcpy(d_array, h_array, n * sizeof(bool), cudaMemcpyHostToDevice);
    return d_array;
}

float *copy_D_to_D(float *d_array1, int n) {
    float *d_array2 = gpu_malloc_float(n);
    cudaMemcpy(d_array2, d_array1, n * sizeof(float), cudaMemcpyDeviceToDevice);
    return d_array2;
}

void copy_H_to_D(int *d_out, int *h_in, int n) {
    cudaMemcpy(d_out, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
}

void copy_H_to_D(float *d_out, float *h_in, int n) {
    cudaMemcpy(d_out, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_H_to_D(bool *d_out, bool *h_in, int n) {
    cudaMemcpy(d_out, h_in, n * sizeof(bool), cudaMemcpyHostToDevice);
}

void copy_D_to_D(int *d_out, int *d_in, int n) {
    cudaMemcpy(d_out, d_in, n * sizeof(int), cudaMemcpyDeviceToDevice);
}

void copy_D_to_D(float *d_out, float *d_in, int n) {
    cudaMemcpy(d_out, d_in, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

void copy_D_to_D(bool *d_out, bool *d_in, int n) {
    cudaMemcpy(d_out, d_in, n * sizeof(bool), cudaMemcpyDeviceToDevice);
}

int copy_last_D_to_H(int *d_array, int n) {
    int tmp = 0;

    if (n > 0) {
        copy_D_to_H(&tmp, &d_array[n - 1], 1);
    }

    return tmp;
}

int copy_last_D_to_H(unsigned int *d_array, int n) {
    unsigned int tmp = 0;

    if (n > 0) {
        copy_D_to_H(&tmp, &d_array[n - 1], 1);
    }

    return tmp;
}

float copy_last_D_to_H(float *d_array, int n) {
    float tmp = 0.;

    if (n > 0) {
        copy_D_to_H(&tmp, &d_array[n - 1], 1);
    }

    return tmp;
}

void gpu_set_all_zero(int *d_var, int n) {
    if (n > 0) {
        cudaMemset(d_var, 0, n * sizeof(int));
    }
}

void gpu_set_all_zero(unsigned int *d_var, int n) {
    if (n > 0) {
        cudaMemset(d_var, 0, n * sizeof(unsigned int));
    }
}

void gpu_set_all_zero(float *d_var, int n) {
    if (n > 0) {
        cudaMemset(d_var, 0, n * sizeof(float));
    }
}

__global__ void gpu_malloc_int_set_value_kernel(int *d_temp, int size, int value) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < size) {
        d_temp[idx] = value;
    }
}

void gpu_set_all(int *d_var, int size, int value) {
    int numBlocks = size / BLOCK_SIZE;
    if (size % BLOCK_SIZE) {
        numBlocks++;
    }
    gpu_malloc_int_set_value_kernel<<<numBlocks, min(size, BLOCK_SIZE)>>>(d_var, size, value);
}

__global__ void gpu_malloc_int_set_value_kernel(float *d_temp, int size, float value) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < size) {
        d_temp[idx] = value;
    }
}

void gpu_set_all(float *d_var, int size, float value) {
    int numBlocks = size / BLOCK_SIZE;
    if (size % BLOCK_SIZE) {
        numBlocks++;
    }
    gpu_malloc_int_set_value_kernel<<<numBlocks, min(size, BLOCK_SIZE)>>>(d_var, size, value);
}

__global__ void print_array_gpu_kernel(int *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 10 && x[i] > -1)
            printf(" ");
        if (x[i] < 100 && x[i] > -10)
            printf(" ");
        printf("%d ", x[i]);
    }
    printf("\n");
}

void print_array_gpu(int *x, int n) {
    print_array_gpu_kernel<<<1, 1>>>(x, n);
    cudaDeviceSynchronize();
}

__global__ void print_array_nonzero_gpu_kernel(int *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 10 && x[i] > -1)
            printf(" ");
        if (x[i] < 100 && x[i] > -10)
            printf(" ");
        if (x[i] < 1000 && x[i] > -100)
            printf(" ");
        if (x[i] != 0) {
            printf("%d ", x[i]);
        } else {
            printf("  ");
        }
    }
    printf("\n");
}

void print_array_nonzero_gpu(int *x, int n) {
    print_array_nonzero_gpu_kernel<<<1, 1>>>(x, n);
    cudaDeviceSynchronize();
}

__global__ void print_array_gpu_kernel(float *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");
}

void print_array_gpu(float *x, int n) {
    print_array_gpu_kernel<<<1, 1>>>(x, n);
    cudaDeviceSynchronize();
}

__global__ void print_array_gpu_kernel(bool *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i])
            printf("1 ");
        else
            printf("0 ");
    }
    printf("\n");
}

void print_array_gpu(bool *x, int n) {
    print_array_gpu_kernel<<<1, 1>>>(x, n);
    cudaDeviceSynchronize();
}

__global__ void print_array_gpu_kernel(float *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%f ", (float) x[i]);
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void print_array_gpu(float *d_X, int n, int m) {
    print_array_gpu_kernel<<<1, 1>>>(d_X, n, m);
    cudaDeviceSynchronize();
}

__global__ void print_array_gpu_kernel(int *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%d ", x[i]);
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void print_array_gpu(int *d_X, int n, int m) {
    print_array_gpu_kernel<<<1, 1>>>(d_X, n, m);
    cudaDeviceSynchronize();
}


__global__
void print_array_gpu_kernel(bool *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        if (x[i]) {
            printf("true ");
        } else {
            printf("false ");
        }
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }
    printf("\n");
}


void print_array_gpu(bool *d_X, int n, int m) {
    print_array_gpu_kernel << < 1, 1 >> > (d_X, n, m);
    cudaDeviceSynchronize();
}

__device__ int get_start(const int *d_array, const int idx) {
    return idx > 0 ? d_array[idx - 1] : 0;
}

__device__ int get_end(const int *d_array, const int idx) {
    return d_array[idx];
}


__global__
void set_kernel(int *x, int i, int value) {
    x[i] = value;
}

void set(int *x, int i, int value) {
    set_kernel << < 1, 1 >> > (x, i, value);
}


__global__
void set_kernel(float *x, int i, float value) {
    x[i] = value;
}

void set(float *x, int i, float value) {
    set_kernel << < 1, 1 >> > (x, i, value);
}


__global__
void set_kernel(int *x, int *idx, int i, int value) {
    x[i] = idx[value];
}

void set(int *x, int *idx, int i, int value) {
    set_kernel << < 1, 1 >> > (x, idx, i, value);
}


__global__
void set_all(float *d_X, float value, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d_X[i] = value;
    }
}



__global__
void set_all(int *d_X, int value, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d_X[i] = value;
    }
}

__global__
void set_all(bool *d_X, bool value, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d_X[i] = value;
    }
}

__global__
void init_seed(curandState *state, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}



__global__ void gpu_random_sample_kernel_locked_v2(int *d_in, int k, int n, curandState *state, int *d_lock) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {

        int j = curand(&state[threadIdx.x]) % (n);

        if (i > j) {
            int tmp = j;
            j = i;
            i = tmp;
        }

        bool success = false;

        while (!success) {
            if (atomicCAS(&d_lock[i], 0, 1) == 0) {
                if (i == j || atomicCAS(&d_lock[j], 0, 1) == 0) {
                    int tmp_idx = d_in[i];
                    d_in[i] = d_in[j];
                    d_in[j] = tmp_idx;

                    success = true;
                    if (i != j) {
                        atomicExch(&d_lock[j], 0);
                    }
                }
                atomicExch(&d_lock[i], 0);
            }
        }
    }
}

void gpu_random_sample_locked(int *d_in, int k, int n, curandState *d_state, int *d_lock) {
    cudaMemset(d_lock, 0, n * sizeof(int));
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    gpu_random_sample_kernel_locked_v2 << < number_of_blocks, min(k, BLOCK_SIZE) >> > (d_in, k, n, d_state, d_lock);
}


__global__
void gpu_gather_1d_kernel(int *d_source, int *d_indices, int length,
                          int *d_result) {//todo change order
    for (int j = 0; j < length; j++) {
        d_result[j] = d_source[d_indices[j]];
    }
}

void gpu_gather_1d(int *d_result, int *d_source, int *d_indices, int length) {
    gpu_gather_1d_kernel << < 1, 1 >> > (d_source, d_indices, length, d_result);
}