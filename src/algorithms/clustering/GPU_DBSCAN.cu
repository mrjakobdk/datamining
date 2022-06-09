//
// Created by jakobrj on 5/31/22.
//
#import <stdio.h>
#include <cooperative_groups.h>

#include "GPU_DBSCAN.cuh"
#include "../../utils/GPU_utils.cuh"

namespace cg = cooperative_groups;

#define BLOCK_SIZE 1024
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__
float euclidian_distance(int i_point, int j_point, const float *__restrict__ d_data, int d) {
    double distance = 0;
    for (int i_dimension = 0; i_dimension < d; i_dimension++) {
        double diff = d_data[i_point * d + i_dimension] - d_data[j_point * d + i_dimension];
        distance += diff * diff;
    }
    return sqrt(distance);
}

__global__
void kernel_neighborhood_sizes(int *__restrict__ d_neighborhood_sizes, const float *__restrict__ d_data, int n, int d,
                               float eps) {
    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {
        int number_of_neighbors = 0;
        for (int j_point = 0; j_point < n; j_point++) {
            float distance = euclidian_distance(i_point, j_point, d_data, d);
            if (distance <= eps) {
                number_of_neighbors++;
            }
        }
        d_neighborhood_sizes[i_point] = number_of_neighbors;
    }
}

__global__
void kernel_neighborhood_populate(int *d_neighborhood_points, int *d_neighborhood_ends, float *d_data, int n, int d,
                                  float eps) {
    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {
        int neighborhood_idx = 0;
        int neighborhood_start = get_start(d_neighborhood_ends, i_point);
        int *_neighborhood_points = &d_neighborhood_points[neighborhood_start];
        for (int j_point = 0; j_point < n; j_point++) {
            float distance = euclidian_distance(i_point, j_point, d_data, d);
            if (distance <= eps) {
                _neighborhood_points[neighborhood_idx] = j_point;
                neighborhood_idx++;
            }
        }
    }
}

__global__
void kernel_init_singleton_clusters(int *d_C, int *d_neighborhood_sizes, int n, int minPts) {
    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {
        d_C[i_point] = d_neighborhood_sizes[i_point] >= minPts ? i_point : -1;
    }
}

__global__
void kernel_expand_clusters(int *d_C, int *d_changed, int *d_neighborhood_points, int *d_neighborhood_ends,
                            int *d_neighborhood_sizes,
                            int n, int minPts) {
    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {
        int neighborhood_start = get_start(d_neighborhood_ends, i_point);
        int neighborhood_end = get_end(d_neighborhood_ends, i_point);
        for (int i_neighbor = neighborhood_start; i_neighbor < neighborhood_end; i_neighbor++) {
            int j_point = d_neighborhood_points[i_neighbor];
            if (d_neighborhood_sizes[j_point] >= minPts &&
                d_C[j_point] > d_C[i_point]) {//todo could be replace with just checking that C is positive
                d_C[i_point] = d_C[j_point];
                d_changed[0] = 1;
            }
        }
    }
}


__global__
void kernel_expand_clusters_CG(
        int *d_C, int *d_changed, int *d_neighborhood_points, int *d_neighborhood_ends, int *d_neighborhood_sizes,
        int n, int minPts
) {

    auto g = cg::this_grid();

    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {
        d_C[i_point] = d_neighborhood_sizes[i_point] >= minPts ? i_point : -1;
    }

    if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
        d_changed[0] = 1;
    }
    g.sync();

    while (d_changed[0]) {
        g.sync();
        if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
            d_changed[0] = 0;
        }
        g.sync();

        for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {
            int neighborhood_start = get_start(d_neighborhood_ends, i_point);
            int neighborhood_end = get_end(d_neighborhood_ends, i_point);
            for (int i_neighbor = neighborhood_start; i_neighbor < neighborhood_end; i_neighbor++) {
                int j_point = d_neighborhood_points[i_neighbor];
                if (d_neighborhood_sizes[j_point] >= minPts &&
                    d_C[j_point] > d_C[i_point]) {//todo could be replace with just checking that C is positive
                    d_C[i_point] = d_C[j_point];
                    d_changed[0] = 1;
                }
            }
        }
        g.sync();
    }
}


__global__
void kernel_expand_cluster(int *d_exp, int *d_changed, int *d_neighborhood_points, int *d_neighborhood_ends,
                           int *d_neighborhood_sizes,
                           int n, int minPts) {
    for (int i_point = threadIdx.x + blockIdx.x * blockDim.x; i_point < n; i_point += blockDim.x * gridDim.x) {
        int neighborhood_start = get_start(d_neighborhood_ends, i_point);
        int neighborhood_end = get_end(d_neighborhood_ends, i_point);
        for (int i_neighbor = neighborhood_start; i_neighbor < neighborhood_end; i_neighbor++) {
            int j_point = d_neighborhood_points[i_neighbor];
            if (d_neighborhood_sizes[j_point] >= minPts && d_exp[j_point] == 1 && d_exp[i_point] == 0) {
                d_exp[i_point] = 1;
                d_changed[0] = 1;
            }
        }
    }
}

void GPU_DBSCAN(int *h_C, float *h_data, int n, int d, float eps, int minPts) {
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);

    float *d_data = copy_H_to_D(h_data, n * d);
    int *d_C = gpu_malloc_int(n);
    int *d_neighborhood_sizes = gpu_malloc_int(n);
    int *d_neighborhood_ends = gpu_malloc_int(n);
    int *d_neighborhood_points;
    int *d_changed = gpu_malloc_int(1);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    ///compute neighborhoods
    kernel_neighborhood_sizes<<< number_of_blocks, number_of_threads >>>(
            d_neighborhood_sizes, d_data, n, d, eps
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    inclusive_scan(d_neighborhood_sizes, d_neighborhood_ends, n);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    int total_number_of_neighbors = copy_last_D_to_H(d_neighborhood_ends, n);
    d_neighborhood_points = gpu_malloc_int(total_number_of_neighbors);
    kernel_neighborhood_populate<<< number_of_blocks, number_of_threads >>>(
            d_neighborhood_points, d_neighborhood_ends, d_data, n, d, eps
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    ///compute clustering
//    kernel_init_singleton_clusters<<< number_of_blocks, number_of_threads >>>(
//            d_C, d_neighborhood_sizes, n, minPts
//    );
//    int changed = 1;
//    while (changed) {
//        gpu_set_all_zero(d_changed, 1);
//        kernel_expand_clusters<<< number_of_blocks, number_of_threads >>>(
//                d_C, d_changed, d_neighborhood_points, d_neighborhood_ends, d_neighborhood_sizes, n, minPts
//        );
//        changed = copy_last_D_to_H(d_changed, 1);
//    }

    void *params[] = {
            &d_C, &d_changed, &d_neighborhood_points, &d_neighborhood_ends, &d_neighborhood_sizes,
            &n, &minPts
    };
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int SMs = deviceProp.multiProcessorCount;

    dim3 grid(SMs);
    dim3 block(number_of_threads);
    cudaLaunchCooperativeKernel(
            (void *) kernel_expand_clusters_CG,
            grid,
            block,
            params,
            0,
            0
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());


    copy_D_to_H(h_C, d_C, n);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaFree(d_data);
    cudaFree(d_C);
    cudaFree(d_neighborhood_sizes);
    cudaFree(d_neighborhood_ends);
    cudaFree(d_neighborhood_points);
    cudaFree(d_changed);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}

void G_DBSCAN(int *h_C, float *h_data, int n, int d, float eps, int minPts) {
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);

    float *d_data = copy_H_to_D(h_data, n * d);
//    int *d_C = gpu_malloc_int(n);
    int *d_neighborhood_sizes = gpu_malloc_int(n);
    int *d_neighborhood_ends = gpu_malloc_int(n);
    int *d_neighborhood_points;
    int *d_changed = gpu_malloc_int(1);
    int *d_exp = gpu_malloc_int(n);
    int *h_exp = new int[n];

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    ///compute neighborhoods
    kernel_neighborhood_sizes<<< number_of_blocks, number_of_threads >>>(
            d_neighborhood_sizes, d_data, n, d, eps
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    inclusive_scan(d_neighborhood_sizes, d_neighborhood_ends, n);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    int total_number_of_neighbors = copy_last_D_to_H(d_neighborhood_ends, n);
    d_neighborhood_points = gpu_malloc_int(total_number_of_neighbors);
    kernel_neighborhood_populate<<< number_of_blocks, number_of_threads >>>(
            d_neighborhood_points, d_neighborhood_ends, d_data, n, d, eps
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    ///compute clustering
    int *h_neighborhood_sizes = copy_D_to_H(d_neighborhood_sizes, n);
    for (int i = 0; i < n; i++) {
        h_C[i] = -1;
    }
    int cluster_id = 0;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != -1 || h_neighborhood_sizes[i] < minPts) {
            continue;
        }

        gpu_set_all_zero(d_exp, n);
        gpu_set_all(&d_exp[i], 1, 1);
        int changed = 1;
        while (changed) {
            gpu_set_all_zero(d_changed, 1);
            kernel_expand_cluster<<< number_of_blocks, number_of_threads >>>(
                    d_exp, d_changed, d_neighborhood_points, d_neighborhood_ends, d_neighborhood_sizes, n, minPts
            );
            changed = copy_last_D_to_H(d_changed, 1);
        }
        copy_D_to_H(h_exp, d_exp, n);
        for (int j = 0; j < n; j++) {
            if (h_exp[j]) {
                h_C[j] = cluster_id;
            }
        }

        cluster_id++;
    }

//    copy_D_to_H(h_C, d_C, n);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaFree(d_data);
//    cudaFree(d_C);
    cudaFree(d_neighborhood_sizes);
    cudaFree(d_neighborhood_ends);
    cudaFree(d_neighborhood_points);
    cudaFree(d_changed);
    cudaFree(d_exp);
    delete h_exp;
    delete h_neighborhood_sizes;

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}