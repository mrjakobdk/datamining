//
// Created by mrjak on 20-07-2021.
//

#ifndef GPU_SYNC_GPU_UTILS_H
#define GPU_SYNC_GPU_UTILS_H

#include "curand_kernel.h"

void gpu_malloc(void **devPtr, size_t size);

void gpu_free(void *devPtr);

size_t gpu_max_memory_usage();

void gpu_reset_max_memory_usage();

size_t gpu_total_memory_usage();

void inclusive_scan(int *source, int *result, int n);

void inclusive_scan(unsigned int *source, unsigned int *result, int n);

int *gpu_malloc_int(int n);

unsigned int *gpu_malloc_unsigned_int(int n);

float *gpu_malloc_float(int n);

bool *gpu_malloc_bool(int n);

int *gpu_malloc_int_zero(int n);

float *gpu_malloc_float_zero(int n);

bool *gpu_malloc_bool_false(int n);

void copy_D_to_H(int *h_out, int *d_in, int n);

void copy_D_to_H(unsigned int *h_out, unsigned int *d_in, int n);

void copy_D_to_H(float *h_out, float *d_in, int n);

void copy_D_to_H(bool *h_out, bool *d_in, int n);

int *copy_D_to_H(int *d_array, int n);

float *copy_D_to_H(float *d_array, int n);

bool *copy_D_to_H(bool *d_array, int n);

int *copy_H_to_D(int *h_array, int n);

float *copy_H_to_D(float *h_array, int n);

bool *copy_H_to_D(bool *h_array, int n);

float *copy_D_to_D(float *d_array1, int n);

void copy_H_to_D(int *d_out, int *h_in, int n);

void copy_H_to_D(float *d_out, float *h_in, int n);

void copy_H_to_D(bool *d_out, bool *h_in, int n);

void copy_D_to_D(int *d_out, int *d_in, int n);

void copy_D_to_D(float *d_out, float *d_in, int n);

void copy_D_to_D(bool *d_out, bool *d_in, int n);

int copy_last_D_to_H(int *d_array, int n);

int copy_last_D_to_H(unsigned int *d_array, int n);

float copy_last_D_to_H(float *d_array, int n);

void gpu_set_all_zero(int *d_var, int n);

void gpu_set_all_zero(unsigned int *d_var, int n);

void gpu_set_all_zero(float *d_var, int n);

void gpu_set_all(int *d_var, int size, int value);

void gpu_set_all(float *d_var, int size, float value);

void print_array_gpu(int *x, int n);

void print_array_nonzero_gpu(int *x, int n);

void print_array_gpu(float *x, int n);

void print_array_gpu(bool *x, int n);

void print_array_gpu(float *d_X, int n, int m);

void print_array_gpu(int *d_X, int n, int m);

void print_array_gpu(bool *d_X, int n, int m);

__device__ int get_start(const int *d_array, const int idx);

__device__ int get_end(const int *d_array, const int idx);

void set(int *x, int i, int value);

void set(float *x, int i, float value);

void set(int *x, int *idx, int i, int value);

__global__
void set_all(float *d_X, float value, int n);

__global__
void set_all(int *d_X, int value, int n);

__global__
void set_all(bool *d_X, bool value, int n);

__global__
void init_seed(curandState *state, int seed);

void gpu_random_sample_locked(int *d_in, int k, int n, curandState *d_state, int *d_lock);

void gpu_gather_1d(int *d_result, int *d_source, int *d_indices, int length);


#endif //GPU_SYNC_GPU_UTILS_H
