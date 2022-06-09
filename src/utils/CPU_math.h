//
// Created by mrjak on 14-07-2021.
//

#ifndef GPU_SYNC_CPU_MATH_H
#define GPU_SYNC_CPU_MATH_H

#include <vector>
#include <functional>
#include <math.h>
#include <algorithm>
#include <limits>

const float INF = std::numeric_limits<float>::infinity();
const float PI = 2 * acos(0.0);

float distance(float *x, float *y, int d);

float mean(float *x, int n);

float *full(int n, float value);

int *full(int n, int value);

float *full(int n, std::function<int(float)> func);

float *clone(float *array, int n);

bool contains(std::vector<int> A, int i);

void set_all(float *dists, int k, float value);

float *variance(float *D, int n, int d);

float *iqr(float *D, int n, int d);

float min(float a, float b);

int arg_min(const std::vector<float> &v);

float min(const std::vector<float> &v);

int maximum(int *v, int n);

void print_array(int *x, int n);

void print_array(int *x, int n, int m);

void print_array(bool *x, int n);

void compute_l2_norm_to_medoid(float *dist, float *h_data, int m_i, int n, int d);

void compute_l2_norm_to_medoid(float *dist, float *h_data, int *S, int m_i, int n, int d);

template<typename T>
extern int argmin_1d(T *values, int n);

int *random_sample(int *indices, int k, int n);

int argmax_1d(float *values, int n);

void index_wise_minimum(float *values_1, float *values_2, int n);

float mean_1d(float *values, int n);

std::pair<int, int> *argmin_2d(float **values, int n, int m);

float *compute_l1_norm_to_medoid(float *h_data, int m_i, bool *D_i, int n, int d);


#endif //GPU_SYNC_CPU_MATH_H
