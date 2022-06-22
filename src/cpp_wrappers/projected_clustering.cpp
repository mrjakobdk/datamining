//
// Created by jakobrj on 6/3/22.
//
#include "../algorithms/projected_clustering/GPU_PROCLUS.cuh"
#include "../algorithms/projected_clustering/PROCLUS.h"

void GPU_FAST_PROCLUS_cpp(int *h_C, int *h_D, float *h_data, int n, int d, int k, int l, float a, float b,
                          float min_deviation, int termination_rounds) {
    GPU_FAST_PROCLUS(h_C, h_D, h_data, n, d, k, l, a, b, min_deviation, termination_rounds);
}


void PROCLUS_cpp(int *h_C, int *h_D, float *h_data, int n, int d, int k, int l, float a, float b,
                 float min_deviation, int termination_rounds) {
    PROCLUS(h_C, h_D, h_data, n, d, k, l, a, b, min_deviation, termination_rounds);
}