//
// Created by jakobrj on 5/31/22.
//

#import <stdio.h>

#include "../algorithms/clustering/GPU_DBSCAN.cuh"
#include "../algorithms/clustering/EGG_SynC.cuh"
#include "../algorithms/clustering/SynC.h"

#include "clustering.h"

void GPU_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts) {
    GPU_DBSCAN(h_C, h_data, n, d, eps, minPts);
}

void G_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts) {
    G_DBSCAN(h_C, h_data, n, d, eps, minPts);
}

void EGG_SynC_cpp(int *h_C, float *h_data, int n, int d, float eps) {
    EGG_SynC(h_C, h_data, n, d, eps);
}

void SynC_cpp(int *h_C, float *h_data, int n, int d, float eps, float lam) {
    SynC(h_C, h_data, n, d, eps, lam);
}