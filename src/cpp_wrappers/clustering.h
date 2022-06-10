//
// Created by jakobrj on 5/31/22.
//

#ifndef GPU_DBSCAN_DBSCAN_GPU_H
#define GPU_DBSCAN_DBSCAN_GPU_H

void DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts);

void GPU_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts);

void G_DBSCAN_cpp(int *h_C, float *h_data, int n, int d, float eps, int minPts);

void EGG_SynC_cpp(int *h_C, float *h_data, int n, int d, float eps);

#endif //GPU_DBSCAN_DBSCAN_GPU_H
