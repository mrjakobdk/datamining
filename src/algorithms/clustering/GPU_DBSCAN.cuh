//
// Created by jakobrj on 5/31/22.
//

#ifndef GPU_DBSCAN_GPU_DBSCAN_CUH
#define GPU_DBSCAN_GPU_DBSCAN_CUH

void GPU_DBSCAN(int *C, float *data, int n, int d, float eps, int minPts);
void G_DBSCAN(int *C, float *data, int n, int d, float eps, int minPts);

#endif //GPU_DBSCAN_GPU_DBSCAN_CUH
