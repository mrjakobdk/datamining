//
// Created by mrjak on 14-07-2021.
//

#ifndef GPU_SYNC_SYNC_H
#define GPU_SYNC_SYNC_H

#include <vector>

using clustering_outliers = std::vector <std::vector<std::vector < int>>>;
using clustering = std::vector <std::vector<int>>;
using outliers = std::vector<int>;
using neighborhood = std::vector<int>;
using cluster = std::vector<int>;

void SynC(int *C, float *D, int n, int d, float eps, float lam);

//void SynC_parallel(int *C, float *D, int n, int d, float eps, float lam);

void FSynC(int *C, float *D, int n, int d, float eps, float lam);

#endif //GPU_SYNC_SYNC_H
