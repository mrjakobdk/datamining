//
// Created by jakobrj on 3/5/21.
//

#include "DBSCAN.h"
#include "../../structure/RTree.h"
#include <cmath>
#include <limits>
#include <queue>

#define NOISE  -1
#define UNCLASSIFIED  -2

void DBSCAN(int *C, float *data, int n, int d, double eps, int min_samples) {

    RTree tree(d, 10);
    for (int p = 0; p < n; p++) {
        tree.insert(new std::pair<int, float *>(p, &data[p * d]));
    }

    for (int i = 0; i < n; i++) {
        C[i] = UNCLASSIFIED;
    }

    int cluster_label = 0;

    std::queue<int> q;
    for (int i = 0; i < n; i++) {

        if (C[i] == UNCLASSIFIED) {
            std::vector<int> neighbors = tree.range(eps, &data[i*d]);

            if (neighbors.size() < min_samples) {
                C[i] = NOISE;
            } else {
                q.push(i);
                C[i] = cluster_label;

                while (!q.empty()) {
                    int p_id = q.front();
                    q.pop();

                    std::vector<int> neighbors = tree.range(eps, &data[p_id*d]);
                    if (neighbors.size() >= min_samples) {
                        for (int q_id: neighbors) {
                            if (C[q_id] == UNCLASSIFIED || C[q_id] == NOISE) {
                                if (C[q_id] == UNCLASSIFIED)
                                    q.push(q_id);
                                C[q_id] = cluster_label;
                            }
                        }
                    }
                }

                cluster_label++;
            }
        }
    }
}
