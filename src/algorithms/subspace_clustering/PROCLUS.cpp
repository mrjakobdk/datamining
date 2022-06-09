#include "PROCLUS.h"
#include "../../utils/CPU_math.h"
#include "../../utils/CPU_mem_util.h"
#include <numeric>
#include <cassert>
#include <time.h>
#include <cstdio>
#include <fstream>
#include <cmath>
#include "omp.h"

float manhattan_segmental_distance(bool *D_i, float *h_data, int m_i, int m_j, int d) {
    float sum = 0.;
    int size = 0;
    float *x_m_i = &h_data[m_i * d];
    float *x_m_j = &h_data[m_j * d];
    for (int k = 0; k < d; k++) {
        if (D_i[k]) {
            sum += std::abs(x_m_i[k] - x_m_j[k]);
            size++;
        }
    }
    return sum / size;
}

void greedy(int *M, float *dist, float *new_dist, float *h_data, int *S, int Bk, int Ak, int d) {

    int rnd_start = Ak / 2;//std::rand() % Ak
    M[0] = S[rnd_start];
    compute_l2_norm_to_medoid(dist, h_data, S, M[0], Ak, d);

    for (int i = 1; i < Bk; i++) {
        M[i] = S[argmax_1d(dist, Ak)];
        int m_i = M[i];

        compute_l2_norm_to_medoid(new_dist, h_data, S, m_i, Ak, d);
        index_wise_minimum(dist, new_dist, Ak);
    }
}

bool **find_dimensions(float * h_data, int **L, int *L_sizes, int *M, int k, int n, int d, int l) {

    float **X = array_2d<float>(k, d);
    float **Z = array_2d<float>(k, d);
    float *Y = array_1d<float>(k);
    bool **D = zeros_2d<bool>(k, d);
    float *sigma = array_1d<float>(k);


    //compute X,Y,Z
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            X[i][j] = 0;
        }
    }

    for (int i = 0; i < k; i++) {
        if (L_sizes[i] == 0) {

        } else {
            float *x_m_i = &h_data[M[i] * d];
            for (int p = 0; p < L_sizes[i]; p++) {
                int point = L[i][p];
                float *x_p = &h_data[point * d];
                for (int j = 0; j < d; j++) {
                    X[i][j] += std::abs(x_p[j] - x_m_i[j]);
                }
            }
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            if (L_sizes[i] == 0) {

            } else {
                X[i][j] /= L_sizes[i];
            }
        }
    }

    for (int i = 0; i < k; i++) {

        Y[i] = mean_1d(X[i], d);

        sigma[i] = 0;
        for (int j = 0; j < d; j++) {
            float sub = X[i][j] - Y[i];
            sigma[i] += sub * sub;
        }
        sigma[i] /= (d - 1);
        sigma[i] = std::sqrt(sigma[i]);

        for (int j = 0; j < d; j++) {
            if (sigma[i] == 0) //todo not good... case not defined
                Z[i][j] = 0;
            else
                Z[i][j] = (X[i][j] - Y[i]) / sigma[i];
        }
    }

    //# ensuring that we find atleast 2 for each and than the k*l #todo fast - sort first instead
    for (int i = 0; i < k; i++) {
        for (int _ = 0; _ < 2; _++) {
            int j = argmin_1d<float>(Z[i], d);
            Z[i][j] = std::numeric_limits<float>::max();
            D[i][j] = true;
        }
    }

    for (int _ = k * 2; _ < k * l; _++) {
        std::pair<int, int> *p_i_j = argmin_2d(Z, k, d);
        int i = p_i_j->first;
        int j = p_i_j->second;
        Z[i][j] = std::numeric_limits<float>::max();
        D[i][j] = true;
    }

    free(X, k);
    free(Z, k);
    free(Y);
    free(sigma);

    return D;
}

int *assign_points(float *h_data, bool **D, int *M, int n, int d, int k) {
    int *C = array_1d<int>(n);
    float **dist = array_2d<float>(n, k);
    for (int i = 0; i < k; i++) {

        float *tmp_dist = compute_l1_norm_to_medoid(h_data, M[i], D[i], n, d);
        for (int j = 0; j < n; j++) {
            dist[j][i] = tmp_dist[j];
        }
        free(tmp_dist);
    }

    for (int p = 0; p < n; p++) {
        int i = argmin_1d<float>(dist[p], k);
        C[p] = i;
    }

    for (int i = 0; i < k; i++) {
        C[M[i]] = i;
    }

    free(dist, n);

    return C;
}

float evaluate_cluster(float *h_data, bool **D, int *C, int n, int d, int k) {
    float **Y = zeros_2d<float>(k, d);
    float *w = array_1d<float>(k);

    float **means = zeros_2d<float>(k, d);
    int *counts = zeros_1d<int>(k);

    for (int i = 0; i < n; i++) {
        counts[C[i]] += 1;
        float *x_i = &h_data[i * d];
        for (int j = 0; j < d; j++) {
            if (D[C[i]][j]) {
                means[C[i]][j] += x_i[j];
            }
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            means[i][j] /= counts[i];
        }
    }

    for (int i = 0; i < n; i++) {
        float *x_i = &h_data[i*d];
        for (int j = 0; j < d; j++) {
            if (D[C[i]][j]) {
                Y[C[i]][j] += std::abs(x_i[j] - means[C[i]][j]);
            }
        }
    }
    for (int j = 0; j < d; j++) {
        for (int i = 0; i < k; i++) {
            if (D[i][j]) {
                Y[i][j] /= counts[i];
            }
        }
    }

    for (int i = 0; i < k; i++) {
        w[i] = 0.;
        int size = 0;
        for (int j = 0; j < d; j++) {
            if (D[i][j]) {
                w[i] += Y[i][j];
                size++;
            }
        }
        w[i] /= size;
    }


    float sum = 0;
    for (int i = 0; i < k; i++) {
        sum += counts[i] * w[i];
    }

    free(Y, k);
    free(w);
    free(means, k);
    free(counts);

    return sum / n;
}

bool *bad_medoids(int *C, int k, float min_deviation, int n) {
    bool *bad = zeros_1d<bool>(k);
    int *sizes = zeros_1d<int>(k);

    for (int i = 0; i < n; i++) {
        sizes[C[i]] += 1;
    }

    int first = argmin_1d<int>(sizes, k);

    bad[first] = true;

    for (int i = 0; i < k; i++) {
        if (sizes[i] < n / k * min_deviation) {
            bad[i] = true;
        }
    }

    free(sizes);

    return bad;
}

int *
replace_medoids(int *M, int M_length, int *M_best, bool *bad, int *M_random, int k) {
    int *M_kept = array_1d<int>(k);

    M_random = random_sample(M_random, k, M_length);

    int *M_current = array_1d<int>(k);

    int j = 0;
    for (int i = 0; i < k; i++) {
        if (!bad[i]) {
            M_current[i] = M_best[i];
            M_kept[j] = M_best[i];
            j += 1;
        }
    }

    int old_count = j;
    int p = 0;
    for (int i = 0; i < k; i++) {
        if (bad[i]) {
            bool is_in = true;
            while (is_in) {
                is_in = false;
                for (int q = 0; q < old_count; q++) {
                    if (M[M_random[p]] == M_kept[q]) {
                        is_in = true;
                        p++;
                        break;
                    }
                }
            }
            M_current[i] = M[M_random[p]];
            M_kept[j] = M[M_random[p]];
            j += 1;
            p += 1;
        }
    }

    free(M_kept);

    return M_current;
}

void remove_outliers(float *h_data, int *C, bool **D, int *M, int n, int k, int d) {
    float *delta = array_1d<float>(k);

    for (int i = 0; i < k; i++) {
        delta[i] = 1000000.;//todo not nice
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (i != j) {
                float msd = manhattan_segmental_distance(D[i], h_data, M[i], M[j], d);
                if (delta[i] > msd) {
                    delta[i] = msd;
                }
            }
        }
    }

    for (int p = 0; p < n; p++) {
        bool clustered = false;
        for (int i = 0; i < k; i++) {
            float msd = manhattan_segmental_distance(D[i], h_data, M[i], p, d);
            if (msd <= delta[i]) {
                clustered = true;
                break;
            }
        }
        if (!clustered) {
            C[p] = -1;
        }
    }
    free(delta);
}

void PROCLUS(int *h_C, int *h_D, float *h_data,
             int n, int d, int k, int l, float a, float b, float min_deviation, int termination_rounds) {

    /// Initialization Phase
    l = std::min(l, d);
    int Ak = std::min(n, int(a * k));
    int Bk = std::min(n, int(b * k));

    int *indices = array_1d<int>(n);
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    int *S = random_sample(indices, Ak, n);


    float *dist = array_1d<float>(n);
    float *new_dist = array_1d<float>(n);
    int *M = array_1d<int>(Bk);
    greedy(M, dist, new_dist, h_data, S, Bk, Ak, d);
    free(S);
    free(new_dist);

    // Iterative Phase
    float best_objective = std::numeric_limits<float>::max();
    int *M_current = array_1d<int>(k);

    indices = fill_with_indices(Bk);

    int *M_random = random_sample(indices, k, Bk);

    for (int i = 0; i < k; i++) {
        int r_i = M_random[i];
        M_current[i] = M[r_i];
    }

    int termination_criterion = 0;
    int *M_best = nullptr;
    int *C_best = nullptr;
    bool *bad = nullptr;

    int **L = array_2d<int>(k, n);
    int *L_sizes = array_1d<int>(k);

    while (termination_criterion < termination_rounds) {

        for (int i = 0; i < k; i++) {
            int m_i = M_current[i];
            compute_l2_norm_to_medoid(dist, h_data, M_current, m_i, k, d);
            dist[i] = std::numeric_limits<float>::max();
            float delta_i = *std::min_element(dist, dist + k);

            compute_l2_norm_to_medoid(dist, h_data, m_i, n, d);
            int j = 0;
            for (int p = 0; p < n; p++) {
                if (dist[p] <= delta_i) {
                    L[i][j] = p;
                    j += 1;
                }
            }
            L_sizes[i] = j;

        }

        bool **D = find_dimensions(h_data, L, L_sizes, M_current, k, n, d, l);
        int *C = assign_points(h_data, D, M_current, n, d, k);

        float objective_function = evaluate_cluster(h_data, D, C, n, d, k);

        free(D, k);

        termination_criterion += 1;
        if (objective_function < best_objective) {
            termination_criterion = 0;
            best_objective = objective_function;
            if (M_best != nullptr) {
                free(M_best);
            }
            M_best = M_current;
            if (C_best != nullptr) {
                free(C_best);
            }
            C_best = C;
            if (bad != nullptr) {
                free(bad);
            }
            bad = bad_medoids(C, k, min_deviation, n);
        } else {
            free(C);
            free(M_current);
        }

        M_current = replace_medoids(M, Bk, M_best, bad, M_random, k);
    }

    // Refinement Phase
    for (int i = 0; i < k; i++) {
        L_sizes[i] = 0;
    }

    for (int p = 0; p < n; p++) {
        L_sizes[C_best[p]] += 1;
    }

    int *l_j = zeros_1d<int>(k);
    for (int i = 0; i < n; i++) {
        int cl = C_best[i];
        L[cl][l_j[cl]] = i;
        l_j[cl] += 1;
    }

    bool **D = find_dimensions(h_data, L, L_sizes, M_best, k, n, d, l);

    int *C = assign_points(h_data, D, M_best, n, d, k);

    remove_outliers(h_data, C, D, M_best, n, k, d);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            h_D[i * k + j] = D[i][j];
        }
    }

    for (int i = 0; i < n; i++) {
        h_C[i] = C[i];
    }

    free(bad);
    free(C);
    free(C_best);
    free(D, k);
    free(dist);
    free(L, k);
    free(l_j);
    free(L_sizes);
    free(M);
    free(M_best);
    free(M_current);
    free(M_random);
}