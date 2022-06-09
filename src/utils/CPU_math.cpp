//
// Created by mrjak on 14-07-2021.
//

#include "CPU_math.h"
#include "CPU_mem_util.h"

float distance(float *x, float *y, int d) {
    float dist = 0.;
    for (int l = 0; l < d; l++) {
        float diff = x[l] - y[l];
        dist += diff * diff;
    }
    return sqrt(dist);
}

float mean(float *x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum / n;
}

float *full(int n, float value) {
    float *temp = new float[n];
    for (int i = 0; i < n; i++) {
        temp[i] = value;
    }
    return temp;
}

int *full(int n, int value) {
    int *temp = new int[n];
    for (int i = 0; i < n; i++) {
        temp[i] = value;
    }
    return temp;
}

float *full(int n, std::function<int(float)> func) {
    float *temp = new float[n];
    for (int i = 0; i < n; i++) {
        temp[i] = func(i);
    }
    return temp;
}

float *clone(float *array, int n) {
    float *temp = new float[n];
    for (int i = 0; i < n; i++) {
        temp[i] = array[i];
    }
    return temp;
}

bool contains(std::vector<int> A, int i) {
    return std::count(A.begin(), A.end(), i);
}

void set_all(float *A, int n, float value) {
    for (int i = 0; i < n; i++) {
        A[i] = value;
    }
}

float *variance(float *D, int n, int d) {
    float *sigma = new float[d];
    for (int j = 0; j < d; j++) {
        float m = 0.;
        for (int i = 0; i < n; i++) {
            m += D[i * d + j];
        }
        m /= n;

        float var = 0;
        for (int i = 0; i < n; i++) {
            var += (D[i * d + j] - m) * (D[i * d + j] - m);
        }
        var /= n;
        sigma[j] = sqrt(var);
    }

    return sigma;
}

int middle(int start, int end) {
    int n = end - start;
    int m = n / 2;
    return start + m;
}

float *iqr(float *D, int n, int d) {

    float *r = new float[d];
    float *a = new float[n];

    for (int l = 0; l < d; l++) {
        for (int i = 0; i < n; i++) {
            a[i] = D[i * d + l];
        }

        std::sort(a, a + n);

        int m2 = middle(0, n);
        int m1 = middle(0, m2 + 1); //we never include end!
        int m3 = middle(m2 + 1, n);

        float Q1 = a[m1];
        float Q3 = a[m3];
        r[l] = Q3 - Q1;
    }

    return r;
}

float min(float a, float b) {
    if (a <= b)
        return a;
    return b;
}

int arg_min(const std::vector<float> &v) {
    if (v.size() == 0)
        return 0;

    int i = 0;
    int value = v[0];
    for (int j = 1; j < v.size(); j++) {
        if (v[j] < value) {
            i = j;
            value = v[j];
        }
    }
    return i;
}

float min(const std::vector<float> &v) {
    if (v.size() == 0)
        return 0.;

    int value = v[0];
    for (int j = 1; j < v.size(); j++) {
        if (v[j] < value) {
            value = v[j];
        }
    }
    return value;
}

int maximum(int *v, int n) {
    if (n == 0)
        return 0.;

    int value = v[0];
    for (int j = 1; j < n; j++) {
        if (v[j] > value) {
            value = v[j];
        }
    }
    return value;
}

void print_array(int *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");
}


void print_array(int *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%d ", x[i]);
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void print_array(bool *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i]) {
            printf("1 ");
        } else {
            printf("0 ");
        }
    }
    printf("\n");
}


void compute_l2_norm_to_medoid(float *dist, float *h_data, int *S, int m_i, int n, int d) {
    float *x_m_i = &h_data[m_i * d];
    for (int i = 0; i < n; i++) {
        float *x_S_i = &h_data[S[i] * d];
        dist[i] = 0;
        for (int j = 0; j < d; j++) {
            float x1 = x_S_i[j];
            float x2 = x_m_i[j];
            float sub = x1 - x2;
            dist[i] += sub * sub;
        }
        dist[i] = std::sqrt(dist[i]);
    }
}


void compute_l2_norm_to_medoid(float *dist, float *h_data, int m_i, int n, int d) {
    float *x_m_i = &h_data[m_i * d];
    for (int i = 0; i < n; i++) {

        float *x_i = &h_data[i * d];
        dist[i] = 0;
        for (int j = 0; j < d; j++) {
            float sub = x_i[j] - x_m_i[j];
            dist[i] += sub * sub;
        }
        dist[i] = std::sqrt(dist[i]);
    }
}


template<typename T>
int argmin_1d(T *values, int n) {
    int min_idx = -1;
    T min_value = std::numeric_limits<T>::max();
    for (int i = 0; i < n; i++) {
        if (values[i] < min_value) {
            min_value = values[i];
            min_idx = i;
        }
    }
    return min_idx;
}

template int argmin_1d<int>(int *values, int n);

template int argmin_1d<float>(float *values, int n);


int *random_sample(int *indices, int k, int n) {
    for (int i = 0; i < k; i++) {
        int j = std::rand() % n;
        int tmp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp_idx;
    }
    return indices;
}

int argmax_1d(float *values, int n) {
    int max_idx = -1;
    float max_value = -10000;//todo something smaller
    //printf("min: %f\n", max_value);
    for (int i = 0; i < n; i++) {
        if (values[i] >= max_value) {
            max_value = values[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void index_wise_minimum(float *values_1, float *values_2, int n) {
    for (int i = 0; i < n; i++) {
        values_1[i] = std::min(values_1[i], values_2[i]);
    }
}


float mean_1d(float *values, int n) {
    float sum = 0.;
    for (int i = 0; i < n; i++) {
        sum += values[i];
    }
    return sum / (float) n;
}

std::pair<int, int> *argmin_2d(float **values, int n, int m) {
    int min_x = -1;
    int min_y = -1;
    float min_value = std::numeric_limits<float>::max();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (values[i][j] < min_value) {
                min_value = values[i][j];
                min_x = i;
                min_y = j;
            }
        }
    }
    return new std::pair<int, int>(min_x, min_y);
}

float *compute_l1_norm_to_medoid(float *h_data, int m_i, bool *D_i, int n, int d) {
    float *dist = array_1d<float>(n);

    float *data_m_i = &h_data[m_i * d];

    for (int i = 0; i < n; i++) {
        dist[i] = 0;
        float *data_i = &h_data[i];
        for (int j = 0; j < d; j++) {
            if (D_i[j]) {
                dist[i] += std::abs(data_i[j] - data_m_i[j]);
            }
        }
    }

    return dist;
}