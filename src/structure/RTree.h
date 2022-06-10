#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <algorithm>

// https://www.cse.cuhk.edu.hk/~taoyf/course/infs4205/lec/rtree.pdf

class RTree_Node;

class RTree {
private:
    int dims;
    int B;
    RTree_Node *root;

    void insert(RTree_Node *u, std::pair<int, float *> *p);

    void insert(RTree_Node *E);

    RTree_Node *choose_subtree(RTree_Node *u, RTree_Node *E);

    RTree_Node *choose_subtree(RTree_Node *u, std::pair<int, float *> *p);

    void handle_overflow(RTree_Node *u);

    RTree_Node *split_leaf(RTree_Node *u);

    RTree_Node *split_internal(RTree_Node *u);

    float dist(float *p, float *q);

    void range(std::vector<int> &R, RTree_Node *u, float eps, float *p);

    RTree_Node *node_query(RTree_Node *u, std::pair<int, float *> *p);

    RTree_Node *node_query_check_all(RTree_Node *u, std::pair<int, float *> *p);

    void condense_tree(RTree_Node *L);

public:

    RTree(int dims, int B);

    void insert(std::pair<int, float *> *p);

    // https://books.google.dk/books?hl=en&lr=&id=i4ri1sWRPEcC&oi=fnd&pg=PR7&ots=3an4OtpG8o&sig=wMUzLDEQR2jEeLMOOw7tZOUEerE&redir_esc=y#v=onepage&q&f=falsehttps://books.google.dk/books?hl=en&lr=&id=i4ri1sWRPEcC&oi=fnd&pg=PR7&ots=3an4OtpG8o&sig=wMUzLDEQR2jEeLMOOw7tZOUEerE&redir_esc=y#v=onepage&q&f=false
    // R-Trees: Theory and Applications p. 11, 12,
    void remove(std::pair<int, float *> *p);

    std::vector<int> range(float eps, float *p);
};