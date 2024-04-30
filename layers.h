#pragma once

#include "operations.h"


class LinearLayer
{
    int N;
    int M;
    Matrix weights;
    Matrix bias;

    public:
        LinearLayer(): N(0), M(0), weights(), bias() {};
        LinearLayer(int NIn, int MIn);
        LinearLayer(initializer_list<float> weightsIn, initializer_list<float> biasIn, int NIn, int MIn);
        Matrix forward(Matrix& x);

};