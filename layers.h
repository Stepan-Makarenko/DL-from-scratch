#pragma once

#include "operations.h"
#include <math.h>


class LinearLayer
{
    int N;
    int M;

    Matrix input;
    Matrix weights;
    Matrix dweights;
    Matrix bias;
    Matrix dbias;

    public:
        LinearLayer(): N(0), M(0), input(), weights(), bias(), dweights(), dbias() {};
        // LinearLayer(): N(0), M(0), weights(), bias() {};
        LinearLayer(int NIn, int MIn);
        LinearLayer(initializer_list<float> weightsIn, initializer_list<float> biasIn, int NIn, int MIn);
        Matrix forward(const Matrix& x);
        Matrix backward(const Matrix& dy_dx);

};

class Sigmoid
{
    friend class Matrix;
    Matrix input;

    public:
        Sigmoid(): input() {};
        float _func(const float& x);
        Matrix forward(const Matrix& x);
        // Matrix backward(const Matrix& dy_dx);
};