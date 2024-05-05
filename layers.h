#pragma once

#include "operations.h"
#include <math.h>


class LinearLayer
{
    int N;
    int M;

    Matrix2d input;
    Matrix2d weights;
    Matrix2d dweights;
    Matrix2d bias;
    Matrix2d dbias;

    public:
        LinearLayer(): N(0), M(0), input(), weights(), bias(), dweights(), dbias() {};
        // LinearLayer(): N(0), M(0), weights(), bias() {};
        LinearLayer(int NIn, int MIn);
        LinearLayer(initializer_list<float> weightsIn, initializer_list<float> biasIn, int NIn, int MIn);
        Matrix2d forward(const Matrix2d& x);
        Matrix2d backward(const Matrix2d& dy_dx);
        void gradient_step(const float lr);
        void print_weights();
        void print_bias();

};

class Sigmoid
{
    friend class Matrix2d;
    Matrix2d input;

    public:
        Sigmoid(): input() {};
        float _func(const float& x);
        Matrix2d forward(const Matrix2d& x);
        Matrix2d backward(const Matrix2d& dy_dx);
};

class CrossEntropyLoss
{
    friend class Matrix2d;
    Matrix2d input;
    Matrix2d target;

    public:
        CrossEntropyLoss(): input() {};
        float _func(const float& pred, const int targetIn);
        Matrix2d forward(const Matrix2d& x, const Matrix2d& targetIn);
        Matrix2d backward();
};