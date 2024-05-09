#pragma once

#include "operations.h"
#include <math.h>


template <int MatrixDim>
class LinearLayer
{
    int N;
    int M;
    // Input could be of different dimensions
    Matrix3d<MatrixDim> input;
    Matrix3d<2> weights;
    Matrix3d<2> dweights;
    Matrix3d<2> bias;
    Matrix3d<2> dbias;

    public:
        LinearLayer(): N(0), M(0), input(), weights(), bias(), dweights(), dbias() {};
        // LinearLayer(): N(0), M(0), weights(), bias() {};
        LinearLayer(int NIn, int MIn): N(NIn), M(MIn)
        {
            weights = Matrix3d<2>({N, M});
            bias = Matrix3d<2>({1, M});
        }
        LinearLayer(initializer_list<float> weightsIn, initializer_list<float> biasIn, int NIn, int MIn): N(NIn), M(MIn)
        {
            weights = Matrix3d<2>(weightsIn, {N, M});
            // cout << "weight init \n";
            bias = Matrix3d<2>(biasIn, {1, M});
            // cout << "bias init \n";
        }

        Matrix3d<2> forward(const Matrix3d<MatrixDim>& x)
        {
            input = x;
            int resultShape[MatrixDim];
            for (int i = 0; i < MatrixDim - 1; ++i) {
                resultShape[i] = x.shape[i];
            }
            resultShape[MatrixDim - 1] = this->M;
            Matrix3d<MatrixDim> result(0, resultShape);
            try {
                if (x.shape[MatrixDim - 1] != this->N) {
                    throw runtime_error(
                        "LinearLayer with wrong dimensions");
                }


                result = x.dot(weights);
                // result.printMatrix();
                result += bias;
            }
            catch (const exception& e) {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }
            return result;
        }
        template <int OtherDim>
        Matrix3d<MatrixDim> backward(const Matrix3d<OtherDim>& dy_dx)
        {
            dweights = input.T().dot(dy_dx);
            // cout << input.T().shape[0] << " " << input.T().shape[1]  << "\n";
            // cout << dy_dx.shape[0] << " " << dy_dx.shape[1]  << "\n";
            // cout << input.T().dot(dy_dx).shape[0] << " " << input.T().dot(dy_dx).shape[1]  << "\n";
            // dweights.printMatrix();
            dbias = Matrix3d<2>(1, {1, input.shape[0]}).dot(dy_dx);
            Matrix3d<MatrixDim> dy_dinput = dy_dx.dot(weights.T());
            return dy_dinput;
        }
        void print_weights() const
        {
            weights.printMatrix();
        }

        void print_bias() const
        {
            bias.printMatrix();
        }

        void gradient_step(const float lr)
        {
            weights = weights + dweights * lr;
            bias = bias + dbias * lr;
        }

};

template <int MatrixDim>
class Sigmoid
{
    friend class Matrix3d<MatrixDim>;
    Matrix3d<MatrixDim> input;

    public:
        Sigmoid(): input() {};
        float _func(const float& x)
        {
            return 1.0 / (1 + exp(-x));
        }

        Matrix3d<MatrixDim> forward(const Matrix3d<MatrixDim>& x)
        {
            input = x;
            // int resultShape[MatrixDim];
            // for (int i = 0; i < MatrixDim; ++i) {
            //     resultShape[i] = x.shape[i];
            // }
            // Matrix3d result(0, {resultShape});
            // Matrix3d<MatrixDim> result = x.apply([this](float x) { return this->_func(x); });
            return x.apply([this](float x) { return this->_func(x); });
        }

        Matrix3d<MatrixDim> backward(const Matrix3d<MatrixDim>& dy_dx)
        {
            return dy_dx * input.apply([this](float x) { return (1 - this->_func(x)) * this->_func(x); });
        }
};

template <int MatrixDim>
class CrossEntropyLoss
{
    friend class Matrix3d<MatrixDim>;
    Matrix3d<MatrixDim> input;
    Matrix3d<MatrixDim> target;

    public:
        CrossEntropyLoss(): input() {};
        float _func(const float& pred, const int target)
        {
            // target should be 0 or 1 !! TODO assert

            return - target * log(pred + 1e-6) - (1 - target) * log(1 - pred + 1e-6);
        }

        Matrix3d<MatrixDim> forward(const Matrix3d<MatrixDim>& x, const Matrix3d<MatrixDim>& targetIn)
        {
            input = x;
            target = targetIn;
            // Matrix2d result(0, x.N, x.M);
            // result = x.apply(targetIn, [this](float x, float y) { return this->_func(x, y); });
            // return result;
            return x.apply(targetIn, [this](float x, float y) { return this->_func(x, y); });
        }

        Matrix3d<MatrixDim> backward()
        {
            return input.apply(target, [](float x, float y) { return (x - y) / (x * (1 - x) + 1e-6); });
        }
};