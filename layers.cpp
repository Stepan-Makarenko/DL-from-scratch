#include "layers.h"

LinearLayer::LinearLayer(int NIn, int MIn): N(NIn), M(MIn) {
    weights = Matrix(N, M);
    bias = Matrix(1, M);
}

LinearLayer::LinearLayer(initializer_list<float> weightsIn, initializer_list<float> biasIn, int NIn, int MIn): N(NIn), M(MIn) {
    weights = Matrix(weightsIn, N, M);
    // cout << "weight init \n";
    bias = Matrix(biasIn, 1, M);
    // cout << "bias init \n";
}

Matrix LinearLayer::forward(const Matrix& x) {
    input = x;
    Matrix result(0, x.N, this->M);
    try {
        if (x.M != this->N) {
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

Matrix LinearLayer::backward(const Matrix& dy_dx)
{
    dweights = input.T().dot(dy_dx);
    dbias = Matrix(1, 1, input.N).dot(dy_dx);
    Matrix dy_dinput = dy_dx.dot(weights.T());
    return dy_dinput;
}


float Sigmoid::_func(const float& x)
{
    return 1.0 / (1 - exp(-x));
}

Matrix Sigmoid::forward(const Matrix& x)
{
    Matrix result(0, x.N, x.M);
    result.apply([this](float x) { return this->_func(x); });
    return result;
}

// Matrix Sigmoid::backward(const Matrix& dy_dx)
// {
//     return dy_dx; // FIXME
// }