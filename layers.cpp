#include "layers.h"

LinearLayer::LinearLayer(int NIn, int MIn): N(NIn), M(MIn) {
    weights = Matrix2d(N, M);
    bias = Matrix2d(1, M);
}

LinearLayer::LinearLayer(initializer_list<float> weightsIn, initializer_list<float> biasIn, int NIn, int MIn): N(NIn), M(MIn) {
    weights = Matrix2d(weightsIn, N, M);
    // cout << "weight init \n";
    bias = Matrix2d(biasIn, 1, M);
    // cout << "bias init \n";
}

Matrix2d LinearLayer::forward(const Matrix2d& x) {
    input = x;
    Matrix2d result(0, x.N, this->M);
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

Matrix2d LinearLayer::backward(const Matrix2d& dy_dx)
{
    dweights = input.T().dot(dy_dx);
    dbias = Matrix2d(1, 1, input.N).dot(dy_dx);
    Matrix2d dy_dinput = dy_dx.dot(weights.T());
    return dy_dinput;
}


void LinearLayer::print_weights()
{
    weights.printMatrix();
}

void LinearLayer::print_bias()
{
    bias.printMatrix();
}

void LinearLayer::gradient_step(const float lr)
{
    weights = weights + dweights * lr;
    bias = bias + dbias * lr;
}


float Sigmoid::_func(const float& x)
{
    return 1.0 / (1 + exp(-x));
}

Matrix2d Sigmoid::forward(const Matrix2d& x)
{
    input = x;
    Matrix2d result(0, x.N, x.M);
    result = x.apply([this](float x) { return this->_func(x); });
    return result;
}

Matrix2d Sigmoid::backward(const Matrix2d& dy_dx)
{
    return dy_dx * input.apply([this](float x) { return (1 - this->_func(x)) * this->_func(x); });
}


float CrossEntropyLoss::_func(const float& pred, const int target)
{
    // target should be 0 or 1 !! TODO assert

    return - target * log(pred + 1e-6) - (1 - target) * log(1 - pred + 1e-6);
}

Matrix2d CrossEntropyLoss::forward(const Matrix2d& x, const Matrix2d& targetIn)
{
    input = x;
    target = targetIn;
    Matrix2d result(0, x.N, x.M);
    result = x.apply(targetIn, [this](float x, float y) { return this->_func(x, y); });
    return result;
}

Matrix2d CrossEntropyLoss::backward()
{
    return input.apply(target, [](float x, float y) { return (x - y) / (x * (1 - x) + 1e-6); });
}