#include "layers.h"

// float Sigmoid::_func(const float& x)
// {
//     return 1.0 / (1 + exp(-x));
// }

// Matrix2d Sigmoid::forward(const Matrix2d& x)
// {
//     input = x;
//     Matrix2d result(0, x.N, x.M);
//     result = x.apply([this](float x) { return this->_func(x); });
//     return result;
// }

// Matrix2d Sigmoid::backward(const Matrix2d& dy_dx)
// {
//     return dy_dx * input.apply([this](float x) { return (1 - this->_func(x)) * this->_func(x); });
// }


// float CrossEntropyLoss::_func(const float& pred, const int target)
// {
//     // target should be 0 or 1 !! TODO assert

//     return - target * log(pred + 1e-6) - (1 - target) * log(1 - pred + 1e-6);
// }

// Matrix2d CrossEntropyLoss::forward(const Matrix2d& x, const Matrix2d& targetIn)
// {
//     input = x;
//     target = targetIn;
//     Matrix2d result(0, x.N, x.M);
//     result = x.apply(targetIn, [this](float x, float y) { return this->_func(x, y); });
//     return result;
// }

// Matrix2d CrossEntropyLoss::backward()
// {
//     return input.apply(target, [](float x, float y) { return (x - y) / (x * (1 - x) + 1e-6); });
// }