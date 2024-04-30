#include "layers.h"

LinearLayer::LinearLayer(int NIn, int MIn): N(NIn), M(MIn) {
    weights = Matrix(N, M);
    bias = Matrix(1, M);
}

LinearLayer::LinearLayer(initializer_list<float> weightsIn, initializer_list<float> biasIn, int NIn, int MIn): N(NIn), M(MIn) {
    weights = Matrix(weightsIn, N, M);
    bias = Matrix(biasIn, 1, M);
}

Matrix LinearLayer::forward(Matrix& x) {
    Matrix result(0, x.N, this->M);
    try {
        if (x.M != this->N) {
            throw runtime_error(
                "LinearLayer with wrong dimensions");
        }

        
        result = x.dot(weights);
        result += bias;
    }
    catch (const exception& e) {
        // print the exception
        cout << "Exception " << e.what() << endl;
    }
    return result;
}