#include "operations.h"


Matrix::Matrix(float val, int NIn, int MIn) : N(NIn), M(MIn) {
    values = new float[N*M];
    for (int i = 0; i < N * M; ++i){
        values[i] = val;
    }
}

Matrix::Matrix(Matrix&& other) : values(other.values), N(other.N), M(other.M)  {
    if (this != &other){
        other.values = nullptr;
        other.N = 0;
        other.M = 0;
    }
}

Matrix& Matrix::operator=(Matrix&& other) 
{
    if (this == &other)
        return *this;

    delete[] values;
    values = other.values;
    N = other.N;
    M = other.M;

    other.values = nullptr;
    other.N = 0;
    other.M = 0;
    return *this;
}

bool Matrix::operator==(Matrix& other) 
{
    if (N != other.N || M != other.M) {
        return false;
    }
    for (int i = 0; i < N * M; ++i) {
        if (abs(values[i] - other.values[i]) > 0.0001) { // Float comparison?
            return false;
        }
    }
    return true;
}

Matrix& Matrix::operator+=(const Matrix& other) 
{
    try {
        if (this->M != other.M && this->N != other.N && (other.M != 1 && other.N != 1)) {
            throw runtime_error(
                "Matrix sum with wrong dimensions");
        }
        std::cout << "Get (" << other.N << ", " << other.M << ")" << endl;

        // float result[this->N, other.M] = { 0 };
        if (this->M == other.M && this->N == other.N){
            for (int i = 0; i < N * M; ++i)
            {
                *(this->values + i / M * M + i % M) += *(other.values + i / M * M + i % M);
            }
        }
        else if (other.N == 1)
        {
            for (int i = 0; i < N * M; ++i)
            {
                *(this->values + i / M * M + i % M) += *(other.values + i % M);
            }
        }
        else if (other.M == 1)
        {
            for (int i = 0; i < N * M; ++i)
            {
                *(this->values + i / M + i % M * M) += *(other.values + i % M);
            }
        }
        // is else occure? 
        
        
    }
    catch (const exception& e) {
        // print the exception
        cout << "Exception " << e.what() << endl;
    }
    return *this;
}

Matrix::Matrix(initializer_list<float> list, int NIn, int MIn) : N(NIn), M(MIn)
{
    values = new float[N * M];
    memcpy(values, list.begin(), N * M * sizeof(float));
}

Matrix::Matrix(int NIn, int MIn) : N(NIn), M(MIn)
{
    values = new float[N*M];
    for (int i = 0; i < N * M; ++i)
    {
        *(values + i / M * M + i % M) = (float)((rand() - RAND_MAX / 2) / (float)RAND_MAX);
    }
}

Matrix Matrix::dot(Matrix &other)
{
    
    Matrix result(0, this->N, other.M); // retrun default value anyway
    try {
        if (this->M != other.N) {
            throw runtime_error(
                "Matrix multiplication with wrong dimensions");
        }

        // float result[this->N, other.M] = { 0 };
        for (int i = 0; i < this->N; ++i)
        {
            for (int j = 0; j < other.M; ++j) 
            {
                for (int k = 0; k < this->M; ++k) 
                {
                    result.values[i * other.M + j] += (this->values[i * this->M + k] * other.values[k * other.M + j]);
                    // cout << i * other.M + j << " " << i * this->M + k << " " << k * other.M + j << "\n";
                }
            }
        }
    }
    catch (const exception& e) {
        // print the exception
        cout << "Exception " << e.what() << endl;
    }
    return result;
}

void Matrix::printMatrix()
{
    // cout << N << " " << M << "\n";
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < M; ++j)
        {
            cout << values[i * M + j] << " ";
        }
        cout << "\n";
    }
}