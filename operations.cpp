#include "operations.h"


Matrix::Matrix(float val, int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1) {
    fill_n(values.get(), N * M, val);
}

// assign refference to existing resource
Matrix::Matrix(Matrix& other) : values(other.values), N(other.N), strideN(other.M), M(other.M), strideM(1)  {}
// deep copy
Matrix Matrix::copy() const
{
    Matrix copy_matrix(N, M);
    std::copy(&values[0], &values[0] + N * M, copy_matrix.values.get());
    return copy_matrix;
}

// Matrix::Matrix(Matrix&& other) : values(other.values), N(other.N), strideN(other.M), M(other.M), strideM(1)  {
//     if (this != &other){
//         other.values = nullptr;
//         other.N = 0;
//         other.M = 0;
//         other.strideN = 0;
//         other.strideM = 0;
//     }
// }


Matrix::Matrix(const MatrixTransposedView& other)  : values(other.values), N(other.N), strideN(other.strideN), M(other.M), strideM(other.strideM)
{}


// Matrix& Matrix::operator=(Matrix&& other)
// {
//     if (this == &other)
//         return *this;

//     delete[] values;
//     values = other.values;
//     N = other.N;
//     strideN = other.strideN;
//     M = other.M;
//     strideM = other.strideM;

//     other.values = nullptr;
//     other.N = 0;
//     other.strideN = 0;
//     other.M = 0;
//     other.strideM = 0;
//     return *this;
// }

bool Matrix::operator==(Matrix& other)
{
    if (N != other.N || M != other.M || strideN != other.strideN || strideM != other.strideM) {
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
                // matrix_i = i / M
                // matrix_j = i % M
                // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
                this->values[i / M * strideN + i % M * strideM] += other.values[i / M * strideN + i % M * strideM];
            }
        }
        else if (other.N == 1)
        {
            for (int i = 0; i < N * M; ++i)
            {
                this->values[i / M * strideN + i % M * strideM] += other.values[i % M];
            }
        }
        else if (other.M == 1)
        {
            for (int i = 0; i < N * M; ++i)
            {
                this->values[i / M * strideN + i % M * strideM] += other.values[i % M];
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

Matrix::Matrix(initializer_list<float> list, int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1)
{
    // values = new float[N * M];
    std::copy(list.begin(), list.end(), values.get());
}

Matrix::Matrix(vector<float> list, int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1)
{
    std::copy(list.begin(), list.end(), values.get());
}

Matrix::Matrix(vector<vector<float>> list): N(list.size())
{
    if (N == 0)
    {
        values = nullptr;
        return;
    }

    M = list.begin()->size();
    strideN = M;
    strideM = 1;
    values = shared_ptr<float[]>(new float[N*M], default_delete<float[]>());

    int row_idx = 0;
    for (auto row: list)
    {
        std::copy(row.begin(), row.end(), &values[row_idx * M]);
        ++row_idx;
    }
}

Matrix::Matrix(int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1)
{
    for (int i = 0; i < N * M; ++i)
    {
        values[i / M * strideN + i % M * strideM] = (float)((rand() - RAND_MAX / 2) / (float)RAND_MAX);
    }
}

shared_ptr<float[]> const Matrix::getValues()
{
    return values;
}

MatrixTransposedView Matrix::T() const
{
    return MatrixTransposedView(*this);
}

void Matrix::printMatrix() const
{
    // cout << N << " " << M << "\n";
    for (int i = 0; i < N * M; ++i)
    {
        cout << values[i / M * strideN + i % M * strideM] << " ";
        if (i % M == M - 1)
        {
            cout << "\n";
        }
    }
}

void MatrixTransposedView::printMatrix() const
{
    // cout << N << " " << M << "\n";
    for (int i = 0; i < N * M; ++i)
    {
        // cout << this->matrix.values[i / M * strideN + i % M * strideM] << " ";
        cout << values[i / M * strideN + i % M * strideM] << " ";
        if (i % M == M - 1)
        {
            cout << "\n";
        }
    }
}