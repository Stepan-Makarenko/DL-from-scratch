#include "operations.h"


void Matrix::checkMatrixCompatibility(const Matrix& other, const std::string prefix) const {
    // Case 1 equal size matrix N x M
    // Case 2 op( (N, M), (1, M) )
    // Case 3 op( (N, M), (N, 1) )
    // Case 4 op( (N, M), (1, 1) )

    if ( !( (this->N == other.N && this->M == other.M) || (this->N == other.N && 1 == other.M) || (1 == other.N && this->M == other.M) || (1 == other.N && 1 == other.M)) ) {
        std::stringstream ss;
        ss << prefix << " #1 shapes = (" << this->N << ", " << this->M << ")" << " 2# shapes = (" << other.N << ", " << other.M << ")\n";
        throw:: std::runtime_error(ss.str());
    }
}

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

Matrix::Matrix(const MatrixTransposedView& other)  : values(other.values), N(other.N), strideN(other.strideN), M(other.M), strideM(other.strideM)
{}

const float& Matrix::operator()(int i, int j)
{
    return values[i * strideN + j * strideM];
}

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
    checkMatrixCompatibility(other, "Matrix sum with wrong dimensions ");

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
    else if (this->M == other.M && this->N == 1)
    {
        for (int i = 0; i < N * M; ++i)
        {
            this->values[i / M * strideN + i % M * strideM] += other.values[i % M];
        }
    }
    else if (1 == other.M && this->N == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            this->values[i % N * strideN + i / N * strideM] += other.values[i % N];
        }
    }
    else // (1 == other.M && 1 == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            this->values[i % N * strideN + i / N * strideM] += other.values[0];
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix& other)
{
    checkMatrixCompatibility(other, "Matrix sum (+) with wrong dimensions ");
    Matrix result = this->copy();
    if (this->M == other.M && this->N == other.N){
        for (int i = 0; i < N * M; ++i)
        {
            // matrix_i = i / M
            // matrix_j = i % M
            // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
            result.values[i / M * strideN + i % M * strideM] += other.values[i / M * strideN + i % M * strideM];
        }
    }
    else if (this->M == other.M && this->N == 1)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i / M * strideN + i % M * strideM] += other.values[i % M];
        }
    }
    else if (1 == other.M && this->N == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] += other.values[i % N];
        }
    }
    else // (1 == other.M && 1 == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] += other.values[0];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    // assert(this->N == other.N && this->M == other.M, std::format("Different shape matrixes in * operator #1 shape = ({}, {}), #2 shape = ({}, {})", this->N, this->M, other.N,other.M));
    // assert(this->N == other.N && this->M == other.M, "Different shape matrixes in * operator");
    // assert(this->N == other.N && this->M == other.M);
    // cout << format("Different shape matrixes in * operator #1 shape = ({}, {}), #2 shape = ({}, {})", this->N, this->M, other.N,other.M);
    checkMatrixCompatibility(other, "Matrix prod (*) with wrong dimensions ");
    Matrix result = this->copy();
    if (this->M == other.M && this->N == other.N){
        for (int i = 0; i < N * M; ++i)
        {
            // matrix_i = i / M
            // matrix_j = i % M
            // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
            result.values[i / M * strideN + i % M * strideM] *= other.values[i / M * strideN + i % M * strideM];
        }
    }
    else if (this->M == other.M && this->N == 1)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i / M * strideN + i % M * strideM] *= other.values[i % M];
        }
    }
    else if (1 == other.M && this->N == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] *= other.values[i % N];
        }
    }
    else // (1 == other.M && 1 == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] *= other.values[0];
        }
    }
    return result;
}

Matrix Matrix::operator*(const float mult) const
{

    Matrix result(0, N, M);
    for (int i = 0; i < N * M; ++i)
    {
        result.values[i] = this->values[i] * mult;
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    checkMatrixCompatibility(other, "Matrix sub (-) with wrong dimensions ");
    Matrix result = this->copy();
    if (this->M == other.M && this->N == other.N){
        for (int i = 0; i < N * M; ++i)
        {
            // matrix_i = i / M
            // matrix_j = i % M
            // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
            result.values[i / M * strideN + i % M * strideM] -= other.values[i / M * strideN + i % M * strideM];
        }
    }
    else if (this->M == other.M && this->N == 1)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i / M * strideN + i % M * strideM] -= other.values[i % M];
        }
    }
    else if (1 == other.M && this->N == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] -= other.values[i % N];
        }
    }
    else // (1 == other.M && 1 == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] -= other.values[0];
        }
    }
    return result;
}

Matrix Matrix::operator/(const Matrix& other) const
{
    checkMatrixCompatibility(other, "Matrix divide (/) with wrong dimensions ");
    Matrix result = this->copy();
    if (this->M == other.M && this->N == other.N){
        for (int i = 0; i < N * M; ++i)
        {
            // matrix_i = i / M
            // matrix_j = i % M
            // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
            result.values[i / M * strideN + i % M * strideM] /= other.values[i / M * strideN + i % M * strideM] + 1e-6;
        }
    }
    else if (this->M == other.M && this->N == 1)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i / M * strideN + i % M * strideM] /= other.values[i % M] + 1e-6;
        }
    }
    else if (1 == other.M && this->N == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] /= other.values[i % N] + 1e-6;
        }
    }
    else // (1 == other.M && 1 == other.N)
    {
        for (int i = 0; i < N * M; ++i)
        {
            result.values[i % N * strideN + i / N * strideM] /= other.values[0] + 1e-6;
        }
    }
    return result;
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

float Matrix::mean() const
{
    float s = 0;
    for (int i = 0; i < N * M; ++i)
    {
        s += values[i] / ( N * M );
    }
    return s;
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