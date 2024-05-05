#include "operations.h"


void Matrix2d::checkMatrixCompatibility(const Matrix2d& other, const std::string prefix) const {
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

Matrix2d::Matrix2d(float val, int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1) {
    fill_n(values.get(), N * M, val);
}

// assign refference to existing resource
Matrix2d::Matrix2d(Matrix2d& other) : values(other.values), N(other.N), strideN(other.M), M(other.M), strideM(1)  {}
// deep copy
Matrix2d Matrix2d::copy() const
{
    Matrix2d copy_matrix(N, M);
    std::copy(&values[0], &values[0] + N * M, copy_matrix.values.get());
    return copy_matrix;
}

Matrix2d::Matrix2d(const Matrix2dTransposedView& other)  : values(other.values), N(other.N), strideN(other.strideN), M(other.M), strideM(other.strideM)
{}

const float& Matrix2d::operator()(int i, int j)
{
    return values[i * strideN + j * strideM];
}

bool Matrix2d::operator==(Matrix2d& other)
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

Matrix2d& Matrix2d::operator+=(const Matrix2d& other)
{
    checkMatrixCompatibility(other, "Matrix2d sum with wrong dimensions ");

    // float result[this->N, other.M] = { 0 };
    applyInPlace(other, [](float a, float b) { return a + b; });
    return *this;
}

Matrix2d Matrix2d::operator+(const Matrix2d& other)
{
    checkMatrixCompatibility(other, "Matrix2d sum (+) with wrong dimensions ");
    Matrix2d result = this->apply(other, [](float a, float b) { return a + b; });
    return result;
}

Matrix2d Matrix2d::operator*(const Matrix2d& other) const
{
    // assert(this->N == other.N && this->M == other.M, std::format("Different shape matrixes in * operator #1 shape = ({}, {}), #2 shape = ({}, {})", this->N, this->M, other.N,other.M));
    // assert(this->N == other.N && this->M == other.M, "Different shape matrixes in * operator");
    // assert(this->N == other.N && this->M == other.M);
    // cout << format("Different shape matrixes in * operator #1 shape = ({}, {}), #2 shape = ({}, {})", this->N, this->M, other.N,other.M);
    checkMatrixCompatibility(other, "Matrix2d prod (*) with wrong dimensions ");
    Matrix2d result = this->apply(other, [](float a, float b) { return a * b; });
    return result;
}

Matrix2d Matrix2d::operator*(const float mult) const
{

    Matrix2d result(0, N, M);
    for (int i = 0; i < N * M; ++i)
    {
        result.values[i] = this->values[i] * mult;
    }
    return result;
}

Matrix2d Matrix2d::operator-(const Matrix2d& other) const
{
    checkMatrixCompatibility(other, "Matrix2d sub (-) with wrong dimensions ");
    Matrix2d result = this->apply(other, [](float a, float b) { return a - b; });
    return result;
}

Matrix2d Matrix2d::operator/(const Matrix2d& other) const
{
    checkMatrixCompatibility(other, "Matrix2d divide (/) with wrong dimensions ");
    Matrix2d result = this->apply(other, [](float a, float b) { return a / (b + 1e-6); });
    return result;
}


Matrix2d::Matrix2d(initializer_list<float> list, int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1)
{
    // values = new float[N * M];
    std::copy(list.begin(), list.end(), values.get());
}

Matrix2d::Matrix2d(vector<float> list, int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1)
{
    std::copy(list.begin(), list.end(), values.get());
}

Matrix2d::Matrix2d(vector<vector<float>> list): N(list.size())
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

Matrix2d::Matrix2d(int NIn, int MIn) : values(new float[NIn*MIn], default_delete<float[]>()), N(NIn), strideN(MIn), M(MIn), strideM(1)
{
    for (int i = 0; i < N * M; ++i)
    {
        values[i / M * strideN + i % M * strideM] = (float)((rand() - RAND_MAX / 2) / (float)RAND_MAX);
    }
}

shared_ptr<float[]> const Matrix2d::getValues()
{
    return values;
}

float Matrix2d::mean() const
{
    float s = 0;
    for (int i = 0; i < N * M; ++i)
    {
        s += values[i] / ( N * M );
    }
    return s;
}

Matrix2dTransposedView Matrix2d::T() const
{
    return Matrix2dTransposedView(*this);
}

void Matrix2d::printMatrix() const
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

void Matrix2dTransposedView::printMatrix() const
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