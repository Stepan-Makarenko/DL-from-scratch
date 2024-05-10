/*
Discribe common matrix operations like dot, skalar multiplication and others
*/

#pragma once

// #include "catch_amalgamated.hpp"
#include <vector>
#include <iostream>
#include <time.h>
#include <cfloat>
#include <cstring>
#include <memory>
#include <initializer_list>
#include <cassert>
#include <format>
#include <stdexcept>
#include <sstream>

using namespace std;

class Matrix2dTransposedView;
template <int MatrixDim> class Matrix3dTransposedView;

class Matrix2d
{
    friend class Matrix2dTransposedView;
    shared_ptr<float[]> values;
    public:
        // shape is accessible
        int N;
        int strideN;
        int M;
        int strideM;
        // Default constructor
        Matrix2d() : values(nullptr), N(0), strideN(0), M(0), strideM(0) {};
        // Move semantic
        // Matrix2d(Matrix2d&& other);
        // Copy semantic
        Matrix2d(Matrix2d& other);
        Matrix2d copy() const;
        // Copy semantic for transposed
        Matrix2d(const Matrix2dTransposedView& other);

        Matrix2d(float val, int NIn, int MIn);
        Matrix2d(int NIn, int MIn);
        Matrix2d(initializer_list<float> list, int NIn, int MIn);
        Matrix2d(vector<float> list, int NIn, int MIn);
        Matrix2d(vector<vector<float>> list);
        shared_ptr<float[]> const getValues();
        // Matrix2d& operator=(Matrix2d&& other);
        const float& operator()(int i, int j);
        bool operator==(Matrix2d& other);
        Matrix2d& operator+=(const Matrix2d& other);
        Matrix2d operator+(const Matrix2d& other);
        Matrix2d operator*(const Matrix2d& other) const;
        Matrix2d operator*(const float mult) const;
        Matrix2d operator-(const Matrix2d& other) const;
        Matrix2d operator/(const Matrix2d& other) const;
        float mean() const;
        Matrix2dTransposedView T() const;
        void printMatrix() const;

        template <typename Func>
        Matrix2d apply(Func func) const
        {
            Matrix2d result(0, N, M);
            for (int i = 0; i < N * M; ++i)
            {
                result.values[i] = func(values[i]);
            }
            return result;
        }

        template <typename Func>
        Matrix2d apply(const Matrix2d &other, Func func) const
        {
            // same dimensions ?? check
            checkMatrixCompatibility(other, "Matrix2d apply with wrong dimensions ");
            Matrix2d result(0, N, M);
            if (this->M == other.M && this->N == other.N){
                for (int i = 0; i < N * M; ++i)
                {
                    // matrix_i = i / M
                    // matrix_j = i % M
                    // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
                    result.values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i / M * strideN + i % M * strideM]);
                }
            }
            else if (this->M == other.M && 1 == other.N)
            {
                for (int i = 0; i < N * M; ++i)
                {
                    result.values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i % M]);
                }
            }
            else if (1 == other.M && this->N == other.N)
            {
                for (int i = 0; i < N * M; ++i)
                {
                    result.values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[i % N]);
                }
            }
            else // (1 == other.M && 1 == other.N)
            {
                for (int i = 0; i < N * M; ++i)
                {
                    result.values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[0]);
                }
            }
            return result;
        }

        template <typename Func>
        void applyInPlace(const Matrix2d &other, Func func) const
        {
            // same dimensions ?? check
            checkMatrixCompatibility(other, "Matrix2d apply with wrong dimensions ");
            if (this->M == other.M && this->N == other.N){
                for (int i = 0; i < N * M; ++i)
                {
                    // matrix_i = i / M
                    // matrix_j = i % M
                    // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
                    values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i / M * strideN + i % M * strideM]);
                }
            }
            else if (this->M == other.M && this->N == 1)
            {
                for (int i = 0; i < N * M; ++i)
                {
                    values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i % M]);
                }
            }
            else if (1 == other.M && this->N == other.N)
            {
                for (int i = 0; i < N * M; ++i)
                {
                    values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[i % N]);
                }
            }
            else // (1 == other.M && 1 == other.N)
            {
                for (int i = 0; i < N * M; ++i)
                {
                    values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[0]);
                }
            }
        }

        template <typename MatrixType>
        Matrix2d dot(const MatrixType &other) const
        {

            Matrix2d result(0, this->N, other.M); // retrun default value anyway
            try {
                if (this->M != other.N) {
                    throw runtime_error(
                        "Matrix2d multiplication with wrong dimensions");
                }

                // float result[this->N, other.M] = { 0 };
                for (int i = 0; i < result.N * result.M; ++i)
                {
                    for (int k = 0; k < this->M; ++k)
                    {
                        result.values[i / result.M * result.strideN + i % result.M * result.strideM] += (this->values[i / result.M * this->strideN + k * this->strideM] * other.values[k * other.strideN + i % result.M * other.strideM]);
                    }
                }
            }
            catch (const exception& e) {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }
            return result;
        }
        void checkMatrixCompatibility(const Matrix2d& other, const std::string prefix = "Matrix2d shape are not compatible ") const;

        ~Matrix2d()
        {
            // cout << "Delete(" << N << ", " << M << ")" << "\n";
            // delete[] values;
        }
};

class Matrix2dTransposedView
{
    friend class Matrix2d;
    const shared_ptr<float[]> values;
    public:
        // shape is accessible
        int N;
        int strideN;
        int M;
        int strideM;
        Matrix2dTransposedView(const Matrix2d &m) : values(m.values), N(m.M), strideN(1), M(m.N), strideM(m.M) {};
        void printMatrix() const;
        template <typename MatrixType>
        Matrix2d dot(const MatrixType &other) const
        {

            Matrix2d result(0, this->N, other.M); // retrun default value anyway
            try {
                if (this->M != other.N) {
                    throw runtime_error(
                        "Matrix2d multiplication with wrong dimensions");
                }

                // float result[this->N, other.M] = { 0 };
                for (int i = 0; i < result.N * result.M; ++i)
                {
                    for (int k = 0; k < this->M; ++k)
                    {
                        result.values[i / result.M * result.strideN + i % result.M * result.strideM] += (this->values[i / result.M * this->strideN + k * this->strideM] * other.values[k * other.strideN + i % result.M * other.strideM]);
                    }
                }
            }
            catch (const exception& e) {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }
            return result;
        }
};

template <int MatrixDim>
class Matrix3d
{
    friend class Matrix3dTransposedView<MatrixDim>;
    shared_ptr<float[]> values;
    public:
        // shape is accessible
        shared_ptr<int[]> shape; // store N x C x M
        shared_ptr<int[]> strides;
        shared_ptr<int[]> stridesDenom; // denoms to get index from raw iterator i
        int dim;
        // Default constructor
        Matrix3d() : values(nullptr), shape(nullptr), strides(nullptr), stridesDenom(nullptr), dim(0) {};
        // Copy semantic
        Matrix3d(Matrix3d& other) : values(other.values), shape(other.shape), strides(other.strides), stridesDenom(other.stridesDenom), dim(other.dim) {};
        Matrix3d copy() const
        {
            Matrix3d copyMatrix(shape);
            std::copy(&values[0], &values[0] + this->getSize(), copyMatrix.values.get());
            return copyMatrix;
        }
        // // Copy semantic for transposed
        Matrix3d(const Matrix3dTransposedView<MatrixDim>& other) : values(other.values), shape(other.shape), strides(other.strides), stridesDenom(other.stridesDenom), dim(other.dim) {};

        // utility
        int getSize() const
        {
            int size = 1;
            for (int i = 0; i < MatrixDim; ++i) {
                size *= shape[i];
            }
            return size;
        }
        void initializeShapesAndAllocateMemory(const int (&shapeIn)[MatrixDim])
        {
            dim = MatrixDim;
            shape.reset(new int[MatrixDim], default_delete<int[]>());
            std::copy(shapeIn, shapeIn + MatrixDim, shape.get());
            strides.reset(new int[MatrixDim], default_delete<int[]>());
            stridesDenom.reset(new int[MatrixDim], default_delete<int[]>());
            values.reset(new float[getSize()], default_delete<float[]>());
            int initStride = 1;
            for (int i = MatrixDim - 1; i > -1; --i)
            {
                strides[i] = initStride;
                stridesDenom[i] = initStride;
                initStride *= shape[i];
            }
        }

        template <int OtherDim>
        void checkMatrixCompatibility(const Matrix3d<OtherDim>& other, const std::string prefix = "Matrix3d shape are not compatible ") const
        {
            static_assert(MatrixDim - OtherDim < 2, "Dimension mismatch for operation");
            // Case 1 equal size matrix
            // Case 2 op( (..., N, M), (..., 1, M) )
            // Case 3 op( (..., N, M), (..., N, 1) )
            // Case 4 op( (..., N, M), (..., 1, 1) )
            std::stringstream ss;
            ss << prefix << " #1 shape = (";
            for (int i = 0; i < MatrixDim; ++i) {
                ss << this->shape[i] << ", ";
            }
            ss << ")" << " 2# shape = (";
            for (int i = 0; i < OtherDim; ++i) {
                ss << other.shape[i] << ", ";
            }
            ss << ")\n";
            switch ( OtherDim )
            {
                case MatrixDim:
                    for (int i = 0; i < MatrixDim - 2; ++i)
                    {
                        if (this->shape[i] != other.shape[i])
                        {
                            throw:: std::runtime_error(ss.str());
                        }
                        if (this->strides[i] == other.strides[i])
                        {
                            throw:: std::runtime_error(ss.str());
                        }
                    }
                    // should we also check strides here?
                    if ( !( (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1])) ) {
                        throw:: std::runtime_error(ss.str());
                    }
                case MatrixDim - 1:
                    for (int i = 0; i < OtherDim - 2; ++i)
                    {
                        if (this->shape[1+i] != other.shape[i])
                        {
                            throw:: std::runtime_error(ss.str());
                        }
                        if (this->strides[1+i] == other.strides[i])
                        {
                            throw:: std::runtime_error(ss.str());
                        }
                    }
                    // should we also check strides here?
                    if ( !( (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1])) ) {
                        throw:: std::runtime_error(ss.str());
                    }
            }
        }
        void printMatrix() const
        {
            // for (int i = 0; i < getSize(); ++i)
            // {
            //     cout << values[i] << " ";
            //     for (int d = 0; d < MatrixDim-1; ++d)
            //     {
            //         if (i % strides[d] == strides[d] - 1)
            //         {
            //             cout << "\n";
            //         }
            //     }
            // }
            int printInd;
            bool newline = false;
            for (int i = 0; i < this->getSize(); ++i)
            {
                printInd = 0;
                for (int d = 0; d < dim; ++d)
                {
                    printInd += ((i / stridesDenom[d]) % shape[d]) * strides[d];
                }
                // cout << i << " " << printInd << " " << strides[1]  << " " << stridesDenom[1] << " " << shape[1] << "\n";
                cout << values[printInd] << " ";
                // cout << values[printInd] << " ";
                for (int d = 0; d < dim - 1; ++d)
                {
                    // newline |= i % stridesDenom[d] == stridesDenom[d] - 1;
                    if ((i / stridesDenom[d]) != ((i + 1) / stridesDenom[d]))
                    {
                        cout << "\n";
                    }
                }
            }
        }


        Matrix3d(const int (&shapeIn)[MatrixDim])
        {
            initializeShapesAndAllocateMemory(shapeIn);
            for (int i = 0; i < getSize(); ++i)
            {
                values[i] = (float)((rand() - RAND_MAX / 2) / (float)RAND_MAX);
            }
        }
        Matrix3d(float val, const int (&shapeIn)[MatrixDim])
        {
            initializeShapesAndAllocateMemory(shapeIn);
            for (int i = 0; i < getSize(); ++i)
            {
                values[i] = val;
            }
        }
        Matrix3d(initializer_list<float> list, const int (&shapeIn)[MatrixDim])
        {
            initializeShapesAndAllocateMemory(shapeIn);
            std::copy(list.begin(), list.end(), values.get());
        }
        Matrix3d(vector<float> list, const int (&shapeIn)[MatrixDim])
        {
            initializeShapesAndAllocateMemory(shapeIn);
            std::copy(list.begin(), list.end(), values.get());
        }

        explicit Matrix3d(vector<vector<float>> list)
        {
            static_assert(MatrixDim == 2, "Dimension mismatch for vector<vector<float>> initialization");
            int d1 = list.size();
            int d2 = list.empty() ? 0 : list[0].size();
            int shapeIn[2] = {d1, d2};
            initializeShapesAndAllocateMemory(shapeIn);
            for (int i = 0; i < d1; ++i) {
                for (int j = 0; j < d2; ++j) {
                    values[i * strides[0] + j * strides[1]] = list[i][j];
                }
            }

        }

        explicit Matrix3d(vector<vector<vector<float>>> list)
        {
            static_assert(MatrixDim == 3, "Dimension mismatch for vector<vector<vector<float>>> initialization");
            int d1 = list.size();
            int d2 = list.empty() ? 0 : list[0].size();
            int d3 = (d2 == 0 || list[0].empty()) ? 0 : list[0][0].size();
            int shapeIn[3] = {d1, d2, d3};
            initializeShapesAndAllocateMemory(shapeIn);
            for (int i = 0; i < d1; ++i) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        values[i * strides[0] + j * strides[1] + k * strides[2]] = list[i][j][k];
                    }
                }
            }
        }
        // shared_ptr<float[]> const getValues();
        // // Matrix2d& operator=(Matrix2d&& other);
        const float& operator()(int i, int j)
        {
            static_assert(MatrixDim == 2, "Undefind operator (i, j) for matrix of dim != 2");
            return values[i * strides[0] + j * strides[1]];
        }
        const float& operator()(int i, int j, int k)
        {
            static_assert(MatrixDim == 3, "Undefind operator (i, j, k) for matrix of dim != 3");
            return values[i * strides[0] + j * strides[1] + j * strides[2]];
        }
        bool operator==(Matrix3d& other)
        {
            // compare strides and shape
            for (int i = 0; i < MatrixDim; ++i)
            {
                if (shape[i] != other.shape[i])
                {
                    return false;
                }
                // We don't need strides equality
                // if (strides[i] != other.strides[i])
                // {
                //     return false;
                // }
            }

            int thisCompareInd = 0;
            int otherCompareInd = 0;
            for (int i = 0; i < this->getSize(); ++i) {
                // if (abs(values[i] - other.values[i]) > 0.0001) { // Float comparison?
                //     return false;
                // }
                thisCompareInd = 0;
                otherCompareInd = 0;
                for (int d = 0; d < dim; ++d)
                {
                    thisCompareInd += ((i / stridesDenom[d]) % shape[d]) * strides[d];
                    otherCompareInd += ((i / other.stridesDenom[d]) % other.shape[d]) * other.strides[d];
                }
                if (abs(values[thisCompareInd] - other.values[otherCompareInd]) > 0.0001) { // Float comparison?
                    cout << thisCompareInd << " " << otherCompareInd << "\n";
                    cout << values[thisCompareInd] << " " << other.values[otherCompareInd] << "\n";
                    return false;
                }
            }
            return true;
        }
        template <int OtherDim>
        Matrix3d& operator+=(const Matrix3d<OtherDim>& other)
        {
            checkMatrixCompatibility(other, "Matrix3d op (+) with wrong dimensions ");
            applyInPlace(other, [](float a, float b){ return a + b; });
            return *this;
        }
        template <int OtherDim>
        Matrix3d operator+(const Matrix3d<OtherDim>& other) const
        {
            checkMatrixCompatibility(other, "Matrix3d op (+) with wrong dimensions ");
            return apply(other, [](float a, float b){ return a + b; });
        }
        template <int OtherDim>
        Matrix3d operator*(const Matrix3d<OtherDim>& other) const
        {
            checkMatrixCompatibility(other, "Matrix3d op (*) with wrong dimensions ");
            return apply(other, [](float a, float b){ return a * b; });
        }
        Matrix3d operator*(const float mul) const
        {
            int resultShape[MatrixDim];
            for (int i = 0; i < MatrixDim; ++i)
            {
                resultShape[i] = shape[i];
            }
            Matrix3d result(resultShape);
            for (int i = 0; i < this->getSize(); i++)
            {
                result.values[i] = this->values[i] * mul;
            }
            return result;
        }
        template <int OtherDim>
        Matrix3d operator-(const Matrix3d<OtherDim>& other) const
        {
            checkMatrixCompatibility(other, "Matrix3d op (-) with wrong dimensions ");
            return apply(other, [](float a, float b){ return a - b; });
        }
        template <int OtherDim>
        Matrix3d operator/(const Matrix3d<OtherDim>& other) const
        {
            checkMatrixCompatibility(other, "Matrix3d op (/) with wrong dimensions ");
            return apply(other, [](float a, float b){ return a / (b + 1e-6); });
        }
        // float mean() const;
        Matrix3d sum(int dim) const
        {
            int resultShape[MatrixDim];
            for (int i = 0; i < MatrixDim; ++i)
            {
                resultShape[i] = shape[i];
                if (i == dim)
                {
                    resultShape[i] = 1;
                }
            }
            Matrix3d<MatrixDim> result(0, resultShape);
            for (int i = 0; i < this->getSize(); ++i)
            {
                int indThis = 0;
                int indResult = 0;
                for (int d = 0; d < MatrixDim; ++d)
                {
                    indThis += ((i / stridesDenom[d]) % shape[d]) * strides[d];
                    indResult += ((i / stridesDenom[d]) % result.shape[d]) * result.strides[d];
                }
                // cout << indThis << " " << indResult << "\n";
                result.values[indResult] += values[indThis];
            }
            return result;
        }

        Matrix3d<MatrixDim> softmax(int dim=MatrixDim-1) const
        {
            Matrix3d<MatrixDim> result = this->apply([](float x){ return exp(x); });
            Matrix3d<MatrixDim> denoms = result.sum(dim);
            // TODO should devide by column max
            for (int i = 0; i < result.getSize(); ++i)
            {
                int indResult = 0;
                int indDenom = 0;
                for (int d = 0; d < MatrixDim; ++d)
                {
                    indResult += ((i / result.stridesDenom[d]) % result.shape[d]) * result.strides[d];
                    // indOther += (((i / stridesDenom[d]) % shape[d]) * strides[d]) % other.shape[d];
                    indDenom += ((i / result.stridesDenom[d]) % denoms.shape[d]) * denoms.strides[d];
                }
                result.values[indResult] = result.values[indResult] / denoms.values[indDenom];
            }
            return result;
        }

        Matrix3dTransposedView<MatrixDim> T() const;

        template <typename Func>
        Matrix3d apply(Func func) const
        {
            int resultShape[MatrixDim];
            for (int i = 0; i < MatrixDim; ++i)
            {
                resultShape[i] = shape[i];
            }
            Matrix3d<MatrixDim> result(0, resultShape);
            for (int i = 0; i < getSize(); ++i)
            {
                result.values[i] = func(values[i]);
            }
            return result;
        }

        template <typename Func, int OtherDim>
        Matrix3d apply(const Matrix3d<OtherDim> &other, Func func) const
        {
            // same dimensions ?? check
            checkMatrixCompatibility(other, "Matrix2d apply with wrong dimensions ");
            int resultShape[MatrixDim];
            for (int i = 0; i < MatrixDim; ++i)
            {
                resultShape[i] = shape[i];
            }
            Matrix3d<MatrixDim> result(0, resultShape);
            switch ( OtherDim )
            {
                case MatrixDim:
                    for (int i = 0; i < this->getSize(); ++i)
                    {
                        int indThis = 0;
                        int indOther = 0;
                        for (int d = 0; d < MatrixDim; ++d)
                        {
                            indThis += ((i / stridesDenom[d]) % shape[d]) * strides[d];
                            // indOther += (((i / stridesDenom[d]) % shape[d]) * strides[d]) % other.shape[d];
                            indOther += ((i / stridesDenom[d]) % other.shape[d]) * other.strides[d];
                        }
                        result.values[indThis] = func(values[indThis], other.values[indOther]);
                    }
                    break;
                case MatrixDim - 1:
                    for (int i = 0; i < this->getSize(); ++i)
                    {
                        int indThis = ((i / stridesDenom[0]) % shape[0]) * strides[0];
                        int indOther = 0;
                        for (int d = 0; d < OtherDim; ++d)
                        {
                            indThis += ((i / stridesDenom[d+1]) % shape[d+1]) * strides[d+1];
                            // indOther += (((i / stridesDenom[d+1]) % shape[d+1]) * strides[d+1]) % other.shape[d];
                            indOther += ((i / stridesDenom[d+1]) % other.shape[d+1]) * other.strides[d+1];
                        }
                        result.values[indThis] = func(values[indThis], other.values[indOther]);
                    }
                    break;
            }

            return result;
        }

        template <typename Func, int OtherDim>
        void applyInPlace(const Matrix3d<OtherDim> &other, Func func) const
        {
            // same dimensions ?? check
            checkMatrixCompatibility(other, "Matrix2d apply with wrong dimensions ");
            switch ( OtherDim )
            {
                case MatrixDim:
                    for (int i = 0; i < this->getSize(); ++i)
                    {
                        int indThis = 0;
                        int indOther = 0;
                        for (int d = 0; d < MatrixDim; ++d)
                        {
                            indThis += ((i / stridesDenom[d]) % shape[d]) * strides[d];
                            // indOther += (((i / stridesDenom[d]) % shape[d]) * strides[d]) % other.shape[d];
                            indOther += ((i / stridesDenom[d]) % other.shape[d]) * other.strides[d];
                        }
                        // cout << "indThis = " << indThis << ", indOther= " << indOther << "\n";
                        // cout << values[indThis] << " + " << other.values[indOther];
                        values[indThis] = func(values[indThis], other.values[indOther]);
                        // cout << " = " << values[indThis] << "\n";
                    }
                    break;
                case MatrixDim - 1:
                    for (int i = 0; i < this->getSize(); ++i)
                    {
                        int indThis = ((i / stridesDenom[0]) % shape[0]) * strides[0];
                        int indOther = 0;
                        for (int d = 0; d < OtherDim; ++d)
                        {
                            indThis += ((i / stridesDenom[d+1]) % shape[d+1]) * strides[d+1];
                            // indOther += (((i / stridesDenom[d+1]) % shape[d+1]) * strides[d+1]) % other.shape[d];
                            indOther += ((i / stridesDenom[d+1]) % other.shape[d+1]) * other.strides[d+1];
                        }
                        values[indThis] = func(values[indThis], other.values[indOther]);
                    }
                    break;
            }
        }

        template <template<int> class MatrixType, int OtherDim>
        Matrix3d dot(const MatrixType<OtherDim>& other) const
        {
            // should check something befor TODO
            static_assert((MatrixDim == 2 || MatrixDim == 3) && OtherDim == 2 || (MatrixDim == 3 && OtherDim == 3), "Dimension mismatch for dot");
            int resultShape[MatrixDim];
            for (int i = 0; i < MatrixDim - 1; ++i)
            {
                resultShape[i] = shape[i];
            }
            resultShape[MatrixDim - 1] = other.shape[OtherDim - 1];

            Matrix3d result(0, resultShape); // retrun default value anyway
            try {
                if (this->shape[OtherDim - 1] != other.shape[OtherDim - 2]) {
                    throw runtime_error(
                        "Matrix2d multiplication with wrong dimensions");
                }

                // float result[this->N, other.M] = { 0 };
                // for (int i = 0; i < result.N * result.M; ++i)
                // {
                //     for (int k = 0; k < this->M; ++k)
                //     {
                //         result.values[i / result.M * result.strideN + i % result.M * result.strideM] += (this->values[i / result.M * this->strideN + k * this->strideM] * other.values[k * other.strideN + i % result.M * other.strideM]);
                //     }
                // }
                for (int i = 0; i < result.getSize(); ++i)
                {
                    int indResult = 0;
                    int indThis = 0;
                    int indOther = 0;
                    for (int d = 0; d < MatrixDim; ++d)
                    {
                        indResult += ((i / result.stridesDenom[d]) % result.shape[d]) * result.strides[d];
                        // consider dimensions < MatrixDim - 2 as common for all 3 matrix
                        if (d < MatrixDim - 2)
                        {
                            // indThis += ((i / this->stridesDenom[d]) % this->shape[d]) * this->strides[d];
                            indThis += ((i / result.stridesDenom[d]) % result.shape[d]) * this->strides[d];
                        }
                        if (d < OtherDim - 2)
                        {
                            // indOther += ((i / other.stridesDenom[d]) % other.shape[d]) * other.strides[d];
                            indOther += ((i / result.stridesDenom[d]) % result.shape[d]) * other.strides[d];
                        }
                    }
                    for (int k = 0; k < this->shape[MatrixDim - 1]; ++k)
                    {
                        // cout << indResult << " " << indThis + ((i / result.strides[MatrixDim-2]) % result.shape[MatrixDim-2]) * this->strides[MatrixDim-2] + k * this->strides[MatrixDim-1] << " " << indOther + k * other.strides[OtherDim - 2] + ((i / result.strides[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[MatrixDim - 1] << "\n";
                        result.values[indResult] += (this->values[indThis + ((i / result.stridesDenom[MatrixDim-2]) % result.shape[MatrixDim-2]) * this->strides[MatrixDim-2] + k * this->strides[MatrixDim-1]] * other.values[indOther + k * other.strides[OtherDim - 2] + ((i / result.stridesDenom[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[MatrixDim - 1]]);
                    }
                }
            }
            catch (const exception& e) {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }
            return result;
        }

        ~Matrix3d()
        {
            // cout << "Delete(" << N << ", " << M << ")" << "\n";
            // delete[] values;
        }
};


template <int MatrixDim>
class Matrix3dTransposedView
{
    friend class Matrix3d<MatrixDim>;
    const shared_ptr<float[]> values;
    public:
        // shape is accessible
        shared_ptr<int[]> shape; // store N x M x C (swap M and C as transpose)
        shared_ptr<int[]> strides;
        // strides and stridesDenom are equal for regular matrix!!!
        shared_ptr<int[]> stridesDenom;
        int dim;

        Matrix3dTransposedView(const Matrix3d<MatrixDim> &m) : values(m.values), dim(MatrixDim) {
            static_assert(MatrixDim > 1, "Dimension <2 in transpose operation");
            shape.reset(new int[dim], default_delete<int[]>());
            strides.reset(new int[dim], default_delete<int[]>());
            stridesDenom.reset(new int[dim], default_delete<int[]>());
            for (int i = 0; i < dim; ++i)
            {
                shape[i] = m.shape[i];
                strides[i] = m.strides[i];
                stridesDenom[i] = m.strides[i];
            }
            shape[dim-2] = m.shape[dim-1];
            shape[dim-1] = m.shape[dim-2];
            strides[dim-2] = m.strides[dim-1];
            strides[dim-1] = m.strides[dim-2];
            int currStrideDenom = 1;
            for (int i = dim-1; i > -1; --i)
            {
                stridesDenom[i] = currStrideDenom;
                currStrideDenom *= shape[i];
            }
        };
        int getSize() const
        {
            int size = 1;
            for (int i = 0; i < dim; ++i) {
                size *= shape[i];
            }
            return size;
        }
        void printMatrix() const
        {
            int printInd;
            bool newline = false;
            for (int i = 0; i < this->getSize(); ++i)
            {
                printInd = 0;
                newline = false;
                for (int d = 0; d < dim; ++d)
                {
                    printInd += ((i / stridesDenom[d]) % shape[d]) * strides[d];
                    if (d < dim - 1) {
                        newline |= i % stridesDenom[d] == stridesDenom[d] - 1;
                    }
                }
                // cout << i << " " << printInd << " " << strides[1]  << " " << stridesDenom[1] << " " << shape[1] << "\n";
                cout << values[printInd] << " ";
                // cout << values[printInd] << " ";
                if (newline)
                {
                    cout << "\n";
                }
            }
        }
        template <template<int> class MatrixType, int OtherDim>
        Matrix3d<MatrixDim> dot(const MatrixType<OtherDim>& other) const
        {
            // should check something befor TODO
            static_assert((MatrixDim == 2 || MatrixDim == 3) && OtherDim == 2, "Dimension mismatch for dot");
            int resultShape[MatrixDim];
            for (int i = 0; i < MatrixDim - 1; ++i)
            {
                resultShape[i] = shape[i];
            }
            resultShape[MatrixDim - 1] = other.shape[OtherDim - 1];

            Matrix3d result(0, resultShape); // retrun default value anyway
            // cout << "Inside tdot " << result.shape[0] << " " << result.shape[1]  << "\n";
            try {
                if (this->shape[OtherDim - 1] != other.shape[OtherDim - 2]) {
                    throw runtime_error(
                        "Matrix2d multiplication with wrong dimensions");
                }
                for (int i = 0; i < result.getSize(); ++i)
                {
                    int indResult = 0;
                    int indThis = 0;
                    int indOther = 0;
                    for (int d = 0; d < MatrixDim; ++d)
                    {
                        indResult += ((i / result.stridesDenom[d]) % result.shape[d]) * result.strides[d];
                        if (d < MatrixDim - 2)
                        {
                            indThis += ((i / this->stridesDenom[d]) % this->shape[d]) * this->strides[d];
                        }
                        if (d < OtherDim - 2)
                        {
                            indOther += ((i / other.stridesDenom[d]) % other.shape[d]) * other.strides[d];
                        }
                    }
                    for (int k = 0; k < this->shape[MatrixDim - 1]; ++k)
                    {
                        // cout << indResult << " " << indThis + ((i / result.strides[MatrixDim-2]) % result.shape[MatrixDim-2]) * this->strides[MatrixDim-2] + k * this->strides[MatrixDim-1] << " " << indOther + k * other.strides[OtherDim - 2] + ((i / result.strides[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[MatrixDim - 1] << "\n";
                        result.values[indResult] += (this->values[indThis + ((i / result.stridesDenom[MatrixDim-2]) % result.shape[MatrixDim-2]) * this->strides[MatrixDim-2] + k * this->strides[MatrixDim-1]] * other.values[indOther + k * other.strides[OtherDim - 2] + ((i / result.stridesDenom[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[MatrixDim - 1]]);
                    }
                }
            }
            catch (const exception& e) {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }
            return result;
        }
};

template <int MatrixDim>
Matrix3dTransposedView<MatrixDim> Matrix3d<MatrixDim>::T() const
{
    return Matrix3dTransposedView(*this);
}