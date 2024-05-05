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
class Matrix3dTransposedView;

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
            else if (this->M == other.M && this->N == 1)
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
        Matrix2d applyInPlace(const Matrix2d &other, Func func) const
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
            return result;
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
        void checkMatrixCompatibility(const Matrix2d& other, const std::string prefix = "Matrix2d shapes are not compatible ") const;

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
    friend class Matrix3dTransposedView;
    shared_ptr<float[]> values;
    public:
        // shape is accessible
        shared_ptr<int[]> shape; // store N x C x M
        shared_ptr<int[]> strides;
        int dim;
        // Default constructor
        Matrix3d() : values(nullptr), shape(nullptr), strides(nullptr), dim(0) {};
        // Copy semantic
        Matrix3d(Matrix3d& other) : values(other.values), shape(other.shape), strides(other.strides), dim(other.dim) {};
        Matrix3d copy() const
        {
            Matrix3d copy_matrix(shape);
            std::copy(&values[0], &values[0] + this->get_size(), copy_matrix.values.get());
            return copy_matrix;
        }
        // // Copy semantic for transposed
        // Matrix2d(const Matrix2dTransposedView& other);

        // utility
        int get_size() const
        {
            int size = 1;
            for (int i = 0; i < MatrixDim; ++i) {
                size *= shape[i];
            }
            return size;
        }
        // Matrix2d(float val, int NIn, int MIn);
        // template <int MatrixDim>
        Matrix3d(const float (&shapeIn)[MatrixDim]): dim(MatrixDim), shape(new int[MatrixDim], default_delete<int[]>()),
                                                    strides(new int[MatrixDim], default_delete<int[]>()),
                                                    values(nullptr, default_delete<float[]>())

        {
            copy(shapeIn, shapeIn + MatrixDim, shape.get());
            int size = this->get_size();
            values.reset(new float[size]);
            int initStride = 1;
            for (int i = size - 1; i > -1; --i)
            {
                strides[i] = initStride;
                initStride *= shape[i];
            }
            for (int i = 0; i < size; ++i)
            {
                values[i] = (float)((rand() - RAND_MAX / 2) / (float)RAND_MAX);
            }
        }
        // Matrix2d(initializer_list<float> list, int NIn, int MIn);
        // Matrix2d(vector<float> list, int NIn, int MIn);
        // Matrix2d(vector<vector<float>> list);
        // shared_ptr<float[]> const getValues();
        // // Matrix2d& operator=(Matrix2d&& other);
        // const float& operator()(int i, int j);
        // bool operator==(Matrix2d& other);
        // Matrix2d& operator+=(const Matrix2d& other);
        // Matrix2d operator+(const Matrix2d& other);
        // Matrix2d operator*(const Matrix2d& other) const;
        // Matrix2d operator*(const float mult) const;
        // Matrix2d operator-(const Matrix2d& other) const;
        // Matrix2d operator/(const Matrix2d& other) const;
        // float mean() const;
        // Matrix2dTransposedView T() const;
        // void printMatrix() const;

        // template <typename Func>
        // Matrix2d apply(Func func) const
        // {
        //     Matrix2d result(0, N, M);
        //     for (int i = 0; i < N * M; ++i)
        //     {
        //         result.values[i] = func(values[i]);
        //     }
        //     return result;
        // }

        // template <typename Func>
        // Matrix2d apply(const Matrix2d &other, Func func) const
        // {
        //     // same dimensions ?? check
        //     checkMatrixCompatibility(other, "Matrix2d apply with wrong dimensions ");
        //     Matrix2d result(0, N, M);
        //     if (this->M == other.M && this->N == other.N){
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             // matrix_i = i / M
        //             // matrix_j = i % M
        //             // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
        //             result.values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i / M * strideN + i % M * strideM]);
        //         }
        //     }
        //     else if (this->M == other.M && this->N == 1)
        //     {
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             result.values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i % M]);
        //         }
        //     }
        //     else if (1 == other.M && this->N == other.N)
        //     {
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             result.values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[i % N]);
        //         }
        //     }
        //     else // (1 == other.M && 1 == other.N)
        //     {
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             result.values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[0]);
        //         }
        //     }
        //     return result;
        // }

        // template <typename Func>
        // Matrix2d applyInPlace(const Matrix2d &other, Func func) const
        // {
        //     // same dimensions ?? check
        //     checkMatrixCompatibility(other, "Matrix2d apply with wrong dimensions ");
        //     Matrix2d result(0, N, M);
        //     if (this->M == other.M && this->N == other.N){
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             // matrix_i = i / M
        //             // matrix_j = i % M
        //             // assume matrix has equivalent strides as they have equal shape ?? TODO fix me
        //             values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i / M * strideN + i % M * strideM]);
        //         }
        //     }
        //     else if (this->M == other.M && this->N == 1)
        //     {
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             values[i / M * strideN + i % M * strideM] = func(values[i / M * strideN + i % M * strideM], other.values[i % M]);
        //         }
        //     }
        //     else if (1 == other.M && this->N == other.N)
        //     {
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[i % N]);
        //         }
        //     }
        //     else // (1 == other.M && 1 == other.N)
        //     {
        //         for (int i = 0; i < N * M; ++i)
        //         {
        //             values[i % N * strideN + i / N * strideM] = func(values[i % N * strideN + i / N * strideM], other.values[0]);
        //         }
        //     }
        //     return result;
        // }

        // template <typename MatrixType>
        // Matrix2d dot(const MatrixType &other) const
        // {

        //     Matrix2d result(0, this->N, other.M); // retrun default value anyway
        //     try {
        //         if (this->M != other.N) {
        //             throw runtime_error(
        //                 "Matrix2d multiplication with wrong dimensions");
        //         }

        //         // float result[this->N, other.M] = { 0 };
        //         for (int i = 0; i < result.N * result.M; ++i)
        //         {
        //             for (int k = 0; k < this->M; ++k)
        //             {
        //                 result.values[i / result.M * result.strideN + i % result.M * result.strideM] += (this->values[i / result.M * this->strideN + k * this->strideM] * other.values[k * other.strideN + i % result.M * other.strideM]);
        //             }
        //         }
        //     }
        //     catch (const exception& e) {
        //         // print the exception
        //         cout << "Exception " << e.what() << endl;
        //     }
        //     return result;
        // }
        // void checkMatrixCompatibility(const Matrix2d& other, const std::string prefix = "Matrix2d shapes are not compatible ") const;

        ~Matrix3d()
        {
            // cout << "Delete(" << N << ", " << M << ")" << "\n";
            // delete[] values;
        }
};

// class Matrix2dTransposedView
// {
//     friend class Matrix2d;
//     const shared_ptr<float[]> values;
//     public:
//         // shape is accessible
//         int N;
//         int strideN;
//         int M;
//         int strideM;
//         Matrix2dTransposedView(const Matrix2d &m) : values(m.values), N(m.M), strideN(1), M(m.N), strideM(m.M) {};
//         void printMatrix() const;
//         template <typename MatrixType>
//         Matrix2d dot(const MatrixType &other) const
//         {

//             Matrix2d result(0, this->N, other.M); // retrun default value anyway
//             try {
//                 if (this->M != other.N) {
//                     throw runtime_error(
//                         "Matrix2d multiplication with wrong dimensions");
//                 }

//                 // float result[this->N, other.M] = { 0 };
//                 for (int i = 0; i < result.N * result.M; ++i)
//                 {
//                     for (int k = 0; k < this->M; ++k)
//                     {
//                         result.values[i / result.M * result.strideN + i % result.M * result.strideM] += (this->values[i / result.M * this->strideN + k * this->strideM] * other.values[k * other.strideN + i % result.M * other.strideM]);
//                     }
//                 }
//             }
//             catch (const exception& e) {
//                 // print the exception
//                 cout << "Exception " << e.what() << endl;
//             }
//             return result;
//         }
// };