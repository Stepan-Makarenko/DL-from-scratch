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

class MatrixTransposedView;

class Matrix
{
    friend class MatrixTransposedView;
    shared_ptr<float[]> values;
    public:
        // shape is accessible
        int N;
        int strideN;
        int M;
        int strideM;
        // Default constructor
        Matrix() : values(nullptr), N(0), M(0) {};
        // Move semantic
        // Matrix(Matrix&& other);
        // Copy semantic
        Matrix(Matrix& other);
        Matrix copy() const;
        // Copy semantic for transposed
        Matrix(const MatrixTransposedView& other);

        Matrix(float val, int NIn, int MIn);
        Matrix(int NIn, int MIn);
        Matrix(initializer_list<float> list, int NIn, int MIn);
        Matrix(vector<float> list, int NIn, int MIn);
        Matrix(vector<vector<float>> list);
        shared_ptr<float[]> const getValues();
        // Matrix& operator=(Matrix&& other);
        const float& operator()(int i, int j);
        bool operator==(Matrix& other);
        Matrix& operator+=(const Matrix& other);
        Matrix operator+(const Matrix& other);
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(const float mult) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator/(const Matrix& other) const;
        float mean() const;
        MatrixTransposedView T() const;
        void printMatrix() const;

        template <typename Func>
        Matrix apply(Func func) const
        {
            Matrix result(0, N, M);
            for (int i = 0; i < N * M; ++i)
            {
                result.values[i] = func(values[i]);
            }
            return result;
        }

        template <typename Func>
        Matrix apply(const Matrix &other, Func func) const
        {
            // same dimensions ?? check
            Matrix result(0, N, M);
            for (int i = 0; i < N * M; ++i)
            {
                result.values[i] = func(values[i], other.values[i]);
            }
            return result;
        }

        template <typename MatrixType>
        Matrix dot(const MatrixType &other) const
        {

            Matrix result(0, this->N, other.M); // retrun default value anyway
            try {
                if (this->M != other.N) {
                    throw runtime_error(
                        "Matrix multiplication with wrong dimensions");
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
        void checkMatrixCompatibility(const Matrix& other, const std::string prefix = "Matrix shapes are not compatible ") const;

        ~Matrix()
        {
            // cout << "Delete(" << N << ", " << M << ")" << "\n";
            // delete[] values;
        }
};

class MatrixTransposedView
{
    friend class Matrix;
    const shared_ptr<float[]> values;
    public:
        // shape is accessible
        int N;
        int strideN;
        int M;
        int strideM;
        MatrixTransposedView(const Matrix &m) : values(m.values), N(m.M), strideN(1), M(m.N), strideM(m.M) {};
        void printMatrix() const;
        template <typename MatrixType>
        Matrix dot(const MatrixType &other) const
        {

            Matrix result(0, this->N, other.M); // retrun default value anyway
            try {
                if (this->M != other.N) {
                    throw runtime_error(
                        "Matrix multiplication with wrong dimensions");
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