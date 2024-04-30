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
#include <initializer_list>

using namespace std; 

class Matrix
{
    float* values;
    public:
        // shape is accessible
        int N;
        int M;
        // Default constructor
        Matrix() : values(nullptr), N(0), M(0) {};
        // Move semantic
        Matrix(Matrix&& other);
        Matrix(float val, int NIn, int MIn);
        Matrix(int NIn, int MIn);
        Matrix(initializer_list<float> list, int NIn, int MIn);
        Matrix& operator=(Matrix&& other);
        bool operator==(Matrix& other);
        Matrix& operator+=(const Matrix& other);
        Matrix dot(Matrix &other);
        void printMatrix();
        ~Matrix()
        {
            // cout << "Delete(" << N << ", " << M << ")" << "\n";
            delete[] values;
        }
};
