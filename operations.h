/*
Discribe common matrix operations like dot, scalar multiplication and others
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

template <int MatrixDim>
class Matrix3d
{
    template <int FriendMatrixDim>
    friend class Matrix3d;
    public:
        // shape is accessible
        shared_ptr<float[]> values;
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

        // utility
        int getSize() const
        {
            int size = 1;
            for (int i = 0; i < MatrixDim; ++i) {
                size *= shape[i];
            }
            return size;
        }
        void initializeShapesAndAllocateMemory(const int (&shapeIn)[MatrixDim], bool resetValues=true)
        {
            dim = MatrixDim;
            shape.reset(new int[MatrixDim], default_delete<int[]>());
            std::copy(shapeIn, shapeIn + MatrixDim, shape.get());
            strides.reset(new int[MatrixDim], default_delete<int[]>());
            stridesDenom.reset(new int[MatrixDim], default_delete<int[]>());
            if (resetValues)
            {
                values.reset(new float[getSize()], default_delete<float[]>());
            }
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

                        // if (this->strides[i] == other.strides[i])
                        // {
                        //     throw:: std::runtime_error(ss.str());
                        // }
                    }
                    // should we also check strides here? No
                    if ( !( (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1])) ) {

                        throw:: std::runtime_error(ss.str());
                    }
                    break;
                case MatrixDim - 1:
                    for (int i = 0; i < OtherDim - 2; ++i)
                    {
                        if (this->shape[1+i] != other.shape[i])
                        {
                            throw:: std::runtime_error(ss.str());
                        }
                        // if (this->strides[1+i] == other.strides[i])
                        // {
                        //     throw:: std::runtime_error(ss.str());
                        // }
                    }
                    // should we also check strides here? No
                    if ( !( (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (this->shape[MatrixDim-2] == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && this->shape[MatrixDim-1] == other.shape[OtherDim-1]) || (1 == other.shape[OtherDim-2] && 1 == other.shape[OtherDim-1])) ) {
                        throw:: std::runtime_error(ss.str());
                    }
                    break;
            }
        }
        void printMatrix() const
        {
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

        Matrix3d(std::shared_ptr<float[]> values, const int (&shapeIn)[MatrixDim]): values(values)
        {
            initializeShapesAndAllocateMemory(shapeIn, false);
        }

        Matrix3d(std::shared_ptr<float[]> values, const int (&shapeIn)[MatrixDim], const int (&stridesIn)[MatrixDim],  const int (&stridesDenomIn)[MatrixDim]): values(values)
        {
            dim = MatrixDim;
            shape.reset(new int[MatrixDim], default_delete<int[]>());
            std::copy(shapeIn, shapeIn + MatrixDim, shape.get());
            strides.reset(new int[MatrixDim], default_delete<int[]>());
            std::copy(stridesIn, stridesIn + MatrixDim, strides.get());
            stridesDenom.reset(new int[MatrixDim], default_delete<int[]>());
            std::copy(stridesDenomIn, stridesDenomIn + MatrixDim, stridesDenom.get());
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
            return values[i * strides[0] + j * strides[1] + k * strides[2]];
        }
        bool operator==(Matrix3d& other)
        {
            double accuracy = 0.001;
            // compare shape
            for (int i = 0; i < MatrixDim; ++i)
            {
                if (shape[i] != other.shape[i])
                {
                    return false;
                }
            }

            int thisCompareInd = 0;
            int otherCompareInd = 0;
            for (int i = 0; i < this->getSize(); ++i) {
                thisCompareInd = 0;
                otherCompareInd = 0;
                for (int d = 0; d < dim; ++d)
                {
                    thisCompareInd += ((i / stridesDenom[d]) % shape[d]) * strides[d];
                    otherCompareInd += ((i / other.stridesDenom[d]) % other.shape[d]) * other.strides[d];
                }
                if (abs(values[thisCompareInd] - other.values[otherCompareInd]) > accuracy) {
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
            checkMatrixCompatibility(other, "Matrix3d op (+=) with wrong dimensions ");
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
            std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
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

        // add reduce option?
        Matrix3d sum(int dim) const
        {
            int resultShape[MatrixDim];
            std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
            resultShape[dim] = 1; // check that dim is proper value
            // for (int i = 0; i < MatrixDim; ++i)
            // {
            //     resultShape[i] = shape[i];
            //     if (i == dim)
            //     {
            //         resultShape[i] = 1;
            //     }
            // }
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

        Matrix3d<MatrixDim-1> squeeze(int dim) const
        {
            // we should create squeeze view without values copy
            if ( (dim < 0) || (dim > MatrixDim - 1)) {
                throw:: std::runtime_error("squeeze with inapropriate dim");
            }
            if ( shape[dim] != 1 ) {
                throw:: std::runtime_error("Do not know squeze operation for shape[dim] != 1");
            }

            int resultShape[MatrixDim - 1];
            int j = 0;
            for (int i = 0; i < MatrixDim; ++i) {
                if (i == dim) continue;
                resultShape[j] = shape[i];
                ++j;
            }
            Matrix3d<MatrixDim-1> result(this->values, resultShape);
            return result;
        }

        Matrix3d<MatrixDim+1> unsqueeze(int dim) const
        {
            if ( (dim < 0) || (dim > MatrixDim)) {
                throw:: std::runtime_error("unsqueeze with inapropriate dim");
            }

            int resultShape[MatrixDim + 1];
            int j = 0;
            for (int i = 0; i < MatrixDim + 1; ++i) {
                if (i == dim)
                {
                    resultShape[i] = 1;
                    continue;
                }
                resultShape[i] = shape[j];
                ++j;
            }
            Matrix3d<MatrixDim+1> result(this->values, resultShape);
            return result;
        }

        template <int OtherDim>
        Matrix3d<OtherDim> reshape(const int (&outShape)[OtherDim]) const
        {
            // do we also need to change strides and strides denom??
            int otherSize = 1;
            for (int i = 0; i < OtherDim; ++i) {
                otherSize *= outShape[i];
            }
            if ( this->getSize() !=  otherSize) {
                throw:: std::runtime_error("Do not know reshape operation this->size() != result->size()");
            }

            // create reshape view without value copying
            Matrix3d<OtherDim> result(this->values, outShape);
            return result;
        }

        Matrix3d<MatrixDim> swapAxes(int i, int j) const
        {
            static_assert(MatrixDim > 1, "Undefined swapAxis for matrix with MatrixDim < 2");

            // create swaped view without value copying
            int resultShape[MatrixDim];
            std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
            Matrix3d<MatrixDim> result(this->values, resultShape);
            result.shape[i] = this->shape[j];
            result.shape[j] = this->shape[i];
            result.strides[i] = this->strides[j];
            result.strides[j] = this->strides[i];
            int currStrideDenom = 1;
            for (int i = dim-1; i > -1; --i)
            {
                result.stridesDenom[i] = currStrideDenom;
                currStrideDenom *= result.shape[i];
            }
            return result;
        }

        // Matrix3d<MatrixDim> view(const int (&swapDest)[MatrixDim]) const // It's ore like torch.view
        // {
        //     // We need to use this function carefulli or make good error handling
        //     // swapDest is for destination axis for each original axis

        //     // create swaped view without value copying
        //     int resultShape[MatrixDim];
        //     std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
        //     Matrix3d<MatrixDim> result(this->values, resultShape);
        //     for (int i = 0; i < MatrixDim; ++i)
        //     {
        //         result.shape[i] = this->shape[swapDest[i]];
        //         result.strides[i] = this->strides[swapDest[i]];
        //     }
        //     int currStrideDenom = 1;
        //     for (int i = dim-1; i > -1; --i)
        //     {
        //         result.stridesDenom[i] = currStrideDenom;
        //         currStrideDenom *= result.shape[i];
        //     }
        //     return result;
        // }

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

        Matrix3d<MatrixDim> T() const {
            static_assert(MatrixDim > 1, "Dimension <2 in transpose operation");
            int resultShape[MatrixDim];
            std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
            Matrix3d<MatrixDim> TMatrix(this->values, resultShape);
            TMatrix.shape[dim-2] = shape[dim-1];
            TMatrix.shape[dim-1] = shape[dim-2];
            TMatrix.strides[dim-2] = strides[dim-1];
            TMatrix.strides[dim-1] = strides[dim-2];
            int currStrideDenom = 1;
            for (int i = dim-1; i > -1; --i)
            {
                TMatrix.stridesDenom[i] = currStrideDenom;
                // currStrideDenom *= shape[i];
                currStrideDenom *= TMatrix.shape[i];
            }
            return TMatrix;
        };

        template <typename Func>
        Matrix3d apply(Func func) const
        {
            int resultShape[MatrixDim];
            std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
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
            checkMatrixCompatibility(other, "Matrix3d apply with wrong dimensions ");
            int resultShape[MatrixDim];
            std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
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
        Matrix3d<max(MatrixDim, OtherDim)> dot(const MatrixType<OtherDim>& other) const
        {
            // should check something befor TODO
            // static_assert((MatrixDim == 2 || MatrixDim == 3) && OtherDim == 2 || (MatrixDim == 3 && OtherDim == 3), "Dimension mismatch for dot");
            const int resultDim = max(MatrixDim, OtherDim);
            int resultShape[resultDim];
            // broadcasting stage
            // TODO we should properly preserve strides and strides denom for each broadcasted matrix!!!
            int broadcastThisShape[resultDim];
            int broadcastOtherShape[resultDim];
            int broadcastThisStrides[resultDim];
            int broadcastOtherStrides[resultDim];
            int broadcastThisStridesDenom[resultDim];
            int broadcastOtherStridesDenom[resultDim];
            try {
                if (this->shape[MatrixDim - 1] != other.shape[OtherDim - 2]) {
                    throw runtime_error(
                        "Matrix2d multiplication with wrong dimensions");
                }
                resultShape[resultDim - 1] = other.shape[OtherDim - 1];
                resultShape[resultDim - 2] = this->shape[MatrixDim - 2];

                broadcastThisShape[resultDim - 1] = this->shape[MatrixDim - 1];
                broadcastThisShape[resultDim - 2] = this->shape[MatrixDim - 2];

                broadcastThisStrides[resultDim - 1] = this->strides[MatrixDim - 1];
                broadcastThisStrides[resultDim - 2] = this->strides[MatrixDim - 2];

                broadcastThisStridesDenom[resultDim - 1] = this->stridesDenom[MatrixDim - 1];
                broadcastThisStridesDenom[resultDim - 2] = this->stridesDenom[MatrixDim - 2];

                broadcastOtherShape[resultDim - 1] = other.shape[OtherDim - 1];
                broadcastOtherShape[resultDim - 2] = other.shape[OtherDim - 2];

                broadcastOtherStrides[resultDim - 1] = other.strides[OtherDim - 1];
                broadcastOtherStrides[resultDim - 2] = other.strides[OtherDim - 2];

                broadcastOtherStridesDenom[resultDim - 1] = other.stridesDenom[OtherDim - 1];
                broadcastOtherStridesDenom[resultDim - 2] = other.stridesDenom[OtherDim - 2];

                bool thisBigger = false;
                if (MatrixDim > OtherDim)
                {
                    thisBigger = true;
                }
                int i = MatrixDim - 3;
                int j = OtherDim - 3;
                int max_i_j;
                while (max(i, j) > - 1)
                {
                    max_i_j = max(i, j);
                    if (i < 0)
                    {
                        broadcastThisShape[j] = 1;
                        broadcastThisStrides[j] = broadcastThisStrides[j + 1];
                        broadcastThisStridesDenom[j] = broadcastThisStridesDenom[j + 1];
                    }
                    else
                    {
                        // cout << "i = " << i << "this->shape[i] = " <<this->shape[i] << endl;
                        broadcastThisShape[max_i_j] = this->shape[i];
                        broadcastThisStrides[max_i_j] = this->strides[i];
                        broadcastThisStridesDenom[max_i_j] = this->stridesDenom[i];
                    }
                    if (j < 0)
                    {
                        broadcastOtherShape[i] = 1;
                        broadcastOtherStrides[i] = broadcastOtherStrides[i + 1];
                        broadcastOtherStridesDenom[i] = broadcastOtherStridesDenom[i + 1];
                    }
                    else
                    {
                        broadcastOtherShape[max_i_j] = other.shape[j];
                        broadcastOtherStrides[max_i_j] = other.strides[j];
                        broadcastOtherStridesDenom[max_i_j] = other.stridesDenom[j];
                    }
                    if ((i > -1) && (j > -1) && (this->shape[i] != other.shape[j]) && (min(this->shape[i], other.shape[j]) != 1)) {
                        throw runtime_error(
                            "Matrix3d dot with wrong dimensions");
                    }
                    if (thisBigger)
                    {
                        resultShape[i] = max(this->shape[i], j > -1 ? other.shape[j]: 1);
                    }
                    else
                    {
                        resultShape[j] = max(other.shape[j], i > -1 ? this->shape[i]: 1);
                    }
                    --i;
                    --j;
                }
            }
            catch (const exception& e) {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }

            // cout << "Result broadcast shape = ";
            // for (auto el: resultShape)
            // {
            //     cout << el << " ";
            // }
            // cout << endl;
            // cout << "This shape = ";
            // for (int i = 0; i < MatrixDim; ++i)
            // {
            //     cout << shape[i] << " ";
            // }
            // cout << endl;
            // cout << "This broadcast shape = ";
            // for (auto el: broadcastThisShape)
            // {
            //     cout << el << " ";
            // }
            // cout << endl;
            // cout << "Other broadcast shape = ";
            // for (auto el: broadcastOtherShape)
            // {
            //     cout << el << " ";
            // }
            // cout << endl;

            Matrix3d<resultDim> broadcastedThis(this->values, broadcastThisShape, broadcastThisStrides, broadcastThisStridesDenom);
            Matrix3d<resultDim> broadcastedOther(other.values, broadcastOtherShape, broadcastOtherStrides, broadcastOtherStridesDenom);
            Matrix3d<resultDim> result(0, resultShape); // retrun default value anyway
            // Matrix3d result({1, 1}, {1, 2}); // retrun default value anyway

            try {
                for (int i = 0; i < result.getSize(); ++i)
                {
                    int indResult = 0;
                    int indThis = 0;
                    int indOther = 0;
                    for (int d = 0; d < resultDim; ++d)
                    {
                        indResult += ((i / result.stridesDenom[d]) % result.shape[d]) * result.strides[d];
                        // consider dimensions < MatrixDim - 2 as common for all 3 matrix
                        if (d < resultDim - 2)
                        {
                            indThis += ((i / result.stridesDenom[d]) % broadcastedThis.shape[d]) * broadcastedThis.strides[d];
                            indOther += ((i / result.stridesDenom[d]) % broadcastedOther.shape[d]) * broadcastedOther.strides[d];
                        }
                    }
                    for (int k = 0; k < broadcastedThis.shape[resultDim - 1]; ++k)
                    {
                        // cout << "--" << i << " " << indOther << " " << ((i / result.stridesDenom[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[MatrixDim - 1] << "\n";
                        // cout << indResult << " " << indThis + ((i / result.stridesDenom[resultDim-2]) % result.shape[resultDim-2]) * broadcastedThis.strides[resultDim-2] + k * broadcastedThis.strides[resultDim-1] << " " << indOther + k * broadcastedOther.strides[resultDim - 2] + ((i / result.stridesDenom[resultDim - 1]) % result.shape[resultDim - 1]) * broadcastedOther.strides[resultDim - 1] << "\n";
                        result.values[indResult] += (broadcastedThis.values[indThis + ((i / result.stridesDenom[resultDim-2]) % result.shape[resultDim-2]) * broadcastedThis.strides[resultDim-2] + k * broadcastedThis.strides[resultDim-1]] * broadcastedOther.values[indOther + k * broadcastedOther.strides[resultDim - 2] + ((i / result.stridesDenom[resultDim - 1]) % result.shape[resultDim - 1]) * broadcastedOther.strides[resultDim - 1]]);
                    }
                }
            }
            catch (const exception& e) {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }

            // std::copy(shape.get(), shape.get() + MatrixDim, resultShape);
            // resultShape[MatrixDim - 1] = other.shape[OtherDim - 1];

            // Matrix3d result(0, resultShape); // retrun default value anyway
            // try {
            //     if (this->shape[MatrixDim - 1] != other.shape[OtherDim - 2]) {
            //         throw runtime_error(
            //             "Matrix2d multiplication with wrong dimensions");
            //     }

            //     // float result[this->N, other.M] = { 0 };
            //     // for (int i = 0; i < result.N * result.M; ++i)
            //     // {
            //     //     for (int k = 0; k < this->M; ++k)
            //     //     {
            //     //         result.values[i / result.M * result.strideN + i % result.M * result.strideM] += (this->values[i / result.M * this->strideN + k * this->strideM] * other.values[k * other.strideN + i % result.M * other.strideM]);
            //     //     }
            //     // }
            //     for (int i = 0; i < result.getSize(); ++i)
            //     {
            //         int indResult = 0;
            //         int indThis = 0;
            //         int indOther = 0;
            //         for (int d = 0; d < MatrixDim; ++d)
            //         {
            //             indResult += ((i / result.stridesDenom[d]) % result.shape[d]) * result.strides[d];
            //             // consider dimensions < MatrixDim - 2 as common for all 3 matrix
            //             if (d < MatrixDim - 2)
            //             {
            //                 // indThis += ((i / this->stridesDenom[d]) % this->shape[d]) * this->strides[d];
            //                 indThis += ((i / result.stridesDenom[d]) % result.shape[d]) * this->strides[d];
            //             }
            //             if (d < OtherDim - 2)
            //             {
            //                 // indOther += ((i / other.stridesDenom[d]) % other.shape[d]) * other.strides[d];
            //                 indOther += ((i / result.stridesDenom[d]) % result.shape[d]) * other.strides[d];
            //             }
            //         }
            //         for (int k = 0; k < this->shape[MatrixDim - 1]; ++k)
            //         {
            //             // cout << "--" << i << " " << indOther << " " << ((i / result.stridesDenom[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[MatrixDim - 1] << "\n";
            //             // cout << indResult << " " << indThis + ((i / result.strides[MatrixDim-2]) % result.shape[MatrixDim-2]) * this->strides[MatrixDim-2] + k * this->strides[MatrixDim-1] << " " << indOther + k * other.strides[OtherDim - 2] + ((i / result.strides[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[MatrixDim - 1] << "\n";
            //             result.values[indResult] += (this->values[indThis + ((i / result.stridesDenom[MatrixDim-2]) % result.shape[MatrixDim-2]) * this->strides[MatrixDim-2] + k * this->strides[MatrixDim-1]] * other.values[indOther + k * other.strides[OtherDim - 2] + ((i / result.stridesDenom[MatrixDim - 1]) % result.shape[MatrixDim - 1]) * other.strides[OtherDim - 1]]);
            //         }
            //     }
            // }
            // catch (const exception& e) {
            //     // print the exception
            //     cout << "Exception " << e.what() << endl;
            // }
            return result;
        }

        ~Matrix3d()
        {
            // cout << "Delete(" << N << ", " << M << ")" << "\n";
            // delete[] values;
        }
};