/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_CATLASS_HPP
#define MATMUL_EMU_CATLASS_CATLASS_HPP

#include <cstdint>
#include <cstddef>
#include <type_traits>

#if defined(__CCE__)
#include <kernel_operator.h>
#endif

#define CATLASS_DEVICE __forceinline__ __aicore__
#ifdef __CCE__
#define CATLASS_HOST_DEVICE __forceinline__[host, aicore]
#else
#define CATLASS_HOST_DEVICE
#endif
#define CATLASS_GLOBAL __global__ __aicore__

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

namespace MatmulEmuAlign {
template <class T>
using RemoveCvrefT = std::remove_cv_t<std::remove_reference_t<T>>;
template <class T>
struct IsStatic : std::bool_constant<std::is_empty<RemoveCvrefT<T>>::value> {};
template <auto V>
struct Int {
    static constexpr auto value = V;
    using value_type = decltype(V);
    CATLASS_HOST_DEVICE constexpr operator value_type() const noexcept { return value; }
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};
} // namespace MatmulEmuAlign

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE constexpr T RoundUp(const T& val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto RoundUp(T const& val, U const& align)
{
    if constexpr (MatmulEmuAlign::IsStatic<T>::value && MatmulEmuAlign::IsStatic<U>::value) {
        constexpr uint32_t res = (T::value + U::value - 1) / U::value * U::value;
        return MatmulEmuAlign::Int<res>{};
    } else if constexpr (MatmulEmuAlign::IsStatic<T>::value) {
        return (T::value + align - 1) / align * align;
    } else if constexpr (MatmulEmuAlign::IsStatic<U>::value) {
        return (val + U::value - 1) / U::value * U::value;
    } else {
        return (val + align - 1) / align * align;
    }
}

template <uint32_t ALIGN, typename T>
CATLASS_HOST_DEVICE constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto RoundDown(T const& val, U const& align)
{
    if constexpr (MatmulEmuAlign::IsStatic<T>::value && MatmulEmuAlign::IsStatic<U>::value) {
        constexpr uint32_t res = T::value / U::value * U::value;
        return MatmulEmuAlign::Int<res>{};
    } else if constexpr (MatmulEmuAlign::IsStatic<T>::value) {
        return T::value / align * align;
    } else if constexpr (MatmulEmuAlign::IsStatic<U>::value) {
        return val / U::value * U::value;
    } else {
        return val / align * align;
    }
}

template <uint32_t DIVISOR, typename T>
CATLASS_HOST_DEVICE constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "DIVISOR must not be 0");
    return (dividend + DIVISOR - 1) / DIVISOR;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto CeilDiv(T const& dividend, U const& divisor)
{
    if constexpr (MatmulEmuAlign::IsStatic<T>::value && MatmulEmuAlign::IsStatic<U>::value) {
        constexpr uint32_t res = (T::value + U::value - 1) / U::value;
        return MatmulEmuAlign::Int<res>{};
    } else if constexpr (MatmulEmuAlign::IsStatic<T>::value) {
        return (T::value + divisor - 1) / divisor;
    } else if constexpr (MatmulEmuAlign::IsStatic<U>::value) {
        return (dividend + U::value - 1) / U::value;
    } else {
        return (dividend + divisor - 1) / divisor;
    }
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto Max(T const& a, U const& b)
{
    return a > b ? a : b;
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto Min(T const& a, U const& b)
{
    return a < b ? a : b;
}

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;
constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;
constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

class EmptyClass {};

#if defined(__CCE__)
using AscendC::SizeOfBits;
#else
template <typename T>
struct SizeOfBits {
    static constexpr size_t value = sizeof(T) * 8;
};
#endif

template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE constexpr ReturnType BitsToBytes(T bits)
{
    return (static_cast<ReturnType>(bits) + static_cast<ReturnType>(7)) / static_cast<ReturnType>(8);
}

template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE constexpr ReturnType BytesToBits(T bytes)
{
    return static_cast<ReturnType>(bytes) * static_cast<ReturnType>(8);
}

/// Statically-sized array specifying Coords within a tensor
template <int RANK_,                 ///< Logical rank of coordinate
          class Index_ = uint32_t,   ///< Index type used for each dimension
          class LongIndex_ = int64_t ///< Long index type used for linear offsets
          >
struct Coord {
public:
    // Number of elements in Coord
    static const int RANK = RANK_;

    // Index typen used to store elements
    using Index = Index_;

    // Type used to represent linear offsets
    using LongIndex = LongIndex_;

    // Default ctor initializes uniformly
    CATLASS_HOST_DEVICE constexpr explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    // Constructs from an array of integers
    CATLASS_HOST_DEVICE constexpr Coord(Index const (&idx_)[RANK])
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    CATLASS_HOST_DEVICE
    int Argmin() const { return ArgminImpl<1>(0); }

    // Returns the index of the dimension with greatest value
    CATLASS_HOST_DEVICE
    int Argmax() const { return ArgmaxImpl<1>(0); }

    // Returns true if Coord is non-zero
    CATLASS_HOST_DEVICE
    explicit operator bool() const { return AnyImpl<0>(); }

    // Return true if Coord is uniformly zero.
    CATLASS_HOST_DEVICE
    bool operator!() const { return !AnyImpl<0>(); }

    // Element-wise addition
    CATLASS_HOST_DEVICE
    Coord operator+(Coord const& b) const
    {
        Coord c;
        AddCoordImpl<0>(c, b);
        return c;
    }

    // Add a scalar to each element
    CATLASS_HOST_DEVICE
    Coord operator+(const Index val) const
    {
        Coord c;
        AddScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise subtraction
    CATLASS_HOST_DEVICE
    Coord operator-(Coord const& b) const
    {
        Coord c;
        SubCoordImpl<0>(c, b);
        return c;
    }

    // Subtract a scalar from each element
    CATLASS_HOST_DEVICE
    Coord operator-(Index const val) const
    {
        Coord c;
        SubScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise multiply
    CATLASS_HOST_DEVICE
    Coord operator*(Coord const& b) const
    {
        Coord c;
        MulCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise division
    CATLASS_HOST_DEVICE
    Coord operator/(Coord const& b) const
    {
        Coord c;
        DivCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise mod
    CATLASS_HOST_DEVICE
    Coord operator%(Coord const& b) const
    {
        Coord c;
        ModCoordImpl<0>(c, b);
        return c;
    }

    // In-place addition
    CATLASS_HOST_DEVICE
    Coord& operator+=(Coord const& b)
    {
        PlusEqualImpl<0>(b);
        return *this;
    }

    // In-place equal
    CATLASS_HOST_DEVICE
    bool operator==(Coord const& b) const { return EqualCoordImpl<0>(b); }

    // In-place equal
    CATLASS_HOST_DEVICE
    bool operator==(Index const val) const { return EqualScalarImpl<0>(val); }

    // Member acces operator
    CATLASS_HOST_DEVICE
    Index& operator[](int dim) { return idx[dim]; }

    // Member access operator
    CATLASS_HOST_DEVICE
    Index const& operator[](int dim) const { return idx[dim]; }

    // Gets the index of a given Coord element
    template <int DIM>
    CATLASS_HOST_DEVICE Index& At()
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    CATLASS_HOST_DEVICE
    Index& At(int dim) { return idx[dim]; }

    // Gets the index of a given Coord element
    template <int DIM>
    CATLASS_HOST_DEVICE Index const& At() const
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    CATLASS_HOST_DEVICE
    Index const& At(int dim) const { return idx[dim]; }

    template <int... Is>
    CATLASS_HOST_DEVICE auto GetCoordByAxis() const
    {
        Index idx_[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{idx_};
    }

    CATLASS_HOST_DEVICE
    static Coord Min(Coord const& a, Coord const& b)
    {
        Coord res;
        for (int i = 0; i < RANK; ++i) {
            res[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return res;
    }

private:
    template <int N>
    CATLASS_HOST_DEVICE int ArgminImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        } else {
            return ArgminImpl<N + 1>(idx[N] < idx[i] ? N : i);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE int ArgmaxImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        } else {
            return ArgmaxImpl<N + 1>(idx[N] > idx[i] ? N : i);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE bool AnyImpl() const
    {
        if constexpr (N == RANK) {
            return false;
        } else {
            return idx[N] || AnyImpl<N + 1>();
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void AddCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + b.idx[N];
            AddCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void AddScalarImpl(Coord& c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + val;
            AddScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void SubCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - b.idx[N];
            SubCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void SubScalarImpl(Coord& c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - val;
            SubScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void MulCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] * b.idx[N];
            MulCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void DivCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] / b.idx[N];
            DivCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void ModCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] % b.idx[N];
            ModCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE void PlusEqualImpl(Coord const& b)
    {
        if constexpr (N < RANK) {
            idx[N] += b.idx[N];
            PlusEqualImpl<N + 1>(b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE bool EqualCoordImpl(Coord const& b) const
    {
        if constexpr (N == RANK) {
            return true;
        } else {
            return idx[N] == b.idx[N] && EqualCoordImpl<N + 1>(b);
        }
    }

    template <int N>
    CATLASS_HOST_DEVICE bool EqualScalarImpl(Index const val) const
    {
        if constexpr (N == RANK) {
            return true;
        } else {
            return idx[N] == val && EqualScalarImpl<N + 1>(val);
        }
    }

    // Indices
    Index idx[RANK];
};

// Helper to make a 1-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<1, T> MakeCoord(T dim0)
{
    T values[1] = {dim0};
    return Coord<1, T>(values);
}

/// Helper to make a 2-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<2, T> MakeCoord(T dim0, T dim1)
{
    T values[2] = {dim0, dim1};
    return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<3, T> MakeCoord(T dim0, T dim1, T dim2)
{
    T values[3] = {dim0, dim1, dim2};
    return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<4, T> MakeCoord(T dim0, T dim1, T dim2, T dim3)
{
    T values[4] = {dim0, dim1, dim2, dim3};
    return Coord<4, T>(values);
}

/// Helper to make a 5-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<5, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4)
{
    T values[5] = {dim0, dim1, dim2, dim3, dim4};
    return Coord<5, T>(values);
}

/// Helper to make a 6-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<6, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5)
{
    T values[6] = {dim0, dim1, dim2, dim3, dim4, dim5};
    return Coord<6, T>(values);
}

/// Helper to make a 7-element coordinate
template <class T>
CATLASS_HOST_DEVICE constexpr Coord<7, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5, T dim6)
{
    T values[7] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6};
    return Coord<7, T>(values);
}

} // namespace Catlass

#endif // MATMUL_EMU_CATLASS_CATLASS_HPP
