/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_GEMM_COORD_HPP
#define MATMUL_EMU_CATLASS_GEMM_COORD_HPP

#include "./matmul_emu_catlass.hpp"

namespace Catlass {

template <uint32_t ROW_ = 1, uint32_t COLUMN_ = 1>
struct MatrixShape {
    static constexpr uint32_t ROW = ROW_;
    static constexpr uint32_t COLUMN = COLUMN_;

    static constexpr int64_t COUNT = ROW * COLUMN;

    CATLASS_HOST_DEVICE
    static Coord<2> ToCoord() { return MakeCoord(ROW, COLUMN); }
};

/// MatrixCoord wraps Coord<2, uint32_t> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=2
    using Base = Coord<2, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// Rows dimension
    static constexpr uint32_t ROW_INDEX = 0;

    /// Columns dimension
    static constexpr uint32_t COLUMN_INDEX = 1;

    /// Default ctor
    CATLASS_HOST_DEVICE
    MatrixCoord() {}

    /// Constructs from Coord<2>
    CATLASS_HOST_DEVICE
    MatrixCoord(Coord<2, Index> const& coord) : Base(coord) {}

    /// Helper to construct from a row and column
    CATLASS_HOST_DEVICE
    MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}

    /// Helper to construct from a row and column, which are LongIndex based
    CATLASS_HOST_DEVICE
    MatrixCoord(LongIndex row, LongIndex column) : Base(MakeCoord(Index(row), Index(column))) {}

    /// Returns the row of the coordinate
    CATLASS_HOST_DEVICE
    Index const& row() const { return this->At(ROW_INDEX); }

    /// Returns the row of the coordinate
    CATLASS_HOST_DEVICE
    Index& row() { return this->At(ROW_INDEX); }

    /// Returns the column of the coordinate
    CATLASS_HOST_DEVICE
    Index const& column() const { return this->At(COLUMN_INDEX); }

    /// Returns the column of the coordinate
    CATLASS_HOST_DEVICE
    Index& column() { return this->At(COLUMN_INDEX); }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    MatrixCoord operator+(Base const& b) const { return MatrixCoord(Base::operator+(b)); }

    /// In-place addition
    CATLASS_HOST_DEVICE
    MatrixCoord& operator+=(Base const& b)
    {
        Base::operator+=(b);
        return *this;
    }
};

/// GemmCoord is a structure derived from Coord<3> that specifies a location within the
/// coordinate space of a Gemm problem.
struct GemmCoord : public Coord<3, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// Gemm M dimension - rows of the output C matrix
    static constexpr int M_INDEX = 0;

    /// Gemm N dimension - columns of the output C matrix
    static constexpr int N_INDEX = 1;

    /// Gemm K dimension - inner dimension of the Gemm problem
    static constexpr int K_INDEX = 2;

    /// Default ctor
    CATLASS_HOST_DEVICE
    GemmCoord() {}

    /// Constructs from Coord<3> and a batch
    CATLASS_HOST_DEVICE
    GemmCoord(Coord<3, Index> const& coord) : Base(coord) {}

    /// Helper to construct from a K, N, M, batch variables
    CATLASS_HOST_DEVICE
    GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

    /// Returns the Gemm M coordinate
    CATLASS_HOST_DEVICE
    Index const& m() const { return this->At(M_INDEX); }

    /// Returns reference to the Gemm M coordinate
    CATLASS_HOST_DEVICE
    Index& m() { return this->At(M_INDEX); }

    /// Returns the Gemm N coordinate
    CATLASS_HOST_DEVICE
    Index const& n() const { return this->At(N_INDEX); }

    /// Returns reference to the Gemm N coordinate
    CATLASS_HOST_DEVICE
    Index& n() { return this->At(N_INDEX); }

    /// Returns the Gemm K coordinate
    CATLASS_HOST_DEVICE
    Index const& k() const { return this->At(K_INDEX); }

    /// Returns reference to the Gemm K coordinate
    CATLASS_HOST_DEVICE
    Index& k() { return this->At(K_INDEX); }

    CATLASS_HOST_DEVICE
    auto GetCoordMN() const { return this->GetCoordByAxis<M_INDEX, N_INDEX>(); }

    CATLASS_HOST_DEVICE
    auto GetCoordMK() const { return this->GetCoordByAxis<M_INDEX, K_INDEX>(); }

    CATLASS_HOST_DEVICE
    auto GetCoordKN() const { return this->GetCoordByAxis<K_INDEX, N_INDEX>(); }
};

} // namespace Catlass

#endif // MATMUL_EMU_CATLASS_GEMM_COORD_HPP
