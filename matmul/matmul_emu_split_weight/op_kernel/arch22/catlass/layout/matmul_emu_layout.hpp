/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_LAYOUT_LAYOUT_HPP
#define MATMUL_EMU_CATLASS_LAYOUT_LAYOUT_HPP

#include "../matmul_emu_catlass.hpp"
#include "../matmul_emu_gemm_coord.hpp"

namespace Catlass::layout {

/// Shared shape/stride accessors for all layout types.
template <class ShapeT, class StrideT>
struct LayoutShapeStride {
    using Shape = ShapeT;
    using Stride = StrideT;

    CATLASS_HOST_DEVICE
    constexpr LayoutShapeStride() = default;

    CATLASS_HOST_DEVICE
    constexpr LayoutShapeStride(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    CATLASS_HOST_DEVICE
    constexpr Shape shape() const { return shape_; }

    CATLASS_HOST_DEVICE
    constexpr Shape& shape() { return shape_; }

    CATLASS_HOST_DEVICE
    constexpr typename Shape::Index shape(int idx) const { return shape_[idx]; }

    CATLASS_HOST_DEVICE
    constexpr typename Shape::Index& shape(int idx) { return shape_[idx]; }

    CATLASS_HOST_DEVICE
    constexpr Stride stride() const { return stride_; }

    CATLASS_HOST_DEVICE
    constexpr Stride& stride() { return stride_; }

    CATLASS_HOST_DEVICE
    constexpr typename Stride::Index stride(int idx) const { return stride_[idx]; }

    CATLASS_HOST_DEVICE
    constexpr typename Stride::Index& stride(int idx) { return stride_[idx]; }

protected:
    Shape shape_{};
    Stride stride_{};
};

/// Shared origin-shape accessors for fractal layouts.
template <class OrgShapeT>
struct LayoutOrgShape {
    using OrgShape = OrgShapeT;

    CATLASS_HOST_DEVICE
    constexpr LayoutOrgShape() = default;

    CATLASS_HOST_DEVICE
    constexpr LayoutOrgShape(OrgShape orgShape) : orgShape_(orgShape) {}

    CATLASS_HOST_DEVICE
    constexpr typename OrgShape::Index orgShape(int idx) const { return orgShape_[idx]; }

    CATLASS_HOST_DEVICE
    constexpr typename OrgShape::Index& orgShape(int idx) { return orgShape_[idx]; }

protected:
    OrgShape orgShape_{};
};

/// Shared fractal (nZ/zN/zZ/L0C) origin/shape/stride storage and constructors.
struct FractalLayoutBase : public LayoutOrgShape<Coord<2, uint32_t>>,
                           public LayoutShapeStride<Coord<4, uint32_t>, Coord<4, int64_t>> {
    static constexpr int RANK = 4;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;

    CATLASS_HOST_DEVICE constexpr FractalLayoutBase(
        Index orgRows = 0,                 /// Number of rows of origin matrices
        Index orgCols = 0,                 /// Number of cols of origin matrices
        Index rowsInFractal = 0,           /// Number of rows inside the fractal
        Index rowsByFractal = 0,           /// number of rows by the fractal
        Index colsInFractal = 0,           /// number of cols inside the fractal
        Index colsByFractal = 0,           /// number of cols by the fractal
        LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
        LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
        LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
        LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : LayoutOrgShape(MakeCoord(orgRows, orgCols)),
          LayoutShapeStride(
              MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
              MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal))
    {}

    CATLASS_HOST_DEVICE constexpr FractalLayoutBase(OrgShape orgShape, Shape shape, Stride stride)
        : LayoutOrgShape(orgShape), LayoutShapeStride(shape, stride)
    {}
};

/// Mapping function for row-major matrices
struct RowMajor : public LayoutShapeStride<Coord<2, uint32_t>, Coord<2, int64_t>> {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    CATLASS_HOST_DEVICE
    RowMajor(Index rows = 0, Index cols = 0)
        : LayoutShapeStride(MakeCoord(rows, cols), MakeCoord(LongIndex(cols), LongIndex(1)))
    {}

    /// Constructor
    CATLASS_HOST_DEVICE
    RowMajor(Index rows, Index cols, LongIndex ldm)
        : LayoutShapeStride(MakeCoord(rows, cols), MakeCoord(ldm, LongIndex(1)))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE
    RowMajor(Shape shape, Stride stride) : LayoutShapeStride(shape, stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static RowMajor MakeLayout(Index rows, Index cols)
    {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        if constexpr (std::is_same_v<Element, float4_e2m1x2_t> || std::is_same_v<Element, float4_e1m2x2_t>) {
            return RowMajor(rows, cols, RoundUp<2>(cols));
        }
#endif
        return RowMajor(rows, cols);
    }

    template <class Element>
    CATLASS_HOST_DEVICE static RowMajor MakeLayoutInUb(MatrixCoord const& shape)
    {
        constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
        return RowMajor(shape.row(), shape.column(), RoundUp<ELE_NUM_PER_BLK>(shape.column()));
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    RowMajor GetTileLayout(MatrixCoord const& tileShape) const { return RowMajor(tileShape, stride()); }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const { return static_cast<LongIndex>(shape_[0]) * stride_[0]; }
};

/// Mapping function for col-major matrices
struct ColumnMajor : public LayoutShapeStride<Coord<2, uint32_t>, Coord<2, int64_t>> {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE
    ColumnMajor(Index rows = 0, Index cols = 0)
        : LayoutShapeStride(MakeCoord(rows, cols), MakeCoord(LongIndex(1), LongIndex(rows)))
    {}

    /// Constructor
    CATLASS_HOST_DEVICE
    ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : LayoutShapeStride(MakeCoord(rows, cols), MakeCoord(LongIndex(1), ldm))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE
    ColumnMajor(Shape shape, Stride stride) : LayoutShapeStride(shape, stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static ColumnMajor MakeLayout(Index rows, Index cols)
    {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        if constexpr (std::is_same_v<Element, float4_e2m1x2_t> || std::is_same_v<Element, float4_e1m2x2_t>) {
            return ColumnMajor(rows, cols, RoundUp<2>(rows));
        }
#endif
        return ColumnMajor(rows, cols);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    ColumnMajor GetTileLayout(MatrixCoord const& tileShape) const { return ColumnMajor(tileShape, stride()); }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const { return static_cast<LongIndex>(shape_[1]) * stride_[1]; }
};

/// Mapping function for nZ matrices which is col-major inside fractal and row-major between fractal
struct nZ : public FractalLayoutBase {
public:
    using FractalLayoutBase::FractalLayoutBase;

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static nZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nZ(orgRows, orgCols, ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL,
                  colsRound / C0_NUM_PER_FRACTAL, 1, colsRound * ELE_NUM_PER_C0, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
               (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    nZ GetTileLayout(MatrixCoord const& tileOriShape) const
    {
        auto tileShape = MakeCoord(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2),
                                   CeilDiv(tileOriShape.column(), shape(2)));
        return nZ(tileOriShape, tileShape, stride());
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const { return static_cast<LongIndex>(stride_[1]) * shape_[1]; }
};

/// Mapping function for zN matrices which is row-major inside fractal and col-major between fractal
struct zN : public FractalLayoutBase {
public:
    using FractalLayoutBase::FractalLayoutBase;

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static zN MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zN(orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0, ELE_NUM_PER_FRACTAL, 1, rowsRound * ELE_NUM_PER_C0);
    }

    CATLASS_HOST_DEVICE
    static zN MakeLayoutInL0C(MatrixCoord const& shape)
    {
        return zN(shape.row(), shape.column(), C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                  C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()), C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL, 1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
               (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    zN GetTileLayout(MatrixCoord const& tileOriShape) const
    {
        auto tileShape = MakeCoord(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2),
                                   CeilDiv(tileOriShape.column(), shape(2)));
        return zN(tileOriShape, tileShape, stride());
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const { return static_cast<LongIndex>(stride_[3]) * shape_[3]; }
};

/// Mapping function for zN matrices which is row-major inside fractal and row-major between fractal
struct zZ : public FractalLayoutBase {
public:
    using FractalLayoutBase::FractalLayoutBase;

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static zZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zZ(orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0, colsRound * C0_NUM_PER_FRACTAL, 1, ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }
};

/// Mapping function for L0C matrices
/// A special data layout for L0C memory, used for accumulator storage.
/// This layout similar to zN layout, but uses C0_NUM_PER_FRACTAL for both row and column inner dimensions.
struct L0C : public FractalLayoutBase {
public:
    using FractalLayoutBase::FractalLayoutBase;

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static L0C MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return L0C(orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, C0_NUM_PER_FRACTAL,
                   colsRound / C0_NUM_PER_FRACTAL, C0_NUM_PER_FRACTAL, ELE_NUM_PER_FRACTAL, 1,
                   rowsRound * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
               (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    L0C GetTileLayout(MatrixCoord const& tileOriShape) const
    {
        auto tileShape = MakeCoord(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2),
                                   CeilDiv(tileOriShape.column(), shape(2)));
        return L0C(tileOriShape, tileShape, stride());
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const { return static_cast<LongIndex>(stride_[3]) * shape_[3]; }
};

struct VectorLayout : public LayoutShapeStride<Coord<1, uint32_t>, Coord<1, int64_t>> {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 1;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Shape vector
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

    /// Logical coordinate
    using TensorCoord = Coord<RANK, Index>;

public:
    // Methods

    CATLASS_HOST_DEVICE
    VectorLayout(Index size = 0) : LayoutShapeStride(MakeCoord(size), MakeCoord(LongIndex(1))) {}

    CATLASS_HOST_DEVICE
    VectorLayout(Shape shape, Stride stride) : LayoutShapeStride(shape, stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE static VectorLayout MakeLayoutInUb(TensorCoord const& tileShape)
    {
        constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
        return VectorLayout{ELE_NUM_PER_BLK > (tileShape[0])};
    }

    CATLASS_HOST_DEVICE
    LongIndex GetOffset(TensorCoord const& coord) const { return stride_[0] * coord[0]; }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    VectorLayout GetTileLayout(TensorCoord const& tileShape) const { return VectorLayout(tileShape, stride()); }
};

} // namespace Catlass::layout

#endif // MATMUL_EMU_CATLASS_LAYOUT_LAYOUT_HPP
