/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_LAYOUT_MAKE_HPP
#define MATMUL_EMU_LAYOUT_MAKE_HPP

#include "matmul_emu_layout_core.hpp"

namespace tla {

// Advanced Layout constructions

// Make a vector layout.
template <class T>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const& len)
{
    return MakeLayout(MakeShape(len), MakeStride(Int<1>{}), MakeShape(len));
}

namespace detail {

template <class Tag>
struct LayoutTagType {};

template <class Element>
struct LayoutElemTraits {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToBits(Catlass::BYTE_PER_C0) /
                                               Catlass::SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) /
                                                    Catlass::SizeOfBits<Element>::value;
};

template <class Element, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutByTag(T const& /*rows*/, U const& cols,
                                                   LayoutTagType<Catlass::layout::VectorLayout>)
{
    return MakeLayout(MakeShape(cols), MakeStride(Int<1>{}), MakeShape(cols));
}

template <class Element, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutByTag(T const& rows, U const& cols,
                                                   LayoutTagType<Catlass::layout::RowMajor>)
{
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    if constexpr (std::is_same_v<Element, float4_e2m1x2_t> || std::is_same_v<Element, float4_e1m2x2_t>) {
        return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)RoundUp(cols, 2), Int<1>{}),
                          MakeShape(rows, cols));
    }
#endif
    return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)cols, Int<1>{}), MakeShape(rows, cols));
}

template <class Element, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutByTag(T const& rows, U const& cols,
                                                   LayoutTagType<Catlass::layout::ColumnMajor>)
{
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    if constexpr (std::is_same_v<Element, float4_e2m1x2_t> || std::is_same_v<Element, float4_e1m2x2_t>) {
        return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)RoundUp(rows, 2)),
                          MakeShape(rows, cols));
    }
#endif
    return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)rows), MakeShape(rows, cols));
}

template <class Element, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutByTag(T const& rows, U const& cols, LayoutTagType<Catlass::layout::zN>)
{
    constexpr uint32_t ELE_NUM_PER_C0 = LayoutElemTraits<Element>::ELE_NUM_PER_C0;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = LayoutElemTraits<Element>::ELE_NUM_PER_FRACTAL;
    return MakeLayout(
        MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                  MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
        MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                   MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0)),
        MakeShape(rows, cols));
}

template <class Element, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutByTag(T const& rows, U const& cols, LayoutTagType<Catlass::layout::zZ>)
{
    constexpr uint32_t ELE_NUM_PER_C0 = LayoutElemTraits<Element>::ELE_NUM_PER_C0;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = LayoutElemTraits<Element>::ELE_NUM_PER_FRACTAL;
    return MakeLayout(
        MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                  MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
        MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{},
                              RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                   MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
        MakeShape(rows, cols));
}

template <class Element, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutByTag(T const& rows, U const& cols, LayoutTagType<Catlass::layout::L0C>)
{
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
    return MakeLayout(
        MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                  MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
        MakeStride(MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                   MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) *
                                            Catlass::C0_NUM_PER_FRACTAL)),
        MakeShape(rows, cols));
}

template <class Element, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutByTag(T const& rows, U const& cols, LayoutTagType<Catlass::layout::nZ>)
{
    constexpr uint32_t ELE_NUM_PER_C0 = LayoutElemTraits<Element>::ELE_NUM_PER_C0;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = LayoutElemTraits<Element>::ELE_NUM_PER_FRACTAL;
    return MakeLayout(
        MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                  MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
        MakeStride(MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                   MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})),
        MakeShape(rows, cols));
}

} // namespace detail

// Make a inner layout with Rows and Cols.
template <class Element, class LayoutTag, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const& rows, U const& cols)
{
    static_assert(std::is_same_v<LayoutTag, Catlass::layout::RowMajor> ||
                      std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
                      std::is_same_v<LayoutTag, Catlass::layout::VectorLayout> ||
                      std::is_same_v<LayoutTag, Catlass::layout::zN> ||
                      std::is_same_v<LayoutTag, Catlass::layout::nZ> ||
                      std::is_same_v<LayoutTag, Catlass::layout::zZ> || std::is_same_v<LayoutTag, Catlass::layout::L0C>,
                  "Unsupported LayoutTag for MakeLayout, only RowMajor/ColumnMajor/VectorLayout/zN/nZ/zZ/L0C");
    return detail::MakeLayoutByTag<Element>(rows, cols, detail::LayoutTagType<LayoutTag>{});
}

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <class Element, class LayoutTag, bool isMxScaleB, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeMxScaleLayout(T const& rows, U const& cols)
{
    static_assert(std::is_same_v<Element, float8_e8m0_t> && (std::is_same_v<LayoutTag, Catlass::layout::RowMajor> ||
                                                             std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
                                                             std::is_same_v<LayoutTag, Catlass::layout::zZ> ||
                                                             std::is_same_v<LayoutTag, Catlass::layout::nN>),
                  "only support RowMajor, ColumnMajor, zZ, nN in fp8_e8m0_t dtype");

    constexpr uint32_t ELE_NUM_PER_C0 = 2;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 32;

    if constexpr (std::is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(
                MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                MakeStride(RoundUp(cols, Int<ELE_NUM_PER_C0>{}), MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        } else {
            return MakeLayout(MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                              MakeStride(MakeStride(Int<1>{}, cols * ELE_NUM_PER_C0), Int<ELE_NUM_PER_C0>{}),
                              MakeShape(rows, cols));
        }
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                              MakeStride(Int<ELE_NUM_PER_C0>{}, MakeStride(Int<1>{}, rows * ELE_NUM_PER_C0)),
                              MakeShape(rows, cols));
        } else {
            return MakeLayout(
                MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                MakeStride(MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{}), RoundUp(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        }
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::zZ>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                      MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{},
                                  RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                       MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                      MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                       MakeStride(Int<ELE_NUM_PER_C0>{},
                                  RoundUp((int64_t)rows, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL)),
            MakeShape(rows, cols));
    }
}
#endif

namespace detail {

template <class OriginBase, class TileShape, class Coord, int... Is>
CATLASS_HOST_DEVICE constexpr auto CropOriginShape(OriginBase const& originBase, TileShape const& tileShape,
                                                   Coord const& coord, seq<Is...>)
{
    return MakeShape(tla::min(static_cast<uint32_t>(get<Is>(tileShape)),
                              (static_cast<uint32_t>(get<Is>(coord)) < static_cast<uint32_t>(get<Is>(originBase))) ?
                                  (static_cast<uint32_t>(get<Is>(originBase)) - static_cast<uint32_t>(get<Is>(coord))) :
                                  0u)...);
}

} // namespace detail

/// 创建 tile layout：使用指定的 tile 尺寸用于内存布局计算，同时携带实际逻辑尺寸（origin_shape）。
/// coord 是元素坐标，用于计算实际的 originShape（处理边界情况）。
/// Supports layouts of any rank (rank >= 1) for depth==1 layouts.
/// For depth>1 (fractal) layouts, currently only rank-2 is supported.
template <class Layout, class TileShape, class Coord>
CATLASS_HOST_DEVICE constexpr auto GetTileLayout(Layout const& layout, TileShape const& tileShape, Coord const& coord)
{
    static_assert(is_tuple<TileShape>::value && depth_v<TileShape> == 1 && rank_v<TileShape> >= 1,
                  "GetTileLayout: TileShape must be a flat tuple with rank >= 1.");
    static_assert(is_tuple<Coord>::value && depth_v<Coord> == 1 && rank_v<Coord> == rank_v<TileShape>,
                  "GetTileLayout: Coord must have the same rank as TileShape.");

    // 统一计算 tail tile 的逻辑尺寸（originShape 裁剪）
    auto tileOriginShape = detail::CropOriginShape(layout.originShape(), tileShape, coord, tuple_seq<TileShape>{});

    // depth==1 的布局（vector/matrix/tensor）：tile shape 直接作为 memory-layout shape
    // 支持任意 rank >= 1（但必须与 layout.rank 匹配）
    if constexpr (Layout::depth == 1) {
        static_assert(Layout::rank == rank_v<TileShape>,
                      "GetTileLayout: for depth==1 layouts, TileShape rank must match layout rank.");
        return MakeLayout(tileShape, layout.stride(), tileOriginShape);
    } else {
        // depth>1 的布局（fractal layout）：目前只支持 rank=2
        // 因为 fractal layout 通常用于矩阵（rank-2），需要把 (rows, cols) 转为同结构嵌套 shape
        static_assert(rank_v<TileShape> == 2,
                      "GetTileLayout: for depth>1 (fractal) layouts, TileShape must be rank-2 (rows, cols).");

        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);

        // MakeMxScaleLayout RowMajor A 等：第一维为行长度，第二维为 (C0, ceil(cols/C0))；与 catlass_dev
        // `MakeLayoutTile` 中 rank(shape<0>)==1 && rank(shape<1>)==2 分支一致。
        if constexpr (Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 1 &&
                      rank_v<decltype(shape<1>(Layout{}))> == 2) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                              layout.stride(), tileOriginShape);
        }
        // MakeMxScaleLayout B 侧等：shape 为 ((C0, ceil(rows/C0)), cols)；与 catlass_dev `MakeLayoutTile` 中
        // rank(shape<0>)==2 && rank(shape<1>)==1 分支一致。
        else if constexpr (Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 2 &&
                           rank_v<decltype(shape<1>(Layout{}))> == 1) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<0, 0>(layout))::value;
            return MakeLayout(MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                              layout.stride(), tileOriginShape);
        }
        // 典型 fractal（zN/nZ 等）：shape<0,0>、shape<1,0> 均为编译期常量；嵌套 ((r0,...),(c0,...))。
        else if constexpr (is_static<decltype(shape<0, 0>(layout))>::value &&
                           is_static<decltype(shape<1, 0>(layout))>::value) {
            constexpr uint32_t dstInnerShapeRow = decltype(shape<0, 0>(layout))::value;
            constexpr uint32_t dstInnerShapeCol = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(MakeShape(MakeShape(Int<dstInnerShapeRow>{}, CeilDiv<dstInnerShapeRow>(rows)),
                                        MakeShape(Int<dstInnerShapeCol>{}, CeilDiv<dstInnerShapeCol>(cols))),
                              layout.stride(), tileOriginShape);
        }
        // 内层块尺寸非编译期常量：运行期从 layout 读取再分块。
        else {
            const uint32_t dstInnerShapeRow = shape<0, 0>(layout);
            const uint32_t dstInnerShapeCol = shape<1, 0>(layout);
            return MakeLayout(MakeShape(MakeShape(dstInnerShapeRow, CeilDiv(rows, dstInnerShapeRow)),
                                        MakeShape(dstInnerShapeCol, CeilDiv(cols, dstInnerShapeCol))),
                              layout.stride(), tileOriginShape);
        }
    }
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutL0C(T const& rows, U const& cols)
{
    return MakeLayout<int8_t, Catlass::layout::L0C>(rows, cols);
}

} // namespace tla

#endif // MATMUL_EMU_LAYOUT_MAKE_HPP
