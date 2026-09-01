/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_GEMM_TILE_TILE_COPY_HPP
#define MATMUL_EMU_CATLASS_GEMM_TILE_TILE_COPY_HPP

#include <type_traits>

#include "../../matmul_emu_catlass.hpp"
#include "../../arch/matmul_emu_arch.hpp"
#include "../matmul_emu_gemm_type.hpp"
#include "../../layout/matmul_emu_layout.hpp"
#include "../../../tla/matmul_emu_layout.hpp"

namespace Catlass::detail {

template <class Element, class LayoutTag>
struct TagToLayout {
    using type = LayoutTag;
};

template <class Element>
struct TagToLayout<Element, layout::RowMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
};

template <class Element>
struct TagToLayout<Element, layout::ColumnMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<tla::Int<1>, int64_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::VectorLayout> {
    using type = tla::Layout<tla::Shape<uint32_t>, tla::Stride<tla::Int<1>>>;
};

template <class Element>
struct TagToLayout<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>,
                    tla::Stride<tla::Int<1>, int64_t>>>;
};

template <class Element>
struct TagToLayout<Element, layout::zZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<ELE_NUM_PER_C0>, int64_t>,
                    tla::Stride<tla::Int<1>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

template <class Element>
struct TagToLayout<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<1>, int64_t>,
                    tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

template <class Element, class LayoutTag>
using TagToLayout_t = typename TagToLayout<Element, LayoutTag>::type;

constexpr uint32_t ELE_NUM_PER_FRACTAL_L0C = 256;
using LayoutL0C = tla::Layout<
    tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
    tla::Stride<tla::Stride<tla::Int<C0_NUM_PER_FRACTAL>, tla::Int<ELE_NUM_PER_FRACTAL_L0C>>,
                tla::Stride<tla::Int<1>, int64_t>>>;

} // namespace Catlass::detail

namespace Catlass::Gemm::helper {

template <class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
};

template <class Element>
struct L1AlignHelper<Element, layout::RowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::ColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template <class Element>
struct L1AlignHelper<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template <class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>,
                  "Unsupported element accumulator selector, can not find the specialization.");
};

template <>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template <class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>, "Unsupported layout selector, can not find the specialization.");
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template <class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>, "Unsupported layout selector, can not find the specialization.");
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template <class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template <class ArchTag>
struct L0ALayoutSelector {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported layout selector, can not find the specialization.");
};

template <>
struct L0ALayoutSelector<Arch::AtlasA2> {
    using Layout = layout::zZ;
};

} // namespace Catlass::Gemm::helper

namespace Catlass::Gemm::Tile {

namespace copy_impl {

template <uint32_t ELE_NUM_PER_C0, class TensorDst, class TensorSrc>
CATLASS_DEVICE void CopyGmNdToL1Nz(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t nValue,
                                   uint32_t dValue, uint32_t srcDValue, uint32_t dstInnerStrideRow,
                                   uint32_t dstOuterStrideCol)
{
    AscendC::Nd2NzParams intriParams;
    intriParams.ndNum = 1;
    intriParams.dValue = dValue;
    intriParams.srcNdMatrixStride = 0;
    intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
    intriParams.dstNzMatrixStride = 0;

    auto dstOffset = dstTensor.layout()(dstTensor.coord());
    auto srcOffset = srcTensor.layout()(srcTensor.coord());

    if (srcDValue < STRIDE_LIMIT) {
        intriParams.nValue = nValue;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    } else {
        intriParams.nValue = 1;
        intriParams.srcDValue = 0;
        intriParams.dstNzNStride = 0;
        for (uint32_t i = 0; i < nValue; i++) {
            AscendC::DataCopy(dstTensor.data()[dstOffset + i * ELE_NUM_PER_C0],
                              srcTensor.data()[srcOffset + i * srcDValue], intriParams);
        }
    }
}

CATLASS_DEVICE
AscendC::LoadData2DParams MakeLoadData2DParams(uint32_t repeatTimes, uint32_t srcStride, bool ifTranspose)
{
    AscendC::LoadData2DParams params;
    params.startIndex = 0;
    params.repeatTimes = repeatTimes;
    params.srcStride = srcStride;
    params.sid = 0;
    params.dstGap = 0;
    params.ifTranspose = ifTranspose;
    params.addrMode = 0;
    return params;
}

template <class TensorDst, class TensorSrc>
CATLASS_DEVICE void LoadData2DRows(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t dstOuterShapeRow,
                                   uint32_t dstOuterStrideRow, uint32_t srcOuterStrideRow,
                                   AscendC::LoadData2DParams const& loadDataParams)
{
    auto dstOffset = dstTensor.layout()(dstTensor.coord());
    auto srcOffset = srcTensor.layout()(srcTensor.coord());
    for (uint32_t i = 0; i < dstOuterShapeRow; i++) {
        AscendC::LoadData(dstTensor.data()[dstOffset + i * dstOuterStrideRow],
                          srcTensor.data()[srcOffset + i * srcOuterStrideRow], loadDataParams);
    }
}

} // namespace copy_impl

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTla, can not find the specialization.");
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyGmToL1, AtlasA2, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN");

        const uint32_t nValue = tla::get<0>(srcTensor.originShape());
        const uint32_t dValue = tla::get<1>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        copy_impl::CopyGmNdToL1Nz<ELE_NUM_PER_C0>(dstTensor, srcTensor, nValue, dValue, srcDValue, dstInnerStrideRow,
                                                  dstOuterStrideCol);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA2, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(tla::detail::isColumnMajor<typename TensorSrc::Layout>::value &&
                          tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::GM &&
                          TensorDst::position == AscendC::TPosition::A1,
                      "The input parameters do not match. TensorSrc must be GM and ColumnMajor, "
                      "while TensorDst must be L1 and nZ");

        const uint32_t nValue = tla::get<1>(srcTensor.originShape());
        const uint32_t dValue = tla::get<0>(srcTensor.originShape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<0, 1>(dstTensor.stride());

        copy_impl::CopyGmNdToL1Nz<ELE_NUM_PER_C0>(dstTensor, srcTensor, nValue, dValue, srcDValue, dstInnerStrideRow,
                                                  dstOuterStrideCol);
    }
};

enum class ScaleGranularity { UNDEFINED = -1, NO_QUANT = 0, PER_TENSOR, PER_CHANNEL, PER_GROUP };

template <class ArchTag, class ElementSrc, class ElementDst,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm fp32 to fp32
template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, float, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

#include "matmul_emu_tile_copy_l0.hpp"

template <class ArchTag, class ElementA_, class LayoutTagA_, class ElementB_, class LayoutTagB_, class ElementC_,
          class LayoutTagC_, class ElementBias = void>
struct PackedTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA,
                                                                                 ElementB>::ElementAccumulator;
    static constexpr bool ReluEnable = false;
    static constexpr ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT;
    static constexpr bool HAS_BIAS = false;
    static constexpr bool HAS_QUANT_TENSOR = false;

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = typename helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = layout::nZ;
    using LayoutTagL0C = layout::L0C;

    using LayoutA = Catlass::detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = Catlass::detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = Catlass::detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = Catlass::detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = Catlass::detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = Catlass::detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = Catlass::detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename Catlass::detail::LayoutL0C;

    using TensorL1A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::A1>;
    using TensorL1B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::A1>;
    using TensorL0A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::A2>;
    using TensorL0B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Gemm::Tile::TileCopyTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorL1B>;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL1ToBT = EmptyClass;

    template <class TensorC>
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
};

template <class ArchTag, class ElementA, class LayoutTagL1A>
struct TileMmadTla {
    CATLASS_DEVICE
    TileMmadTla() {}

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(TensorC const& l0CTensor, TensorA const& l0ATensor, TensorB const& l0BTensor,
                                   uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

} // namespace Catlass::Gemm::Tile

#endif // MATMUL_EMU_CATLASS_GEMM_TILE_TILE_COPY_HPP
