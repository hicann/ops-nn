/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_GEMM_TILE_TILE_COPY_L0_HPP
#define MATMUL_EMU_CATLASS_GEMM_TILE_TILE_COPY_L0_HPP

///////////////////////////////////////////CopyL0CToGmTla/////////////////////////////////////////////////
// L0C copy mode
struct CopyToGM {};
struct CopyToL1 {};

template <class ArchTag, class TensorSrc, class TensorDst,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, bool ReluEnable = false,
          class Enable = void>
struct CopyL0CToGmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<Catlass::Arch::AtlasA2, TensorSrc_,
                      tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
                      ScaleGranularity::NO_QUANT, ReluEnable_,
                      std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto
        quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor");

        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dstTensor.data()[dstOffset],
                                                                         srcTensor.data()[srcOffset], intriParams);
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////

/// Partial specialization for CopyL1ToL0A, AtlasA2, zN in and zZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
                   tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
                   std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::iszZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::iszZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0A and zZ");

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = CeilDiv(tla::get<0>(dstTensor.originShape()),
                                                  tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol = CeilDiv(tla::get<1>(dstTensor.originShape()),
                                                  tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams = copy_impl::MakeLoadData2DParams(
            dstOuterShapeCol, srcOuterStrideCol / ELE_NUM_PER_FRACTAL, false);
        copy_impl::LoadData2DRows(dstTensor, srcTensor, dstOuterShapeRow, dstOuterStrideRow, srcOuterStrideRow,
                                  loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA2, float, zN in and zZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<float>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
    tla::Tensor<AscendC::LocalTensor<float>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
    std::enable_if_t<tla::detail::iszN<float, LayoutSrc>::value && tla::detail::iszZ<float, LayoutDst>::value>> {
    using Element = float;
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            std::is_same_v<float, typename TensorSrc::Element> && std::is_same_v<float, typename TensorDst::Element> &&
                tla::detail::iszN<float, typename TensorSrc::Layout>::value &&
                tla::detail::iszZ<float, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be float, L1 and zN, "
            "while TensorDst must be float, L0A and zZ");

        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1M = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        uint16_t l1K = RoundUp<ELE_NUM_PER_C0>(tla::get<1>(srcTensor.originShape()));
        uint16_t l0M = RoundUp<ELE_NUM_PER_C0>(tla::get<0>(dstTensor.originShape()));
        uint16_t l0K = RoundUp<ELE_NUM_PER_C0>(tla::get<1>(dstTensor.originShape()));
        AscendC::SetFmatrix(1, l1M, PAD_LIST, AscendC::FmatrixMode::FMATRIX_LEFT);
        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0K;
        loadDataParams.mExtension = l0M;
        loadDataParams.channelSize = l1K;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData<Element, config>(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0A, AtlasA2, nZ in and zZ out. (Transpose A)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
                   tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A2>,
                   std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::iszZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::iszZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::A2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0A and zZ");

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = CeilDiv(tla::get<0>(dstTensor.originShape()),
                                                  tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol = CeilDiv(tla::get<1>(dstTensor.originShape()),
                                                  tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams = copy_impl::MakeLoadData2DParams(dstOuterShapeCol, 1, true);
        copy_impl::LoadData2DRows(dstTensor, srcTensor, dstOuterShapeRow, dstOuterStrideRow, srcOuterStrideRow,
                                  loadDataParams);
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// Partial specialization for CopyL1ToL0B, AtlasA2, zN in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
                   tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
                   std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and zN, while TensorDst must be L0B and nZ");

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = CeilDiv(tla::get<0>(dstTensor.originShape()),
                                                  tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol = CeilDiv(tla::get<1>(dstTensor.originShape()),
                                                  tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::LoadData2DParams loadDataParams = copy_impl::MakeLoadData2DParams(
            dstOuterShapeCol, srcOuterStrideCol / ELE_NUM_PER_FRACTAL, true);
        copy_impl::LoadData2DRows(dstTensor, srcTensor, dstOuterShapeRow, dstOuterStrideRow, srcOuterStrideRow,
                                  loadDataParams);
    }
};

/// Partial specialization for CopyL1ToL0B, AtlasA2, nZ in and nZ out. (Transpose B)
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
                   tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::A1>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::B2>,
                   std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<ElementSrc>::value;

    // Methods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::A1 && TensorDst::position == AscendC::TPosition::B2,
            "The input parameters do not match. TensorSrc must be L1 and nZ, while TensorDst must be L0B and nZ");

        const uint32_t srcOuterStrideRow = tla::get<0, 1>(srcTensor.stride());
        const uint32_t srcOuterStrideCol = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterShapeRow = CeilDiv(tla::get<0>(dstTensor.originShape()),
                                                  tla::get<0, 0>(dstTensor.shape()));
        const uint32_t dstOuterShapeCol = CeilDiv(tla::get<1>(dstTensor.originShape()),
                                                  tla::get<1, 0>(dstTensor.shape()));
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData2DParams loadDataParams;
        if (srcOuterStrideRow == dstOuterStrideRow) {
            loadDataParams = copy_impl::MakeLoadData2DParams(dstOuterShapeRow * dstOuterShapeCol, 1, false);

            AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
        } else {
            loadDataParams = copy_impl::MakeLoadData2DParams(dstOuterShapeCol, 1, false);
            copy_impl::LoadData2DRows(dstTensor, srcTensor, dstOuterShapeRow, dstOuterStrideRow, srcOuterStrideRow,
                                      loadDataParams);
        }
    }
};

#endif // MATMUL_EMU_CATLASS_GEMM_TILE_TILE_COPY_L0_HPP
