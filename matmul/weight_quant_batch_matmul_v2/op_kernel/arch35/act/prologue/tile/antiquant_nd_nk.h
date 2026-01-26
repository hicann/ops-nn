/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_ANTIQUANT_ND_NK_H
#define PROLOGUE_TILE_ANTIQUANT_ND_NK_H
#include "kernel_operator_intf.h"
#include "../../utils/underscore.h"
#include "kernel_micro_datacopy_intf_impl.h"
namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile {

using AscendC::MicroAPI::RegTensor;
namespace MicroAPI = AscendC::MicroAPI;
namespace detail {
static constexpr MicroAPI::CastTrait S8_TO_FP16_TRAIT_ODD = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait FP16_TO_BF16_TRAIT = {
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr MicroAPI::CastTrait S4_TO_FP16_TRAIT_ODD = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

template <typename DtypeOut, typename DtypeIn>
__aicore__ inline void CastS8ToF16(
    RegTensor<DtypeOut>& weightF16Vreg, RegTensor<DtypeIn>& weightS8Vreg, RegTensor<half>& weightFp16Vreg,
    MicroAPI::MaskReg& maskAll)
{
    // PART_EVEN 表示按照如下形式处理做cast：
    // Vn 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    // Vd 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 .....
    if constexpr (!AscendC::IsSameType<typename MicroAPI::TypeGet<DtypeOut>::T, vector_f16>::value) {
        MicroAPI::Cast<half, DtypeIn, S8_TO_FP16_TRAIT_ODD>(weightFp16Vreg, weightS8Vreg, maskAll);
        MicroAPI::Cast<DtypeOut, half, FP16_TO_BF16_TRAIT>(weightF16Vreg, weightFp16Vreg, maskAll);
    } else {
        MicroAPI::Cast<DtypeOut, DtypeIn, S8_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightS8Vreg, maskAll);
    }
}

template <typename DtypeOut, typename DtypeIn>
__aicore__ inline void CastS4ToF16(
    RegTensor<DtypeOut>& weightF16Vreg, RegTensor<DtypeIn>& weightS4Vreg, RegTensor<half>& weightFp16Vreg,
    MicroAPI::MaskReg& maskAll)
{
    if constexpr (!AscendC::IsSameType<typename MicroAPI::TypeGet<DtypeOut>::T, vector_f16>::value) {
        MicroAPI::Cast<half, DtypeIn, S4_TO_FP16_TRAIT_ODD>(weightFp16Vreg, weightS4Vreg, maskAll);
        MicroAPI::Cast<DtypeOut, half, FP16_TO_BF16_TRAIT>(weightF16Vreg, weightFp16Vreg, maskAll);
    } else {
        MicroAPI::Cast<DtypeOut, DtypeIn, S4_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightS4Vreg, maskAll);
    }
}

template <typename DtypeOut, typename DtypeIn>
__aicore__ inline void CastFp4ToF16(
    RegTensor<DtypeOut>& f16Vreg, RegTensor<DtypeIn>& fp4Vreg, MicroAPI::MaskReg& maskAll)
{
    if constexpr (!IsSameType<typename MicroAPI::TypeGet<DtypeOut>::T, vector_f16>::value) {
        // CAST_FP4_TO_BF16_TRAIT中设置RegLayout::ZERO 表示按照如下形式处理做cast：
        // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
        // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
        MicroAPI::Cast<DtypeOut, DtypeIn, CAST_FP4_TO_BF16_TRAIT>(f16Vreg, fp4Vreg, maskAll);
    } else {
        // FP4-->FP16指令不支持，需要转换为BF16再转换为FP16
        RegTensor<bfloat16_t> bF16Vreg;
        MicroAPI::Cast<bfloat16_t, DtypeIn, CAST_FP4_TO_BF16_TRAIT>(bF16Vreg, fp4Vreg, maskAll);
        MicroAPI::Cast<DtypeOut, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(f16Vreg, bF16Vreg, maskAll);
    }
}

template <typename DtypeOut, typename DtypeIn, uint64_t OuterSize>
__aicore__ inline void WeightF16NdRegToNzUb(
    __local_mem__ DtypeOut*& weightF16PhyAddr0, __local_mem__ DtypeOut*& weightF16PhyAddr1,
    RegTensor<DtypeOut>& weightF16Vreg0, RegTensor<DtypeOut>& weightF16Vreg1, MicroAPI::MaskReg& maskAll)
{
    MicroAPI::DataCopy<DtypeOut, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr0, weightF16Vreg0, OuterSize + 1, 1, maskAll);
    MicroAPI::DataCopy<DtypeOut, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr1, weightF16Vreg1, OuterSize + 1, 1, maskAll);
}

// nk
// int8,int4->float16
// per-tensor
template <class TensorOut, int32_t K, class TensorTraitIn, class TensorScale, class Shape, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix n to 64
    Arch::Ascend910_95, AntiquantFixTile<64, K, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    TensorScale, TensorScale, Shape,
    typename AscendC::Std::enable_if_t<
        IsRowMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int8_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int4b_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, hifloat8_t>) &&
        AscendC::Std::is_same_v<TensorScale, half>>> {
    using DtypeIn = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int4b_t>, int4x2_t, AscendC::PrimT<TensorTraitIn>>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix n to 64
    using Policy = AntiquantFixTile<64, K, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST = AscendC::Std::is_same_v<DtypeIn, int4x2_t> ?
                                                      MicroAPI::LoadDist::DIST_UNPACK4_B8 :
                                                      MicroAPI::LoadDist::DIST_UNPACK_B8;

    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn, const TensorScale& tensorScale,
        const TensorScale& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 =
            weightLowBitPhyAddr0 + ElemToByte<DtypeIn>(VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::N + 1) * VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopN = static_cast<uint16_t>(::Act::Gemm::Get<0>(shape));

        __VEC_SCOPE__
        {
            RegTensor<DtypeIn> weightVreg0;
            RegTensor<DtypeIn> weightVreg1;
            RegTensor<half> weightFp16Vreg0;
            RegTensor<half> weightFp16Vreg1;
            RegTensor<DtypeOut> weightF16Vreg0;
            RegTensor<DtypeOut> weightF16Vreg1;

            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg0, weightLowBitPhyAddr0 + ubLoopNIdx * ElemToByte<DtypeIn>(Policy::K));
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg1, weightLowBitPhyAddr1 + ubLoopNIdx * ElemToByte<DtypeIn>(Policy::K));
                if constexpr (AscendC::Std::is_same_v<DtypeIn, int4b_t>) {
                    CastS4ToF16<DtypeOut, DtypeIn>(weightF16Vreg0, weightVreg0, weightFp16Vreg0, maskAll);
                    CastS4ToF16<DtypeOut, DtypeIn>(
                        weightF16Vreg1, weightVreg1, weightFp16Vreg1, maskAll); // castS4toF16需要实现
                } else {
                    CastS8ToF16<DtypeOut, DtypeIn>(weightF16Vreg0, weightVreg0, weightFp16Vreg0, maskAll);
                    CastS8ToF16<DtypeOut, DtypeIn>(weightF16Vreg1, weightVreg1, weightFp16Vreg1, maskAll);
                }

                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, tensorOffset, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, tensorOffset, maskAll);
                }
                MicroAPI::Muls(weightF16Vreg0, weightF16Vreg0, tensorScale, maskAll);
                MicroAPI::Muls(weightF16Vreg1, weightF16Vreg1, tensorScale, maskAll);
                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::N>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1, maskAll);
            }
        }
    }
};

// nk
// int8,int4->bfloat16
// per-tensor
template <class TensorOut, int32_t K, class TensorTraitIn, class TensorScale, class Shape, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix n to 64
    Arch::Ascend910_95, AntiquantFixTile<64, K, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    TensorScale, TensorScale, Shape,
    typename AscendC::Std::enable_if_t<
        IsRowMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int8_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int4b_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, hifloat8_t>) &&
        AscendC::Std::is_same_v<TensorScale, bfloat16_t>>> {
    using DtypeIn = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int4b_t>, int4x2_t, AscendC::PrimT<TensorTraitIn>>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix n to 64
    using Policy = AntiquantFixTile<64, K, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST = AscendC::Std::is_same_v<DtypeIn, int4x2_t> ?
                                                      MicroAPI::LoadDist::DIST_UNPACK4_B8 :
                                                      MicroAPI::LoadDist::DIST_UNPACK_B8;

    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn, const TensorScale& tensorScale,
        const TensorScale& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 =
            weightLowBitPhyAddr0 + ElemToByte<DtypeIn>(VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::N + 1) * VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopN = static_cast<uint16_t>(::Act::Gemm::Get<0>(shape));

        __VEC_SCOPE__
        {
            RegTensor<DtypeIn> weightVreg0;
            RegTensor<DtypeIn> weightVreg1;
            RegTensor<half> weightFp16Vreg0;
            RegTensor<half> weightFp16Vreg1;
            RegTensor<DtypeOut> weightF16Vreg0;
            RegTensor<DtypeOut> weightF16Vreg1;
            RegTensor<DtypeOut> antiQuantScaleVreg;

            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(antiQuantScaleVreg, tensorScale);
            for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg0, weightLowBitPhyAddr0 + ubLoopNIdx * ElemToByte<DtypeIn>(Policy::K));
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg1, weightLowBitPhyAddr1 + ubLoopNIdx * ElemToByte<DtypeIn>(Policy::K));
                if constexpr (AscendC::Std::is_same_v<DtypeIn, int4b_t>) {
                    CastS4ToF16<DtypeOut, DtypeIn>(weightF16Vreg0, weightVreg0, weightFp16Vreg0, maskAll);
                    CastS4ToF16<DtypeOut, DtypeIn>(
                        weightF16Vreg1, weightVreg1, weightFp16Vreg1, maskAll); // castS4toF16需要实现
                } else {
                    CastS8ToF16<DtypeOut, DtypeIn>(weightF16Vreg0, weightVreg0, weightFp16Vreg0, maskAll);
                    CastS8ToF16<DtypeOut, DtypeIn>(weightF16Vreg1, weightVreg1, weightFp16Vreg1, maskAll);
                }
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, tensorOffset, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, tensorOffset, maskAll);
                }
                //  硬件指令不支持Muls bfloat16
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::N>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1, maskAll);
            }
        }
    }
};

// nk
// int8,int4
// per-channel
template <
    class TensorOut, int32_t K, class TensorTraitIn, class Shape, typename TensorTraitScale, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix n to 64
    Arch::Ascend910_95, AntiquantFixTile<64, K, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    AscendC::LocalTensor<TensorTraitScale>, AscendC::LocalTensor<TensorTraitScale>, Shape,
    typename AscendC::Std::enable_if_t<
        IsRowMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int8_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int4b_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, hifloat8_t>) &&
        QUANT_TYPE<AscendC::Std::remove_cvref_t<decltype(TensorTraitScale{}.GetShape())>> == QuantType::PER_CHANNEL>> {
    using DtypeIn = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int4b_t>, int4x2_t, AscendC::PrimT<TensorTraitIn>>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix n to 64
    using Policy = AntiquantFixTile<64, K, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_BRC_B16;
    static constexpr MicroAPI::LoadDist LD_DIST_W = AscendC::Std::is_same_v<DtypeIn, int4x2_t> ?
                                                        MicroAPI::LoadDist::DIST_UNPACK4_B8 :
                                                        MicroAPI::LoadDist::DIST_UNPACK_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale,
        const AscendC::LocalTensor<TensorTraitScale>& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeOut* antiQuantScaleBasePhyAddr = (__local_mem__ DtypeOut*)tensorScale.GetPhyAddr();
        __local_mem__ DtypeOut* antiQuantOffsetBasePhyAddr = (__local_mem__ DtypeOut*)tensorOffset.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 =
            weightLowBitPhyAddr0 + ElemToByte<DtypeIn>(VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::N + 1) * VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopN = static_cast<uint16_t>(::Act::Gemm::Get<0>(shape));
        __VEC_SCOPE__
        {
            RegTensor<DtypeOut> antiQuantScaleVreg;
            RegTensor<DtypeOut> antiQuantOffsetVreg;

            RegTensor<DtypeIn> weightVreg0;
            RegTensor<DtypeIn> weightVreg1;
            RegTensor<half> weightFp16Vreg0;
            RegTensor<half> weightFp16Vreg1;
            RegTensor<DtypeOut> weightF16Vreg0;
            RegTensor<DtypeOut> weightF16Vreg1;

            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

            for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(
                        antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr + ubLoopNIdx);
                }
                MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr + ubLoopNIdx);
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(
                    weightVreg0, weightLowBitPhyAddr0 + ubLoopNIdx * ElemToByte<DtypeIn>(Policy::K));
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(
                    weightVreg1, weightLowBitPhyAddr1 + ubLoopNIdx * ElemToByte<DtypeIn>(Policy::K));
                if constexpr (AscendC::Std::is_same_v<DtypeIn, int4b_t>) {
                    CastS4ToF16<DtypeOut, DtypeIn>(weightF16Vreg0, weightVreg0, weightFp16Vreg0, maskAll);
                    CastS4ToF16<DtypeOut, DtypeIn>(
                        weightF16Vreg1, weightVreg1, weightFp16Vreg1, maskAll); // castS4toF16需要实现
                } else {
                    CastS8ToF16<DtypeOut, DtypeIn>(weightF16Vreg0, weightVreg0, weightFp16Vreg0, maskAll);
                    CastS8ToF16<DtypeOut, DtypeIn>(weightF16Vreg1, weightVreg1, weightFp16Vreg1, maskAll);
                }
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
                }
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::N>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1, maskAll);
            }
        }
    }
};

// nk
// fp8_e5m2_t/fp8_e4m3fn_t
// per-channel
template <
    int32_t K, class TensorOut, class TensorTraitIn, class Shape, typename TensorTraitScale, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix n to 64
    Arch::Ascend910_95, AntiquantFixTile<64, K, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    AscendC::LocalTensor<TensorTraitScale>, AscendC::LocalTensor<TensorTraitScale>, Shape,
    typename AscendC::Std::enable_if_t<
        IsRowMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp8_e5m2_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp8_e4m3fn_t>) &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_CHANNEL>> {
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix n to 64
    using Policy = AntiquantFixTile<64, K, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_BRC_B16;
    static constexpr MicroAPI::LoadDist LD_DIST_W = MicroAPI::LoadDist::DIST_UNPACK_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale,
        const AscendC::LocalTensor<TensorTraitScale>& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeOut* antiQuantOffsetBasePhyAddr = (__local_mem__ DtypeOut*)tensorOffset.GetPhyAddr();
        __local_mem__ DtypeOut* antiQuantScaleBasePhyAddr = (__local_mem__ DtypeOut*)tensorScale.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + VECTOR_REG_SIZE<DtypeOut>;
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::N + 1) * VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopN = static_cast<uint16_t>(::Act::Gemm::Get<0>(shape));
        __VEC_SCOPE__
        {
            RegTensor<DtypeOut> antiQuantScaleVreg, antiQuantOffsetVreg;
            RegTensor<DtypeIn> weightF8Vreg0, weightF8Vreg1;
            RegTensor<DtypeOut> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
            RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
            MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

            for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(
                        antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr + ubLoopNIdx);
                }
                MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr + ubLoopNIdx);
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(weightF8Vreg0, weightLowBitPhyAddr0 + ubLoopNIdx * Policy::K);
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(weightF8Vreg1, weightLowBitPhyAddr1 + ubLoopNIdx * Policy::K);
                // 奇数、偶数位置分散到2个fp32寄存器存储
                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);

                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);

                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                    (RegTensor<uint16_t>&)weightF16Vreg2, (RegTensor<uint16_t>&)weightF16Vreg2,
                    (RegTensor<uint16_t>&)weightF16Vreg3, maskAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                    (RegTensor<uint16_t>&)weightF16Vreg0, (RegTensor<uint16_t>&)weightF16Vreg0,
                    (RegTensor<uint16_t>&)weightF16Vreg1, maskAll);

                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg, maskAll);
                }
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg, maskAll);

                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::N>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg2, maskAll);
            }
        }
    }
};

// ND
// fp4_e2m1_t/fp4_e1m2_t
// MX NK
template <
    int32_t K, class TensorOut, class TensorTraitIn, class Shape, typename TensorTraitScale, class TensorOffset,
    bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix N to 64
    Arch::Ascend910_95, AntiquantFixTile<64, K, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    AscendC::LocalTensor<TensorTraitScale>, TensorOffset, Shape,
    typename AscendC::Std::enable_if_t<
        IsRowMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp4x2_e2m1_t> &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_GROUP>> {
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix N to 64
    using Policy = AntiquantFixTile<64, K, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_E2B_B16;
    static constexpr MicroAPI::LoadDist LD_DIST_W = MicroAPI::LoadDist::DIST_UNPACK4_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale, const TensorOffset& tensorOffset, const Shape& shape)
    {
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorScale.GetPhyAddr();
        // 8个scale对应8/2 * 32=128个数
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr1 = antiQuantScaleF16PhyAddr0 + 8;
        __ubuf__ DtypeIn* weightLowBitPhyAddr0 = (__ubuf__ DtypeIn*)tensorIn.GetPhyAddr();
        // 2：一次处理64个4bit数据到16bit，右移2位
        __ubuf__ DtypeIn* weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (AscendC::VECTOR_REG_WIDTH >> 2);
        __ubuf__ DtypeOut* weightF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        __ubuf__ DtypeOut* weightF16PhyAddr1 =
            weightF16PhyAddr0 + (Policy::N + 1) * (VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        uint16_t ubLoopN = ::Act::Gemm::Get<0>(shape);
        __VEC_SCOPE__
        {
            // 每次处理一行， 一行为256个FP4的数， 分两条指令处理，每条指令处理128个FP4的数
            RegTensor<DtypeOut> antiQuantScaleF16Vreg0;
            RegTensor<DtypeOut> antiQuantScaleF16Vreg1;
            RegTensor<DtypeIn> weightFp4Vreg0;
            RegTensor<DtypeIn> weightFp4Vreg1;
            RegTensor<DtypeOut> weightF16Vreg0;
            RegTensor<DtypeOut> weightF16Vreg1;
            MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
                // DIST_E2B_B16 表示搬运模式如下, 将一个f16的数扩展成16个
                // Vn  1 2 3 4 5 6 7 8
                // Vd
                // 11111111111111112222222222222233333333333333334444444444444444455555555555555666666666666666677777777777777
                MicroAPI::LoadAlign<DtypeOut, LD_DIST_SCALE>(
                    antiQuantScaleF16Vreg0, antiQuantScaleF16PhyAddr0 + ubLoopNIdx * (AscendC::VECTOR_REG_WIDTH >> 2));
                MicroAPI::LoadAlign<DtypeOut, LD_DIST_SCALE>(
                    antiQuantScaleF16Vreg1, antiQuantScaleF16PhyAddr1 + ubLoopNIdx * (AscendC::VECTOR_REG_WIDTH >> 2));
                // DIST_UNPK_B8 表示按照如下形式载入, 其中Vn中一个数字为4bit:
                // Vn  1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd  1 2 x x x x x x 3 4 x x x x x x .....
                MicroAPI::LoadAlign<DtypeIn, LD_DIST_W>(
                    weightFp4Vreg0, (__local_mem__ DtypeIn*)(weightLowBitPhyAddr0 + ubLoopNIdx * (Policy::K >> 1)));
                MicroAPI::LoadAlign<DtypeIn, LD_DIST_W>(
                    weightFp4Vreg1, (__local_mem__ DtypeIn*)(weightLowBitPhyAddr1 + ubLoopNIdx * (Policy::K >> 1)));
                CastFp4ToF16<DtypeOut, DtypeIn>(weightF16Vreg0, weightFp4Vreg0, maskAll);
                CastFp4ToF16<DtypeOut, DtypeIn>(weightF16Vreg1, weightFp4Vreg1, maskAll);
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleF16Vreg0, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleF16Vreg1, maskAll);
                MicroAPI::StoreAlign<
                    DtypeOut, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    weightF16PhyAddr0, weightF16Vreg0, Policy::N + 1, 1, maskAll);
                MicroAPI::StoreAlign<
                    DtypeOut, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    weightF16PhyAddr1, weightF16Vreg1, Policy::N + 1, 1, maskAll);
            }
        }
    }
};

} // namespace detail
} // namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile
#endif