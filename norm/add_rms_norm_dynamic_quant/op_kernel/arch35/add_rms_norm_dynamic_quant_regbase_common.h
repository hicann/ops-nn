/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_regbase_common.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_COMMON_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_COMMON_H_

#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "../../rms_norm/arch35/rms_norm_regbase_common.h"

namespace AddRmsNormDynamicQuant {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

constexpr uint64_t ALIGN_512_FACTOR = 512;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t B32_BLOCK_NUM = 8;
constexpr uint64_t B8_BLOCK_NUM = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr uint32_t SUM_COUNT = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_ONE = 1;

constexpr int32_t VL_SIZE = GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / static_cast<int32_t>(sizeof(float)));
constexpr float DIV_FACTOR_INT8 = (static_cast<float>(1.0) / 127);
constexpr float DIV_FACTOR_FP8E4M3FN = (static_cast<float>(1.0) / 448);
constexpr float DIV_FACTOR_FP8E5M2 = (static_cast<float>(1.0) / 57344);
constexpr float DIV_FACTOR_HIFP8 = (static_cast<float>(1.0) / 32768);

constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp162Int8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitFp322Fp8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitFp322Hifp8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND,
};

template <typename T_X>
__aicore__ inline void CopyInX(
    TQue<QuePosition::VECIN, 1>& inQueueX, GlobalTensor<T_X>& srcGm, uint64_t gmOffset, uint32_t blockLen,
    uint32_t left = 0, uint32_t right = 0)
{
    LocalTensor<T_X> xLocal = inQueueX.AllocTensor<T_X>();
    DataCopyPadExtParams<T_X> padParams{
        true,                        // isPad
        static_cast<uint8_t>(left),  // leftPadding
        static_cast<uint8_t>(right), // rightPadding
        static_cast<T_X>(0.0)        // paddingValue
    };
    RmsNorm::DataCopyImpl<T_X>(xLocal, srcGm[gmOffset], 1, blockLen, 0, 0, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T_X>
__aicore__ inline void CopyOutX(
    GlobalTensor<T_X>& xGm, TQue<QuePosition::VECOUT, 1>& outQueueX, uint64_t gmOffset, uint32_t blockLen)
{
    LocalTensor<T_X> xLocal = outQueueX.DeQue<T_X>();
    RmsNorm::DataCopyImpl<T_X>(xGm[gmOffset], xLocal, 1, blockLen);
    outQueueX.FreeTensor(xLocal);
}

template <typename T, typename U, typename R>
__aicore__ inline void YCopyOutImpl(
    const U& dstTensor, const R& srcTensor, uint32_t blockCount, uint32_t blockLen, uint32_t srcStride = 0,
    uint32_t dstStride = 0)
{
    DataCopyExtParams extParams{
        static_cast<uint16_t>(blockCount),           // blockCount
        static_cast<uint32_t>(blockLen * sizeof(T)), // blockLen
        srcStride,                                   // srcStride
        dstStride,                                   // dstStride
        0                                            // rsv
    };
    DataCopyPad(dstTensor, srcTensor, extParams);
}

template <typename T_Y>
__aicore__ inline void CopyOutY(
    GlobalTensor<T_Y>& yGm, TQue<QuePosition::VECOUT, 1>& outQueueY, uint64_t gmOffset, uint32_t blockLen)
{
    LocalTensor<T_Y> yLocal = outQueueY.DeQue<T_Y>();
    YCopyOutImpl<T_Y>(yGm[gmOffset], yLocal, 1, blockLen);
    outQueueY.FreeTensor(yLocal);
}

__aicore__ inline void CopyOutScale(
    GlobalTensor<float>& scaleGm, TQue<QuePosition::VECOUT, 1>& outQueueScale, uint64_t gmOffset, uint32_t blockLen)
{
    LocalTensor<float> scaleLocal = outQueueScale.DeQue<float>();
    RmsNorm::DataCopyImpl<float>(scaleGm[gmOffset], scaleLocal, 1, blockLen);
    outQueueScale.FreeTensor(scaleLocal);
}

template <typename T_X, typename T_GAMMA, typename T_SMOOTH_SCALE = float, bool HAS_SMOOTH_SCALE = true, typename T_Y>
__aicore__ inline void ComputeYScale(
    LocalTensor<T_Y>& yLocal, LocalTensor<float>& scaleLocal, LocalTensor<T_X>& xLocal, LocalTensor<float>& rstdLocal,
    LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_SMOOTH_SCALE>& smoothScaleLocal, LocalTensor<float>& yTmpLocal,
    uint32_t rstdScaleOffset, uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ T_SMOOTH_SCALE* smoothScaleAddr;
    if constexpr (HAS_SMOOTH_SCALE) {
        smoothScaleAddr = (__ubuf__ T_SMOOTH_SCALE*)smoothScaleLocal.GetPhyAddr();
    }
    __local_mem__ T_Y* yAddr = (__ubuf__ T_Y*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        // VF0. Calc scale
        RegTensor<float> rstdReg, scaleReg;
        RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, smoothScaleRegFp32;
        MaskReg maskRegFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg maskReg;

        Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            NormCommon::LoadCastRegVF(xRegFp32, xAddr, idx, maskReg);
            NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
            if constexpr (HAS_SMOOTH_SCALE) {
                NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
            }
            Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
            Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
            if constexpr (HAS_SMOOTH_SCALE) {
                Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, yRegFp32, maskReg);
                Abs(yRegFp32, yRegFp32, maskReg);               // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
            } else {
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, xRegFp32, maskReg);
                Abs(yRegFp32, xRegFp32, maskReg);               // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
            }
        }
        ReduceMax(scaleReg, scaleReg, maskRegFull);
        if constexpr (IsSameType<T_Y, int8_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_INT8, maskRegOne);
        } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E4M3FN, maskRegOne);
        } else if constexpr (IsSameType<T_Y, fp8_e5m2_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E5M2, maskRegOne);
        } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_HIFP8, maskRegOne);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
    PipeBarrier<PIPE_V>();

    uint16_t repeatTimes2 = (uint16_t)CeilDivision(calCount, V_LENGTH);
    __VEC_SCOPE__
    {
        // VF1. Calc y
        RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
        RegTensor<int32_t> yRegInt32;
        RegTensor<half> yRegFp16;
        RegTensor<T_Y> yReg;
        MaskReg maskReg;

        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes2; idx++) {
            maskReg = UpdateMask<float>(calCount);
            DataCopy<float>(yRegFp32, yTmpAddr + idx * V_LENGTH);
            Div(yRegFp32, yRegFp32, scaleReg, maskReg);
            if constexpr (IsSameType<T_Y, int8_t>::value) {
                Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                Cast<half, float, castTraitFp322Fp16>(yRegFp16, yRegFp32Tmp, maskReg);
                Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
            } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                Cast<T_Y, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
            } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                Cast<T_Y, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
            }
            DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
        }
    }
}

template <typename T_X, typename T_GAMMA, typename T_SMOOTH_SCALE = float, bool HAS_SMOOTH_SCALE = true, typename T_Y>
__aicore__ inline void ComputeReduceMax(
    LocalTensor<float>& scaleLocal, LocalTensor<float>& yTmpLocal, LocalTensor<T_X>& xLocal,
    LocalTensor<float>& rstdLocal, LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_SMOOTH_SCALE>& smoothScaleLocal,
    uint32_t rstdScaleOffset, uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ T_SMOOTH_SCALE* smoothScaleAddr;
    if constexpr (HAS_SMOOTH_SCALE) {
        smoothScaleAddr = (__ubuf__ T_SMOOTH_SCALE*)smoothScaleLocal.GetPhyAddr();
    }
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> rstdReg, scaleReg, scaleLastReg;
        RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, smoothScaleRegFp32;
        MaskReg maskRegFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg maskReg;

        Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdScaleOffset);
        DataCopy<float>(scaleLastReg, scaleAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            NormCommon::LoadCastRegVF(xRegFp32, xAddr, idx, maskReg);
            NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
            if constexpr (HAS_SMOOTH_SCALE) {
                NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
            }
            Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
            Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
            if constexpr (HAS_SMOOTH_SCALE) {
                Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, yRegFp32, maskReg);
                Abs(yRegFp32, yRegFp32, maskReg);               // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
            } else {
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, xRegFp32, maskReg);
                Abs(yRegFp32, xRegFp32, maskReg);               // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
            }
        }
        ReduceMax(scaleReg, scaleReg, maskRegFull);
        Max(scaleReg, scaleReg, scaleLastReg, maskRegOne);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
}

template <typename T_Y>
__aicore__ inline void ComputeScale(LocalTensor<float>& scaleLocal, uint32_t rstdScaleOffset)
{
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> scaleReg;
        MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();

        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset);
        if constexpr (IsSameType<T_Y, int8_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_INT8, maskRegOne);
        } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E4M3FN, maskRegOne);
        } else if constexpr (IsSameType<T_Y, fp8_e5m2_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E5M2, maskRegOne);
        } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_HIFP8, maskRegOne);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
}

template <typename T_Y>
__aicore__ inline void ComputeY(
    LocalTensor<T_Y>& yLocal, LocalTensor<float>& scaleLocal, LocalTensor<float>& xLocal, uint32_t rstdScaleOffset,
    uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_Y* yAddr = (__ubuf__ T_Y*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
        RegTensor<int32_t> yRegInt32;
        RegTensor<half> yRegFp16;
        RegTensor<T_Y> yReg;
        MaskReg maskReg;

        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            DataCopy<float>(yRegFp32, xAddr + idx * V_LENGTH);
            Div(yRegFp32, yRegFp32, scaleReg, maskReg);
            if constexpr (IsSameType<T_Y, int8_t>::value) {
                Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                Cast<half, float, castTraitFp322Fp16>(yRegFp16, yRegFp32Tmp, maskReg);
                Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
            } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                Cast<T_Y, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
            } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                Cast<T_Y, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
            }
            DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
        }
    }
}

} // namespace AddRmsNormDynamicQuant
#endif // _ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_COMMON_H_
