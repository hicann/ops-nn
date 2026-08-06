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
#include "add_rms_norm_dynamic_quant_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "../../rms_norm/arch35/rms_norm_regbase_common.h"

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

namespace AddRmsNormDynamicQuant {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;
using NormCommon::NormCommonRegbase::StoreRegForDtype;
using RmsNorm::castTraitFp322Fp8;
using RmsNorm::castTraitFp322Hifp8;
using RmsNorm::CopyInX;
using RmsNorm::CopyOutX;
using RmsNorm::CopyOutY;
using RmsNorm::GetOverflowMode;
using RmsNorm::SetOverflowMode;
using RmsNorm::YCopyOutImpl;

template <typename T_Y>
using YCopyDtype = std::conditional_t<IsSameType<T_Y, int4b_t>::value, uint8_t, T_Y>;

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
constexpr float DIV_FACTOR_INT4 = (static_cast<float>(1.0) / 7);

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
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Int16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitInt162Half = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitF162I8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename TilingData>
__aicore__ inline void InitTiling(uint64_t& numM, uint64_t& numN, uint64_t& baseM, uint64_t& baseN,
                                  uint64_t& baseNDtypeAlign, uint64_t& baseNReduceAlign, uint64_t& powerSplit,
                                  uint64_t& mPerCore, uint64_t& mLastCore, float& epsilon, float& avgFactor,
                                  const TilingData* tilingData)
{
    numM = tilingData->numM;
    numN = tilingData->numN;
    baseM = tilingData->baseM;
    baseN = tilingData->baseN;
    baseNDtypeAlign = tilingData->baseNDtypeAlign;
    baseNReduceAlign = tilingData->baseNReduceAlign;
    powerSplit = tilingData->powerSplit;
    mPerCore = tilingData->mPerCore;
    mLastCore = tilingData->mLastCore;
    epsilon = tilingData->epsilon;
    avgFactor = tilingData->avgFactor;
}

__aicore__ inline void CopyOutScale(GlobalTensor<float>& scaleGm, TQue<QuePosition::VECOUT, 1>& outQueueScale,
                                    uint64_t gmOffset, uint32_t blockLen)
{
    LocalTensor<float> scaleLocal = outQueueScale.DeQue<float>();
    RmsNorm::DataCopyImpl<float>(scaleGm[gmOffset], scaleLocal, 1, blockLen);
    outQueueScale.FreeTensor(scaleLocal);
}

template <typename T, typename Queue>
__aicore__ inline void CopyInParamToQueue(Queue& inQueue, GlobalTensor<T>& srcGm, uint64_t numN)
{
    LocalTensor<T> local = inQueue.template AllocTensor<T>();
    RmsNorm::DataCopyImpl<T>(local, srcGm, 1, numN);
    inQueue.EnQue(local);
}

template <typename T_SMOOTH_SCALE, typename SmoothScale1Queue, typename SmoothScale2Queue>
__aicore__ inline void CopyInDynamicQuantCommon(SmoothScale1Queue& inQueueSmoothScale1,
                                                SmoothScale2Queue& inQueueSmoothScale2,
                                                GlobalTensor<T_SMOOTH_SCALE>& smoothScale1Gm,
                                                GlobalTensor<T_SMOOTH_SCALE>& smoothScale2Gm, uint64_t numN,
                                                bool hasSmoothScale1, bool hasSmoothScale2)
{
    if (hasSmoothScale1) {
        CopyInParamToQueue(inQueueSmoothScale1, smoothScale1Gm, numN);
    }
    if (hasSmoothScale2) {
        CopyInParamToQueue(inQueueSmoothScale2, smoothScale2Gm, numN);
    }
}

template <typename T_X, typename T_SMOOTH_SCALE, typename SmoothScale1Queue, typename SmoothScale2Queue,
          typename BetaQueue>
__aicore__ inline void PrepareOptionalParamLocals(SmoothScale1Queue& inQueueSmoothScale1,
                                                  SmoothScale2Queue& inQueueSmoothScale2, BetaQueue& inQueueBeta,
                                                  GlobalTensor<T_X>& betaGm,
                                                  LocalTensor<T_SMOOTH_SCALE>& smoothScale1Local,
                                                  LocalTensor<T_SMOOTH_SCALE>& smoothScale2Local,
                                                  LocalTensor<T_X>& betaLocal, uint64_t numN, bool hasSmoothScale1,
                                                  bool hasSmoothScale2, bool hasBeta)
{
    if (hasSmoothScale1) {
        smoothScale1Local = inQueueSmoothScale1.template DeQue<T_SMOOTH_SCALE>();
    }
    if (hasSmoothScale2) {
        smoothScale2Local = inQueueSmoothScale2.template DeQue<T_SMOOTH_SCALE>();
    }
    if (hasBeta) {
        CopyInParamToQueue(inQueueBeta, betaGm, numN);
        betaLocal = inQueueBeta.template DeQue<T_X>();
    }
}

template <typename T_X, typename T_Y, typename T_SMOOTH_SCALE, bool Y3_MODE, bool Y4_MODE>
__aicore__ inline void InitOptionalGmBuffers(
    GlobalTensor<T_SMOOTH_SCALE>& smoothScale1Gm, GlobalTensor<T_SMOOTH_SCALE>& smoothScale2Gm,
    GlobalTensor<YCopyDtype<T_Y>>& y2Gm, GlobalTensor<float>& scale2Gm, GlobalTensor<float>& y3Gm,
    GlobalTensor<T_X>& y4Gm, GlobalTensor<T_X>& betaGm, GM_ADDR smoothScale1, GM_ADDR smoothScale2, GM_ADDR y2,
    GM_ADDR scale2, GM_ADDR y3, GM_ADDR y4, GM_ADDR beta, uint64_t gmOffset, uint64_t gmLen, uint64_t scalesGmOffset,
    uint64_t mCore, uint64_t numN, bool hasSmoothScale1, bool hasSmoothScale2, bool outQuant2Flag, bool hasBeta)
{
    using TyCopy = YCopyDtype<T_Y>;
    if (hasSmoothScale1) {
        smoothScale1Gm.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smoothScale1, numN);
    }
    if (hasSmoothScale2) {
        smoothScale2Gm.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smoothScale2, numN);
    }
    if (outQuant2Flag) {
        uint64_t y2GmOffset = gmOffset;
        uint64_t y2GmLen = gmLen;
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            y2GmOffset = y2GmOffset >> 1;
            y2GmLen = y2GmLen >> 1;
        }
        y2Gm.SetGlobalBuffer((__gm__ TyCopy*)y2 + y2GmOffset, y2GmLen);
        scale2Gm.SetGlobalBuffer((__gm__ float*)scale2 + scalesGmOffset, mCore);
    }
    if (hasBeta) {
        betaGm.SetGlobalBuffer((__gm__ T_X*)beta, numN);
    }
    if constexpr (Y3_MODE) {
        y3Gm.SetGlobalBuffer((__gm__ float*)y3 + gmOffset, gmLen);
    }
    if constexpr (Y4_MODE) {
        y4Gm.SetGlobalBuffer((__gm__ T_X*)y4 + gmOffset, gmLen);
    }
}

template <typename T_GAMMA, bool Y3_MODE, bool Y4_MODE, bool HAS_BETA, bool HAS_SMOOTH_SCALE>
__aicore__ inline void ComputeYAndAbsMaxVF(RegTensor<float>& xRegFp32, RegTensor<float>& yRegFp32,
                                           RegTensor<float>& rstdReg, RegTensor<float>& gammaRegFp32,
                                           RegTensor<float>& betaRegFp32, RegTensor<float>& smoothScaleRegFp32,
                                           RegTensor<float>& scaleReg, MaskReg& maskReg, MaskReg& maskRegFull,
                                           __local_mem__ float* yTmpAddr, __local_mem__ float* y3Addr,
                                           __local_mem__ T_GAMMA* y4Addr, uint16_t idx)
{
    Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
    Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
    if constexpr (Y3_MODE) {
        StoreRegForDtype<float>(y3Addr, xRegFp32, maskReg, idx * V_LENGTH);
    }
    if constexpr (Y4_MODE) {
        StoreRegForDtype<T_GAMMA>(y4Addr, xRegFp32, maskReg, idx * V_LENGTH);
    }
    if constexpr (HAS_BETA) {
        Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
    }
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

template <typename T_X, typename T_GAMMA, typename T_SMOOTH_SCALE = float, bool HAS_SMOOTH_SCALE = true,
          bool HAS_BETA = false, typename T_Y, bool Y3_MODE, bool Y4_MODE>
__aicore__ inline void ComputeYScale(LocalTensor<YCopyDtype<T_Y>>& yLocal, LocalTensor<float>& scaleLocal,
                                     LocalTensor<T_X>& xLocal, LocalTensor<float>& rstdLocal,
                                     LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal,
                                     LocalTensor<T_SMOOTH_SCALE>& smoothScaleLocal, LocalTensor<float>& yTmpLocal,
                                     LocalTensor<float>& y3Local, LocalTensor<T_GAMMA>& y4Local,
                                     uint32_t rstdScaleOffset, uint32_t calCount)
{
    using TyCopy = YCopyDtype<T_Y>;
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ T_SMOOTH_SCALE* smoothScaleAddr;
    if constexpr (HAS_SMOOTH_SCALE) {
        smoothScaleAddr = (__ubuf__ T_SMOOTH_SCALE*)smoothScaleLocal.GetPhyAddr();
    }
    __local_mem__ T_GAMMA* betaAddr;
    if constexpr (HAS_BETA) {
        betaAddr = (__ubuf__ T_GAMMA*)betaLocal.GetPhyAddr();
    }
    __local_mem__ TyCopy* yAddr = (__ubuf__ TyCopy*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();
    __local_mem__ float* y3Addr = nullptr;
    if constexpr (Y3_MODE) {
        y3Addr = (__ubuf__ float*)y3Local.GetPhyAddr();
    }
    __local_mem__ T_GAMMA* y4Addr = nullptr;
    if constexpr (Y4_MODE) {
        y4Addr = (__ubuf__ T_GAMMA*)y4Local.GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        // VF0. Calc scale
        RegTensor<float> rstdReg, scaleReg;
        RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, betaRegFp32, smoothScaleRegFp32;
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
            if constexpr (HAS_BETA) {
                NormCommon::LoadCastRegVF(betaRegFp32, betaAddr, idx, maskReg);
            }
            ComputeYAndAbsMaxVF<T_GAMMA, Y3_MODE, Y4_MODE, HAS_BETA, HAS_SMOOTH_SCALE>(
                xRegFp32, yRegFp32, rstdReg, gammaRegFp32, betaRegFp32, smoothScaleRegFp32, scaleReg, maskReg,
                maskRegFull, yTmpAddr, y3Addr, y4Addr, idx);
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
        } else if constexpr (IsSameType<T_Y, int4b_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_INT4, maskRegOne);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
    PipeBarrier<PIPE_V>();

    uint16_t repeatTimes2 = (uint16_t)CeilDivision(calCount, V_LENGTH);
    __VEC_SCOPE__
    {
        // VF1. Calc y
        RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
        RegTensor<half> yRegHalf;
        RegTensor<TyCopy> yReg;
        MaskReg maskReg;

        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes2; idx++) {
            maskReg = UpdateMask<float>(calCount);
            DataCopy<float>(yRegFp32, yTmpAddr + idx * V_LENGTH);
            Div(yRegFp32, yRegFp32, scaleReg, maskReg);
            if constexpr (IsSameType<T_Y, int8_t>::value) {
                Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                Cast<half, float, castTraitFp322Fp16>(yRegHalf, yRegFp32Tmp, maskReg);
                Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegHalf, maskReg);
                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
            } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                Cast<T_Y, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
            } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                Cast<T_Y, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
            } else if constexpr (IsSameType<T_Y, int4b_t>::value) {
                RegTensor<int16_t> vregInt16Y;
                RegTensor<uint16_t> vregTmp1Y;
                RegTensor<uint8_t> vregTmp2Y;
                MaskReg mask4Int4 = CreateMask<float, MaskPattern::H>();
                Cast<int16_t, float, castTraitFp322Int16>(vregInt16Y, yRegFp32, maskReg);
                Cast<half, int16_t, castTraitInt162Half>(yRegHalf, vregInt16Y, maskReg);
                Pack(vregTmp1Y, (RegTensor<uint32_t>&)yRegHalf);
                Cast<int4x2_t, half, castTraitF162I8>((RegTensor<int4x2_t>&)vregTmp2Y, (RegTensor<half>&)vregTmp1Y,
                                                      maskReg);
                DataCopy<uint8_t, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH / 2, vregTmp2Y, mask4Int4);
            }
        }
    }
}

template <typename T_X, typename T_GAMMA, typename T_SMOOTH_SCALE = float, bool HAS_SMOOTH_SCALE = true,
          bool HAS_BETA = false, typename T_Y, bool Y3_MODE, bool Y4_MODE>
__aicore__ inline void ComputeReduceMax(LocalTensor<float>& scaleLocal, LocalTensor<float>& yTmpLocal,
                                        LocalTensor<T_X>& xLocal, LocalTensor<float>& rstdLocal,
                                        LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal,
                                        LocalTensor<float>& y3Local, LocalTensor<T_GAMMA>& y4Local,
                                        LocalTensor<T_SMOOTH_SCALE>& smoothScaleLocal, uint32_t rstdScaleOffset,
                                        uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* betaAddr;
    if constexpr (HAS_BETA) {
        betaAddr = (__ubuf__ T_GAMMA*)betaLocal.GetPhyAddr();
    }
    __local_mem__ T_SMOOTH_SCALE* smoothScaleAddr;
    if constexpr (HAS_SMOOTH_SCALE) {
        smoothScaleAddr = (__ubuf__ T_SMOOTH_SCALE*)smoothScaleLocal.GetPhyAddr();
    }
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();
    __local_mem__ float* y3Addr = nullptr;
    if constexpr (Y3_MODE) {
        y3Addr = (__ubuf__ float*)y3Local.GetPhyAddr();
    }
    __local_mem__ T_GAMMA* y4Addr = nullptr;
    if constexpr (Y4_MODE) {
        y4Addr = (__ubuf__ T_GAMMA*)y4Local.GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        RegTensor<float> rstdReg, scaleReg, scaleLastReg;
        RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, betaRegFp32, smoothScaleRegFp32;
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
            if constexpr (HAS_BETA) {
                NormCommon::LoadCastRegVF(betaRegFp32, betaAddr, idx, maskReg);
            }
            if constexpr (HAS_SMOOTH_SCALE) {
                NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
            }
            ComputeYAndAbsMaxVF<T_GAMMA, Y3_MODE, Y4_MODE, HAS_BETA, HAS_SMOOTH_SCALE>(
                xRegFp32, yRegFp32, rstdReg, gammaRegFp32, betaRegFp32, smoothScaleRegFp32, scaleReg, maskReg,
                maskRegFull, yTmpAddr, y3Addr, y4Addr, idx);
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
        } else if constexpr (IsSameType<T_Y, int4b_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_INT4, maskRegOne);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
}

template <typename T_Y>
__aicore__ inline void ComputeY(LocalTensor<YCopyDtype<T_Y>>& yLocal, LocalTensor<float>& scaleLocal,
                                LocalTensor<float>& xLocal, uint32_t rstdScaleOffset, uint32_t calCount)
{
    using TyCopy = YCopyDtype<T_Y>;
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ TyCopy* yAddr = (__ubuf__ TyCopy*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
        RegTensor<half> yRegHalf;
        RegTensor<TyCopy> yReg;
        MaskReg maskReg;

        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            DataCopy<float>(yRegFp32, xAddr + idx * V_LENGTH);
            Div(yRegFp32, yRegFp32, scaleReg, maskReg);
            if constexpr (IsSameType<T_Y, int8_t>::value) {
                Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                Cast<half, float, castTraitFp322Fp16>(yRegHalf, yRegFp32Tmp, maskReg);
                Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegHalf, maskReg);
                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
            } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                Cast<T_Y, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
            } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                Cast<T_Y, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
            } else if constexpr (IsSameType<T_Y, int4b_t>::value) {
                RegTensor<int16_t> vregInt16Y;
                RegTensor<uint16_t> vregTmp1Y;
                RegTensor<uint8_t> vregTmp2Y;
                MaskReg mask4Int4 = CreateMask<float, MaskPattern::H>();
                Cast<int16_t, float, castTraitFp322Int16>(vregInt16Y, yRegFp32, maskReg);
                Cast<half, int16_t, castTraitInt162Half>(yRegHalf, vregInt16Y, maskReg);
                Pack(vregTmp1Y, (RegTensor<uint32_t>&)yRegHalf);
                Cast<int4x2_t, half, castTraitF162I8>((RegTensor<int4x2_t>&)vregTmp2Y, (RegTensor<half>&)vregTmp1Y,
                                                      maskReg);
                DataCopy<uint8_t, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH / 2, vregTmp2Y, mask4Int4);
            }
        }
    }
}

} // namespace AddRmsNormDynamicQuant
#endif // _ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_COMMON_H_
