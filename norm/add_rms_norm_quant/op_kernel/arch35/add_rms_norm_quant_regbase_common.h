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
 * \file add_rms_norm_quant_regbase_common.h
 * \brief
 */
#ifndef ADD_RMS_NORM_QUANT_REGBASE_COMMON_H_
#define ADD_RMS_NORM_QUANT_REGBASE_COMMON_H_

#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "../../rms_norm/arch35/rms_norm_regbase_common.h"

namespace AddRmsNormQuant {
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

constexpr int32_t VL_SIZE = GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / sizeof(float));

constexpr AscendC::MicroAPI::CastTrait castTraitF162F32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Int32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitInt322Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
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
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Hifp8 = {
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

template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS, uint64_t TILING_KEY>
__aicore__ inline void SetQuantGlobalBuffers(
    GlobalTensor<T_X>& x1Gm, GlobalTensor<T_X>& x2Gm, GlobalTensor<T_X>& gammaGm, GlobalTensor<T_X>& betaGm,
    GlobalTensor<T_SCALES>& scales1Gm, GlobalTensor<T_SCALES>& scales2Gm,
    GlobalTensor<T_ZEROPOINTS>& zeroPoints1Gm, GlobalTensor<T_ZEROPOINTS>& zeroPoints2Gm,
    GlobalTensor<T_Y>& y1Gm, GlobalTensor<T_Y>& y2Gm, GlobalTensor<T_X>& xGm,
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2,
    GM_ADDR zeroPoints1, GM_ADDR zeroPoints2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
    uint64_t blockIdx, uint64_t mPerCore, uint64_t mCore, uint64_t numN)
{
    constexpr bool HAS_BETA = (((TILING_KEY % 1000)/100) == 1);
    constexpr uint64_t INPUT_KEY = ((TILING_KEY % 100) / 10);
    constexpr bool HAS_ZEROPOINTS1 = ((INPUT_KEY >> 2) % 2 == 1);
    constexpr bool HAS_SCALE2 = ((INPUT_KEY >> 1) % 2 == 1);
    constexpr bool HAS_ZEROPOINTS2 = (INPUT_KEY % 2 == 1);

    uint64_t gmOffset = blockIdx * mPerCore * numN;
    uint64_t gmLen = mCore * numN;
    x1Gm.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
    x2Gm.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
    gammaGm.SetGlobalBuffer((__gm__ T_X*)gamma, numN);
    if constexpr (HAS_BETA) {
        betaGm.SetGlobalBuffer((__gm__ T_X*)beta, numN);
    }
    scales1Gm.SetGlobalBuffer((__gm__ T_SCALES*)scales1, numN);
    y1Gm.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
    y2Gm.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
    xGm.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
    if constexpr (HAS_SCALE2) {
        scales2Gm.SetGlobalBuffer((__gm__ T_SCALES*)scales2, numN);
    }
    if constexpr (HAS_ZEROPOINTS1) {
        zeroPoints1Gm.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints1, numN);
    }
    if constexpr (HAS_ZEROPOINTS2) {
        zeroPoints2Gm.SetGlobalBuffer((__gm__ T_ZEROPOINTS*)zeroPoints2, numN);
    }
}

template <typename T_X, typename T_Y, typename T_SCALES, typename T_ZEROPOINTS, uint64_t TILING_KEY>
__aicore__ inline void InitQuantPipeBuffers(
    TPipe* pipe, TQue<QuePosition::VECIN, 1>& inQueueX1, TQue<QuePosition::VECIN, 1>& inQueueX2,
    TQue<QuePosition::VECOUT, 1>& outQueueX, TQue<QuePosition::VECIN, 1>& inQueueGamma,
    TQue<QuePosition::VECIN, 1>& inQueueBeta, TQue<QuePosition::VECIN, 1>& inQueueScales1,
    TQue<QuePosition::VECIN, 1>& inQueueScales2, TQue<QuePosition::VECIN, 1>& inQueueZeroPoints1,
    TQue<QuePosition::VECIN, 1>& inQueueZeroPoints2, TQue<QuePosition::VECOUT, 1>& outQueueY1,
    TQue<QuePosition::VECOUT, 1>& outQueueY2, TBuf<TPosition::VECCALC>& rstdBuf,
    TBuf<TPosition::VECCALC>& reduceBuf, uint64_t baseNReduceAlign, uint64_t baseNDtypeAlign,
    uint64_t baseNB8Align, uint64_t baseN, uint64_t reduceBufAlign)
{
    constexpr bool HAS_BETA = (((TILING_KEY % 1000)/100) == 1);
    constexpr uint64_t INPUT_KEY = ((TILING_KEY % 100) / 10);
    constexpr bool HAS_ZEROPOINTS1 = ((INPUT_KEY >> 2) % 2 == 1);
    constexpr bool HAS_SCALE2 = ((INPUT_KEY >> 1) % 2 == 1);
    constexpr bool HAS_ZEROPOINTS2 = (INPUT_KEY % 2 == 1);
    constexpr bool HAS_Y2 = (HAS_SCALE2 || HAS_ZEROPOINTS2);

    uint64_t ubFactorQuant = CeilAlign(baseN, BLOCK_SIZE / sizeof(T_SCALES));
    uint64_t ubFactorRstd = B32_BLOCK_NUM;
    pipe->InitBuffer(inQueueX1, 1, baseNReduceAlign * sizeof(T_X));
    pipe->InitBuffer(inQueueX2, 1, baseNReduceAlign * sizeof(T_X));
    pipe->InitBuffer(outQueueX, 1, baseNReduceAlign * sizeof(T_X));
    pipe->InitBuffer(inQueueGamma, 1, baseNDtypeAlign * sizeof(T_X));
    if constexpr (HAS_BETA) {
        pipe->InitBuffer(inQueueBeta, 1, baseNDtypeAlign * sizeof(T_X));
    }
    pipe->InitBuffer(inQueueScales1, 1, ubFactorQuant * sizeof(T_SCALES));
    pipe->InitBuffer(outQueueY1, 1, baseNB8Align * sizeof(T_Y));
    if constexpr (HAS_SCALE2) {
        pipe->InitBuffer(inQueueScales2, 1, ubFactorQuant * sizeof(T_SCALES));
    }
    if constexpr (HAS_ZEROPOINTS1) {
        pipe->InitBuffer(inQueueZeroPoints1, 1, ubFactorQuant * sizeof(T_ZEROPOINTS));
    }
    if constexpr (HAS_ZEROPOINTS2) {
        pipe->InitBuffer(inQueueZeroPoints2, 1, ubFactorQuant * sizeof(T_ZEROPOINTS));
    }
    if constexpr (HAS_Y2) {
        pipe->InitBuffer(outQueueY2, 1, baseNB8Align * sizeof(T_Y));
    }
    pipe->InitBuffer(rstdBuf, ubFactorRstd * sizeof(float));
    pipe->InitBuffer(reduceBuf, reduceBufAlign * sizeof(float));
}

/**
 * @brief y = round2Int8(x * rstd * gamma * scales + zeropoints)
 *        Use muti cast(float32->int32->float32->half->int8) to round
 *        Use float32 VL_LENGTH
 */
template <
    typename T_X, typename T_Y, typename T_SCALES = float, typename T_ZEROPOINTS = float, bool HAS_SCALES = true,
    bool HAS_ZEROPOINTS = false, bool HAS_BETA = false, bool DIV_MODE = true>
__aicore__ inline void ComputeY(
    LocalTensor<T_Y>& yLocal, LocalTensor<T_X>& xLocal, LocalTensor<float>& rstdLocal, LocalTensor<T_X>& gammaLocal,
    LocalTensor<T_X>& betaLocal, LocalTensor<T_SCALES>& scalesLocal, LocalTensor<T_ZEROPOINTS>& zeroPointsLocal, uint32_t rstdOffset, uint32_t count)
{
    uint32_t sreg = (uint32_t)count;
    uint16_t repeatTimes = CeilDivision(count, V_LENGTH);

    __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_X* gammaAddr = (__ubuf__ T_X*)gammaLocal.GetPhyAddr();
    __local_mem__ T_X* betaAddr;
    if constexpr (HAS_BETA) {
        betaAddr = (__ubuf__ T_X*)betaLocal.GetPhyAddr();
    }
    __local_mem__ T_SCALES* scalesAddr = (__ubuf__ T_SCALES*)scalesLocal.GetPhyAddr();
    __local_mem__ T_ZEROPOINTS* zeroPointsAddr = (__ubuf__ T_ZEROPOINTS*)zeroPointsLocal.GetPhyAddr();
    __local_mem__ T_Y* yAddr = (__ubuf__ T_Y*)yLocal.GetPhyAddr();

    if constexpr (IsSameType<T_X, float>::value) {
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, rstdReg, gammaReg, betaReg, scalesReg, zeroPointsReg;
            RegTensor<int32_t> yRegInt32;
            RegTensor<half> yRegFp16;
            RegTensor<T_Y> yReg;
            MaskReg maskReg;
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdOffset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                DataCopy(xReg, xAddr + i * V_LENGTH);
                DataCopy(gammaReg, gammaAddr + i * V_LENGTH);
                if constexpr (HAS_BETA) {
                    DataCopy(betaReg, betaAddr + i * V_LENGTH);
                }
                if constexpr (HAS_SCALES) {
                    DataCopy(scalesReg, scalesAddr + i * V_LENGTH);
                }
                if constexpr (HAS_ZEROPOINTS) {
                    DataCopy(zeroPointsReg, zeroPointsAddr + i * V_LENGTH);
                }
                Mul(xReg, xReg, rstdReg, maskReg);
                Mul(xReg, xReg, gammaReg, maskReg);
                if constexpr (HAS_BETA) {
                    Add(xReg, xReg, betaReg, maskReg);
                }
                if constexpr (HAS_SCALES) {
                    if constexpr (DIV_MODE) {
                        Div(xReg, xReg, scalesReg, maskReg);
                    } else {
                        Mul(xReg, xReg, scalesReg, maskReg);
                    }
                }
                if constexpr (HAS_ZEROPOINTS) {
                    Add(xReg, xReg, zeroPointsReg, maskReg);
                }
                if constexpr (IsSameType<T_Y, int8_t>::value) {
                    Cast<int32_t, float, castTraitFp322Int32>(yRegInt32, xReg, maskReg);
                    Cast<float, int32_t, castTraitInt322Fp32>(xReg, yRegInt32, maskReg);
                    Cast<half, float, castTraitFp322Fp16>(yRegFp16, xReg, maskReg);
                    Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
                } else if constexpr (
                    IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                    Cast<T_Y, float, castTraitFp322Fp8>(yReg, xReg, maskReg);
                } else if constexpr (
                    IsSameType<T_Y, hifloat8_t>::value) {
                    Cast<T_Y, float, castTraitFp322Hifp8>(yReg, xReg, maskReg);
                }

                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + i * V_LENGTH, yReg, maskReg);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<T_X> xReg, gammaReg, betaReg;
            RegTensor<T_SCALES> scalesReg;
            RegTensor<T_ZEROPOINTS> zeroPointsReg;
            RegTensor<float> rstdReg;
            RegTensor<float> xRegFp32, gammaRegFp32, betaRegFp32, scalesRegFp32, zeroPointsRegFp32;
            RegTensor<int32_t> yRegInt32;
            RegTensor<half> yRegFp16;
            RegTensor<T_Y> yReg;
            MaskReg maskReg;
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdOffset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                DataCopy<T_X, LoadDist::DIST_UNPACK_B16>(xReg, xAddr + i * V_LENGTH);
                DataCopy<T_X, LoadDist::DIST_UNPACK_B16>(gammaReg, gammaAddr + i * V_LENGTH);
                if constexpr (HAS_BETA) {
                    DataCopy<T_X, LoadDist::DIST_UNPACK_B16>(betaReg, betaAddr + i * V_LENGTH);
                }
                if constexpr (HAS_SCALES) {
                    if constexpr (IsSameType<T_SCALES, float>::value) {
                        DataCopy(scalesRegFp32, scalesAddr + i * V_LENGTH);
                    } else {
                        DataCopy<T_SCALES, LoadDist::DIST_UNPACK_B16>(scalesReg, scalesAddr + i * V_LENGTH);
                    }
                }
                if constexpr (HAS_ZEROPOINTS) {
                    if constexpr (IsSameType<T_ZEROPOINTS, float>::value) {
                        DataCopy(zeroPointsRegFp32, zeroPointsAddr + i * V_LENGTH);
                    } else if constexpr (IsSameType<T_ZEROPOINTS, int32_t>::value) {
                        DataCopy(zeroPointsReg, zeroPointsAddr + i * V_LENGTH);
                    } else {
                        DataCopy<T_ZEROPOINTS, LoadDist::DIST_UNPACK_B16>(zeroPointsReg, zeroPointsAddr + i * V_LENGTH);
                    }
                }
                Cast<float, T_X, NormCommon::castTraitB162B32>(xRegFp32, xReg, maskReg);
                Cast<float, T_X, NormCommon::castTraitB162B32>(gammaRegFp32, gammaReg, maskReg);
                if constexpr (HAS_BETA) {
                    Cast<float, T_X, NormCommon::castTraitB162B32>(betaRegFp32, betaReg, maskReg);
                }
                if constexpr (!IsSameType<T_SCALES, float>::value && HAS_SCALES) {
                    Cast<float, T_SCALES, NormCommon::castTraitB162B32>(scalesRegFp32, scalesReg, maskReg);
                }
                if constexpr (HAS_ZEROPOINTS) {
                    if constexpr (IsSameType<T_ZEROPOINTS, int32_t>::value) {
                        Cast<float, T_ZEROPOINTS, castTraitInt322Fp32>(zeroPointsRegFp32, zeroPointsReg, maskReg);
                    } else {
                        Cast<float, T_ZEROPOINTS, NormCommon::castTraitB162B32>(
                            zeroPointsRegFp32, zeroPointsReg, maskReg);
                    }
                }
                Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                if constexpr (HAS_BETA) {
                    Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
                }
                if constexpr (HAS_SCALES) {
                    if constexpr (DIV_MODE) {
                        Div(xRegFp32, xRegFp32, scalesRegFp32, maskReg);
                    } else {
                        Mul(xRegFp32, xRegFp32, scalesRegFp32, maskReg);
                    }
                }
                if constexpr (HAS_ZEROPOINTS) {
                    Add(xRegFp32, xRegFp32, zeroPointsRegFp32, maskReg);
                }
                if constexpr (IsSameType<T_Y, int8_t>::value) {
                    Cast<int32_t, float, castTraitFp322Int32>(yRegInt32, xRegFp32, maskReg);
                    Cast<float, int32_t, castTraitInt322Fp32>(xRegFp32, yRegInt32, maskReg);
                    Cast<half, float, castTraitFp322Fp16>(yRegFp16, xRegFp32, maskReg);
                    Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
                } else if constexpr (
                    IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                    Cast<T_Y, float, castTraitFp322Fp8>(yReg, xRegFp32, maskReg);
                } else if constexpr (
                    IsSameType<T_Y, hifloat8_t>::value) {
                    Cast<T_Y, float, castTraitFp322Hifp8>(yReg, xRegFp32, maskReg);
                }
                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + i * V_LENGTH, yReg, maskReg);
            }
        }
    }
}

} // namespace AddRmsNormQuant
#endif // _ADD_RMS_NORM_QUANT_REGBASE_COMMON_H_
