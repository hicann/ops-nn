/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_v3_common.h
 * \brief
 */

#ifndef LAYER_NORM_V3_COMMON_H
#define LAYER_NORM_V3_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../inc/platform.h"

namespace LayerNormV3 {
using namespace AscendC;

constexpr static int64_t BLOCK_SIZE = 32;
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static int64_t MAX_STRIDE = 65535;
constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int64_t LOOP_STRIDE_HIGH_SHIFT = 40;
constexpr static int64_t LOOP_HIGH_SHIFT = 21;
constexpr static uint32_t NUM_TWO = 2;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static uint32_t VL_B32 = GetVecLen() / sizeof(float);
constexpr static float zero = 0.0f;
constexpr static uint32_t VL_B16 = GetVecLen() / sizeof(int16_t);
constexpr static uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, const int64_t rowSize)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = rowSize * sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const int64_t rowSize)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = rowSize * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

__aicore__ inline void WelfordInitialize(const LocalTensor<float>& mean, const LocalTensor<float>& variance,
                                         const int64_t elemCnt)
{
    constexpr static uint32_t VL_B32 = platform::GetVRegSize() / sizeof(float);
    uint16_t loopTimes = (elemCnt + VL_B32 - 1) / VL_B32;
    __VEC_SCOPE__
    {
        __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
        __local_mem__ float* variancePtr = (__local_mem__ float*)variance.GetPhyAddr();
        uint32_t count = static_cast<uint32_t>(elemCnt);
        MicroAPI::RegTensor<float> xReg;
        MicroAPI::MaskReg pMask;
        Duplicate(xReg, 0.0f);
        for (uint16_t i = 0; i < loopTimes; ++i) {
            pMask = MicroAPI::UpdateMask<float>(count);
            DataCopy(meanPtr + i * VL_B32, xReg, pMask);
            DataCopy(variancePtr + i * VL_B32, xReg, pMask);
        }
    }
}

template <typename T>
__aicore__ inline void ProcessWelfordUpdate(TQue<QuePosition::VECIN, 1>& inQueueX, const GlobalTensor<T>& xGm,
                                            const int64_t offset, const int64_t elemCnt, const int64_t tileLength,
                                            int64_t& welfordCount, const LocalTensor<float>& mean,
                                            const LocalTensor<float>& variance, const LocalTensor<uint8_t>& shared)
{
    welfordCount++;
    LocalTensor<T> xTensor = inQueueX.template AllocTensor<T>();
    CopyIn(xTensor, xGm[offset], elemCnt);
    inQueueX.EnQue(xTensor);
    xTensor = inQueueX.template DeQue<T>();

    WelfordUpdateParam para;
    para.rnLength = 1;
    para.abLength = tileLength;
    para.abComputeLength = elemCnt;
    para.nRec = 1.0f / static_cast<float>(welfordCount);
    WelfordUpdate<T, float, false>(mean, variance, mean, variance, xTensor, shared, para);

    inQueueX.FreeTensor(xTensor);
}

template <typename M>
__aicore__ inline void CastBatchMeanLastout(LocalTensor<float>& meanTensor, LocalTensor<float>& lastoutTensor,
                                            uint64_t currentANum)
{
    constexpr static uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    constexpr static uint32_t VL_MEAN = VECTOR_REG_WIDTH / sizeof(M);

    __local_mem__ float* batchMeanInAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
    __local_mem__ float* batchLastoutInAddr = (__local_mem__ float*)lastoutTensor.GetPhyAddr();
    __local_mem__ M* batchMeanOutAddr = (__local_mem__ M*)meanTensor.GetPhyAddr();
    __local_mem__ M* batchLastoutOutAddr = (__local_mem__ M*)lastoutTensor.GetPhyAddr();

    uint32_t castCount = static_cast<uint32_t>(currentANum);
    uint16_t castLoops = static_cast<uint32_t>((castCount + VL_F32 - 1) / VL_F32);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> input_mean;
        MicroAPI::RegTensor<float> input_lastout;
        MicroAPI::RegTensor<M> output_mean;
        MicroAPI::RegTensor<M> output_lastout;
        MicroAPI::MaskReg pregLoop;
        for (uint16_t i = 0; i < castLoops; i++) {
            pregLoop = MicroAPI::UpdateMask<float>(castCount);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_mean, batchMeanInAddr + VL_F32 * i);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_lastout, batchLastoutInAddr + VL_F32 * i);
            Cast<M, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
            Cast<M, float, castTraitB322B16>(output_lastout, input_lastout, pregLoop);
            MicroAPI::DataCopy<M, MicroAPI::StoreDist::DIST_PACK_B32>(
                ((__local_mem__ M*)batchMeanOutAddr + i * VL_MEAN), output_mean, pregLoop);
            MicroAPI::DataCopy<M, MicroAPI::StoreDist::DIST_PACK_B32>(
                ((__local_mem__ M*)batchLastoutOutAddr + i * VL_MEAN), output_lastout, pregLoop);
        }
    }
}

template <typename M>
__aicore__ inline void RefreshCache(int64_t& cacheCount, int64_t& paramAddr, LocalTensor<float>& meanTensor,
                                    LocalTensor<float>& lastoutTensor, TQue<QuePosition::VECOUT, 1>& outQueueMean,
                                    TQue<QuePosition::VECOUT, 1>& outQueueLastout, const GlobalTensor<M>& meanGm,
                                    const GlobalTensor<M>& lastoutGm)
{
    constexpr static uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    constexpr static uint32_t VL_MEAN = VECTOR_REG_WIDTH / sizeof(M);

    if constexpr (!IsSameType<M, float>::value) {
        CastBatchMeanLastout<M>(meanTensor, lastoutTensor, cacheCount);
        outQueueMean.EnQue(meanTensor);
        outQueueLastout.EnQue(lastoutTensor);
        meanTensor = outQueueMean.template DeQue<float>();
        lastoutTensor = outQueueLastout.template DeQue<float>();

        uint32_t castDmaCount = static_cast<uint32_t>(cacheCount);
        uint32_t castDmaLoops = static_cast<uint32_t>(castDmaCount / VL_F32);
        if (castDmaLoops > 0) {
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = castDmaLoops;
            copyInParams.blockLen = VL_F32 * sizeof(M);
            copyInParams.srcStride = (VECTOR_REG_WIDTH - VL_F32 * sizeof(M)) / BLOCK_SIZE;
            copyInParams.dstStride = 0;
            DataCopyPad(meanGm[paramAddr], meanTensor.template ReinterpretCast<M>(), copyInParams);
            DataCopyPad(lastoutGm[paramAddr], lastoutTensor.template ReinterpretCast<M>(), copyInParams);
        }

        uint32_t tailSize = static_cast<uint32_t>(castDmaCount % VL_F32);
        if (tailSize > 0) {
            DataCopyExtParams copyInParamsTail;
            copyInParamsTail.blockCount = 1;
            copyInParamsTail.blockLen = tailSize * sizeof(M);
            copyInParamsTail.srcStride = 0;
            copyInParamsTail.dstStride = 0;
            DataCopyPad(meanGm[paramAddr + castDmaLoops * VL_F32],
                        meanTensor[castDmaLoops * VL_MEAN].template ReinterpretCast<M>(), copyInParamsTail);
            DataCopyPad(lastoutGm[paramAddr + castDmaLoops * VL_F32],
                        lastoutTensor[castDmaLoops * VL_MEAN].template ReinterpretCast<M>(), copyInParamsTail);
        }
        outQueueMean.FreeTensor(meanTensor);
        outQueueLastout.FreeTensor(lastoutTensor);
    } else {
        outQueueMean.EnQue(meanTensor);
        meanTensor = outQueueMean.template DeQue<float>();
        CopyOut(meanGm[paramAddr], meanTensor, cacheCount);
        outQueueMean.FreeTensor(meanTensor);

        outQueueLastout.EnQue(lastoutTensor);
        lastoutTensor = outQueueLastout.template DeQue<float>();
        CopyOut(lastoutGm[paramAddr], lastoutTensor, cacheCount);
        outQueueLastout.FreeTensor(lastoutTensor);
    }
    paramAddr += cacheCount;
}

} // namespace LayerNormV3

#endif // LAYER_NORM_V3_COMMON_H
