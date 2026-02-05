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
 * \file layer_norm_v4_welford.h
 * \brief
 */

#ifndef LAYER_NORM_V4_WELFORD_H
#define LAYER_NORM_V4_WELFORD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../inc/platform.h"

namespace LayerNormV4 {
using namespace AscendC;

namespace LayerNormV4Regbase {
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310 || __NPU_ARCH == 5102
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
} // namespace LayerNormV4Regbase

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

template <typename T, typename U, typename M>
class LayerNormV4RegbaseWelford {
public:
    __aicore__ inline LayerNormV4RegbaseWelford(const LayerNormV4TilingDataWelford* tilingData, TPipe* pipeIn)
    {
        // Constructor
        pipe_ = pipeIn;
        td_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd)
    {
        // Init
        if (GetBlockIdx() >= td_->numBlocks) {
            return;
        }

        // init global memory
        currentBlockFactor = (GetBlockIdx() == td_->mainBlockCount) ? td_->tailBlockFactor : td_->mainBlockFactor;
        int64_t fmOffset = GetBlockIdx() * td_->mainBlockFactor * td_->N;
        int64_t paramOffset = GetBlockIdx() * td_->mainBlockFactor;
        int64_t fmSize = currentBlockFactor * td_->N;
        int64_t paramSize = currentBlockFactor;

        xGm.SetGlobalBuffer((__gm__ T*)x + fmOffset, fmSize);
        yGm.SetGlobalBuffer((__gm__ T*)y + fmOffset, fmSize);
        meanGm.SetGlobalBuffer((__gm__ M*)mean + paramOffset, paramSize);
        rstdGm.SetGlobalBuffer((__gm__ M*)rstd + paramOffset, paramSize);
        if (td_->nullptrGamma == 0) {
            gammaGm.SetGlobalBuffer((__gm__ U*)gamma, td_->N);
        }
        if (td_->nullptrBeta == 0) {
            betaGm.SetGlobalBuffer((__gm__ U*)beta, td_->N);
        }

        // init local memory
        pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, td_->tileLength * sizeof(T));
        pipe_->InitBuffer(outQueueY, DOUBLE_BUFFER, td_->tileLength * sizeof(T));
        pipe_->InitBuffer(outQueueMean, DOUBLE_BUFFER, AGGREGATION_COUNT * sizeof(float));
        pipe_->InitBuffer(outQueueRstd, DOUBLE_BUFFER, AGGREGATION_COUNT * sizeof(float));
        pipe_->InitBuffer(welfordTempBuffer, td_->welfordTempSize);
        if (td_->nullptrGamma == 0) {
            pipe_->InitBuffer(inQueueGamma, DOUBLE_BUFFER, td_->tileLength * sizeof(U));
        }
        if (td_->nullptrBeta == 0) {
            pipe_->InitBuffer(inQueueBeta, DOUBLE_BUFFER, td_->tileLength * sizeof(U));
        }
        pipe_->InitBuffer(apiTempBuffer, td_->apiTempBufferSize);
    }

    __aicore__ inline void Process()
    {
        // Process
        if (GetBlockIdx() >= td_->numBlocks) {
            return;
        }

        ResetCache();
        paramAddr = 0;
        mean_ = welfordTempBuffer.Get<float>();
        variance_ = mean_[td_->tileLength];
        varianceTensor = variance_[td_->tileLength];
        shared_ = apiTempBuffer.Get<uint8_t>();

        for (int64_t i = 0; i < currentBlockFactor; ++i) {
            // Welford Algorithm
            WelfordInitialize(mean_, variance_, td_->tileLength);
            for (int64_t welfordUpdateCount = 0; welfordUpdateCount < td_->welfordUpdateTimes; welfordUpdateCount++) {
                int64_t offset = i * td_->N + welfordUpdateCount * td_->tileLength;
                ProcessWelfordUpdate(offset, td_->tileLength);
            }
            if (td_->welfordUpdateTail > 0) {
                int64_t offset = i * td_->N + td_->welfordUpdateTimes * td_->tileLength;
                ProcessWelfordUpdate(offset, td_->welfordUpdateTail);
            }

            WelfordFinalizePara para;
            para.rnLength = welfordCount;
            para.abLength = td_->tileLength;
            para.headCount = welfordCount;
            para.headCountLength = td_->welfordUpdateTail;
            para.tailCount = welfordCount - (td_->welfordUpdateTail > 0);
            para.tailCountLength = td_->tileLength - td_->welfordUpdateTail;
            para.abRec = 1.0f / static_cast<float>(td_->tileLength);
            para.rRec = 1.0f / static_cast<float>(td_->N);
            WelfordFinalize<true>(
                meanTensor[paramAddr + cacheCount], varianceTensor[paramAddr + cacheCount], mean_, variance_, shared_,
                para);

            // Normalize
            for (int64_t welfordUpdateCount = 0; welfordUpdateCount < td_->welfordUpdateTimes; welfordUpdateCount++) {
                int64_t fmOffset = i * td_->N + welfordUpdateCount * td_->tileLength;
                int64_t paramOffset = welfordUpdateCount * td_->tileLength;
                ProcessNormalize(fmOffset, paramOffset, td_->tileLength);
            }
            if (td_->welfordUpdateTail > 0) {
                int64_t fmOffset = i * td_->N + td_->welfordUpdateTimes * td_->tileLength;
                int64_t paramOffset = td_->welfordUpdateTimes * td_->tileLength;
                ProcessNormalize(fmOffset, paramOffset, td_->welfordUpdateTail);
            }

            // check cache buffer
            if (cacheCount >= AGGREGATION_COUNT) {
                RefreshCache();
                ResetCache();
            } else {
                cacheCount++;
            }
        }

        // refresh cache
        if (cacheCount > 0) {
            RefreshCache();
        }
    }

private:
    __aicore__ inline void ResetCache()
    {
        // ResetCache
        cacheCount = 0;
        meanTensor = outQueueMean.template AllocTensor<float>();
        rstdTensor = outQueueRstd.template AllocTensor<float>();
    }

    __aicore__ inline void CastBatchMeanRstd(uint64_t currentANum)
    {
        __local_mem__ float* batchMeanInAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdInAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __local_mem__ M* batchMeanOutAddr = (__local_mem__ M*)meanTensor.GetPhyAddr();
        __local_mem__ M* batchRstdOutAddr = (__local_mem__ M*)rstdTensor.GetPhyAddr();

        uint32_t castCount = static_cast<uint32_t>(currentANum);
        uint16_t castLoops = static_cast<uint32_t>((castCount + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> input_mean;
            RegTensor<float> input_rstd;
            RegTensor<M> output_mean;
            RegTensor<M> output_rstd;
            MicroAPI::MaskReg pregLoop;
            for (uint16_t i = 0; i < castLoops; i++) {
                pregLoop = MicroAPI::UpdateMask<float>(castCount);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_mean, batchMeanInAddr + VL_F32 * i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_rstd, batchRstdInAddr + VL_F32 * i);
                Cast<M, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
                Cast<M, float, castTraitB322B16>(output_rstd, input_rstd, pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ M*)batchMeanOutAddr + i * VL_MEAN), output_mean, pregLoop);
                DataCopy<M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ M*)batchRstdOutAddr + i * VL_MEAN), output_rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void RefreshCache()
    {
        if constexpr (!IsSameType<M, float>::value) {
            // float to bfloat16 or float16, input continue and output each repeat have only half value
            CastBatchMeanRstd(cacheCount);
            outQueueMean.EnQue(meanTensor);
            outQueueRstd.EnQue(rstdTensor);
            meanTensor = outQueueMean.template DeQue<float>();
            rstdTensor = outQueueRstd.template DeQue<float>();

            // VL_F32
            uint32_t castDmaCount = static_cast<uint32_t>(cacheCount);
            uint32_t castDmaLoops = static_cast<uint32_t>(castDmaCount / VL_F32);
            if (castDmaLoops > 0) {
                DataCopyExtParams copyInParams;
                copyInParams.blockCount = castDmaLoops;
                copyInParams.blockLen = VL_F32 * sizeof(M);
                copyInParams.srcStride = (VECTOR_REG_WIDTH - VL_F32 * sizeof(M)) / BLOCK_SIZE;
                copyInParams.dstStride = 0;
                DataCopyPad(meanGm[paramAddr], meanTensor.ReinterpretCast<M>(), copyInParams);
                DataCopyPad(rstdGm[paramAddr], rstdTensor.ReinterpretCast<M>(), copyInParams);
            }

            // tail
            uint32_t tailSize = static_cast<uint32_t>(castDmaCount % VL_F32);
            if (tailSize > 0) {
                DataCopyExtParams copyInParamsTail;
                copyInParamsTail.blockCount = 1;
                copyInParamsTail.blockLen = tailSize * sizeof(M);
                copyInParamsTail.srcStride = 0;
                copyInParamsTail.dstStride = 0;
                DataCopyPad(
                    meanGm[paramAddr + castDmaLoops * VL_F32], meanTensor[castDmaLoops * VL_MEAN].ReinterpretCast<M>(),
                    copyInParamsTail);
                DataCopyPad(
                    rstdGm[paramAddr + castDmaLoops * VL_F32], rstdTensor[castDmaLoops * VL_MEAN].ReinterpretCast<M>(),
                    copyInParamsTail);
            }
            outQueueMean.FreeTensor(meanTensor);
            outQueueRstd.FreeTensor(rstdTensor);
        } else {
            // Refresh Cache
            outQueueMean.EnQue(meanTensor);
            meanTensor = outQueueMean.template DeQue<float>();
            CopyOut(meanGm[paramAddr], meanTensor, cacheCount);
            outQueueMean.FreeTensor(meanTensor);

            outQueueRstd.EnQue(rstdTensor);
            rstdTensor = outQueueRstd.template DeQue<float>();
            CopyOut(rstdGm[paramAddr], rstdTensor, cacheCount);
            outQueueRstd.FreeTensor(rstdTensor);
        }
        paramAddr += cacheCount;
    }

    __aicore__ inline void WelfordInitialize(
        const LocalTensor<float>& mean, const LocalTensor<float>& variance, const int64_t elemCnt)
    {
        // WelfordInitialize
        welfordCount = 0;
        constexpr static uint32_t VL_B32 = LayerNormV4Regbase::GetVRegSize() / sizeof(float);
        uint16_t loopTimes = (elemCnt + VL_B32 - 1) / VL_B32;
        __VEC_SCOPE__
        {
            __local_mem__ float* meamPtr = (__local_mem__ float*)mean.GetPhyAddr();
            __local_mem__ float* variancePtr = (__local_mem__ float*)variance.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(elemCnt);
            AscendC::MicroAPI::RegTensor<float> xReg;
            AscendC::MicroAPI::MaskReg pMask;
            Duplicate(xReg, 0.0f);
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                DataCopy((__local_mem__ float*)meamPtr + i * VL_B32, xReg, pMask);
                DataCopy((__local_mem__ float*)variancePtr + i * VL_B32, xReg, pMask);
            }
        }
    }

    __aicore__ inline void ProcessWelfordUpdate(const int64_t offset, const int64_t elemCnt)
    {
        // ProcessWelfordUpdate
        welfordCount++;
        xTensor = inQueueX.template AllocTensor<T>();
        CopyIn(xTensor, xGm[offset], elemCnt);
        inQueueX.EnQue(xTensor);
        xTensor = inQueueX.template DeQue<T>();

        WelfordUpdateParam para;
        para.rnLength = 1;
        para.abLength = td_->tileLength;
        para.abComputeLength = elemCnt;
        para.nRec = 1.0f / static_cast<float>(welfordCount);
        WelfordUpdate<T, float, false>(mean_, variance_, mean_, variance_, xTensor, shared_, para);

        inQueueX.FreeTensor(xTensor);
    }

    __aicore__ inline void ProcessNormalize(const int64_t fmOffset, const int64_t paramOffset, const int64_t elemCnt)
    {
        // ProcessNormalize
        xTensor = inQueueX.template AllocTensor<T>();
        CopyIn(xTensor, xGm[fmOffset], elemCnt);
        inQueueX.EnQue(xTensor);
        xTensor = inQueueX.template DeQue<T>();

        if (td_->nullptrGamma == 0) {
            gammaTensor = inQueueGamma.template AllocTensor<U>();
            CopyIn(gammaTensor, gammaGm[paramOffset], elemCnt);
            inQueueGamma.EnQue(gammaTensor);
            gammaTensor = inQueueGamma.template DeQue<U>();
        }
        if (td_->nullptrBeta == 0) {
            betaTensor = inQueueBeta.template AllocTensor<U>();
            CopyIn(betaTensor, betaGm[paramOffset], elemCnt);
            inQueueBeta.EnQue(betaTensor);
            betaTensor = inQueueBeta.template DeQue<U>();
        }

        yTensor = outQueueY.template AllocTensor<T>();

        NormalizePara para;
        para.aLength = 1;
        para.rLength = elemCnt;
        para.rLengthWithPadding = elemCnt;

        if (td_->nullptrGamma == 0 && td_->nullptrBeta == 0) {
            constexpr static NormalizeConfig hasGammaBetaNormalizeConfig = {
                ReducePattern::AR,-1,false,false,false
            };
            Normalize<U, T, false, hasGammaBetaNormalizeConfig>(
                yTensor,rstdTensor[paramAddr + cacheCount], meanTensor[paramAddr + cacheCount], varianceTensor[paramAddr + cacheCount], 
                xTensor, gammaTensor, betaTensor, shared_, td_->epsilon, para);
        } else if (td_->nullptrGamma == 0 && !td_->nullptrBeta == 0) {
            constexpr static NormalizeConfig hasGammaNoBetaNormalizeConfig = {
                ReducePattern::AR,-1,true,false,false
            };
            Normalize<U, T, false, hasGammaNoBetaNormalizeConfig>(
                yTensor,rstdTensor[paramAddr + cacheCount], meanTensor[paramAddr + cacheCount], varianceTensor[paramAddr + cacheCount], 
                xTensor, gammaTensor, betaTensor, shared_, td_->epsilon, para);
        } else if (!td_->nullptrGamma == 0 && td_->nullptrBeta == 0) {
            constexpr static NormalizeConfig noGammaHasBetaNormalizeConfig = {
                ReducePattern::AR,-1,false,true,false
            };
            Normalize<U, T, false, noGammaHasBetaNormalizeConfig>(
                yTensor,rstdTensor[paramAddr + cacheCount], meanTensor[paramAddr + cacheCount], varianceTensor[paramAddr + cacheCount], 
                xTensor, gammaTensor, betaTensor, shared_, td_->epsilon, para);
        } else if (!td_->nullptrGamma == 0 && !td_->nullptrBeta == 0) {
            constexpr static NormalizeConfig noGammaNoBetaNormalizeConfig = {
                ReducePattern::AR,-1,true,true,false
            };
            Normalize<U, T, false, noGammaNoBetaNormalizeConfig>(
                yTensor,rstdTensor[paramAddr + cacheCount], meanTensor[paramAddr + cacheCount], varianceTensor[paramAddr + cacheCount], 
                xTensor, gammaTensor, betaTensor, shared_, td_->epsilon, para);
        }

        inQueueX.FreeTensor(xTensor);
        if (td_->nullptrGamma == 0) {
            inQueueGamma.FreeTensor(gammaTensor);
        }
        if (td_->nullptrBeta == 0) {
            inQueueBeta.FreeTensor(betaTensor);
        }

        outQueueY.EnQue(yTensor);
        yTensor = outQueueY.template DeQue<T>();
        CopyOut(yGm[fmOffset], yTensor, elemCnt);
        outQueueY.FreeTensor(yTensor);
    }

private:
    TPipe* pipe_;
    const LayerNormV4TilingDataWelford* __restrict td_;

    // Constants
    constexpr static int64_t DOUBLE_BUFFER = 2;
    constexpr static int64_t AGGREGATION_COUNT = 256;
    constexpr static uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    constexpr static uint32_t VL_MEAN = VECTOR_REG_WIDTH / sizeof(M);
    constexpr static int64_t BLOCK_SIZE = 32;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;

    TQue<QuePosition::VECOUT, 1> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    TBuf<TPosition::VECCALC> welfordTempBuffer;
    TBuf<TPosition::VECCALC> apiTempBuffer;

    // Global Tensor
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<M> meanGm;
    GlobalTensor<M> rstdGm;
    GlobalTensor<U> gammaGm;
    GlobalTensor<U> betaGm;

    // Local Tensor
    LocalTensor<T> xTensor;
    LocalTensor<T> yTensor;
    LocalTensor<U> gammaTensor;
    LocalTensor<U> betaTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<float> varianceTensor;
    LocalTensor<float> mean_;
    LocalTensor<float> variance_;
    LocalTensor<uint8_t> shared_;

    // params
    int64_t currentBlockFactor = 0;
    int64_t cacheCount = 0;
    int64_t paramAddr = 0;
    int64_t welfordCount = 0;
};

} // namespace LayerNormV4

#endif // LAYER_NORM_V4_WELFORD_H
