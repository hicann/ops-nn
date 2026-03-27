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
 * \file sync_batch_norm_gather_stats_n_full_load.h
 * \brief
 */

#ifndef __SYNC_BATCH_NORM_GATHER_STATS_FULL_REDUCE_H__
#define __SYNC_BATCH_NORM_GATHER_STATS_FULL_REDUCE_H__

#include "sync_batch_norm_gather_stats_struct.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace SyncBatchNormGatherStats
{
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

template<typename T>
class SyncBatchNormGatherStatsNFullLoad
{
public:
    __aicore__ inline SyncBatchNormGatherStatsNFullLoad(const SyncBatchNormGatherStatsTilingData* tilingData, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->blockDim = tilingData->blockDim;
        this->blockFormer = tilingData->blockFormer;
        this->nLen = tilingData->nLen;
        this->cLen = tilingData->cLen;
        this->ubFormer = tilingData->ubFormer;
        this->momentum = tilingData->momentum;
        this->eps = tilingData->eps;

        if (GetBlockIdx() == (this->blockDim - 1)) {
            this->blockLoopNum = tilingData->blockTail;
            this->ubTail = tilingData->ubTail;
        } else {
            this->blockLoopNum = tilingData->blockFormer;
            this->ubTail = tilingData->ubFormer;
        }
    }

    __aicore__ inline void Init(GM_ADDR total_sum, GM_ADDR total_square_sum, GM_ADDR sample_count, GM_ADDR running_mean, 
                                GM_ADDR running_var, GM_ADDR batch_mean, GM_ADDR batch_invstd, GM_ADDR running_mean_update, 
                                GM_ADDR running_var_update)
    {
        auto blockIdx = GetBlockIdx();
        
        int64_t cGmOffset = this->blockFormer * this->ubFormer * blockIdx;
        int64_t nAlignLen = (this->nLen + (BLOCK_SIZE / sizeof(int32_t)) - 1) / (BLOCK_SIZE / sizeof(int32_t)) * (BLOCK_SIZE / sizeof(int32_t));

        sumGm.SetGlobalBuffer((__gm__ T*)total_sum + cGmOffset);
        squareSumGm.SetGlobalBuffer((__gm__ T*)total_square_sum + cGmOffset);
        countGm.SetGlobalBuffer((__gm__ int32_t*)sample_count);
        runningMeanInGm.SetGlobalBuffer((__gm__ T*)running_mean + cGmOffset);
        runningVarInGm.SetGlobalBuffer((__gm__ T*)running_var + cGmOffset);

        batchMeanGm.SetGlobalBuffer((__gm__ T*)batch_mean + cGmOffset);
        batchRstdGm.SetGlobalBuffer((__gm__ T*)batch_invstd + cGmOffset);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T*)running_mean_update + cGmOffset);
        runningVarOutGm.SetGlobalBuffer((__gm__ T*)running_var_update + cGmOffset);

        pipe->InitBuffer(sumQueue, DOUBLE_BUFFER, this->nLen * this->ubFormer * sizeof(T));
        pipe->InitBuffer(squareSumQueue, DOUBLE_BUFFER, this->nLen * this->ubFormer * sizeof(T));
        pipe->InitBuffer(countQueue, 1, nAlignLen * sizeof(int32_t));
        pipe->InitBuffer(runningMeanInQueue, DOUBLE_BUFFER, this->ubFormer * sizeof(T));
        pipe->InitBuffer(runningVarInQueue, DOUBLE_BUFFER, this->ubFormer * sizeof(T));

        pipe->InitBuffer(batchMeanQueue, DOUBLE_BUFFER, this->ubFormer * sizeof(T));
        pipe->InitBuffer(batchRstdQueue, DOUBLE_BUFFER, this->ubFormer * sizeof(T));
        pipe->InitBuffer(runningMeanOutQueue, DOUBLE_BUFFER, this->ubFormer * sizeof(T));
        pipe->InitBuffer(runningVarOutQueue, DOUBLE_BUFFER, this->ubFormer * sizeof(T));

        pipe->InitBuffer(castCountBuff, nAlignLen * sizeof(int64_t));
        pipe->InitBuffer(sumAllBuff, this->ubFormer * sizeof(float));
        pipe->InitBuffer(squareSumAllBuff, this->ubFormer * sizeof(float));
        pipe->InitBuffer(countAllBuff, sizeof(int64_t));

        if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, bfloat16_t>::value) {
            pipe->InitBuffer(castSumBuff, this->ubFormer * sizeof(float));
            pipe->InitBuffer(castSquareSumBuff, this->ubFormer * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        CopyInN();
        LocalTensor<int32_t> countUb = countQueue.DeQue<int32_t>();
        LocalTensor<int64_t> castCountUb = castCountBuff.Get<int64_t>();
        LocalTensor<int64_t> countAllUb = countAllBuff.Get<int64_t>();

        Cast(castCountUb, countUb, RoundMode::CAST_NONE, this->nLen);
        uint32_t shape[] = { 1, static_cast<uint32_t>(this->nLen)};
        ReduceSum<int64_t, AscendC::Pattern::Reduce::AR, true>(countAllUb, castCountUb, shape, false);
        countQueue.FreeTensor(countUb);

        int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDSToV);
        WaitFlag<HardEvent::V_S>(eventIDSToV);
        int64_t int64CountAll = countAllUb.GetValue(0);
        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDVToS);
        WaitFlag<HardEvent::S_V>(eventIDVToS);

        for (int64_t i = 0; i < this->blockLoopNum - 1; i++) {
            ProcessOnce(i, this->ubFormer, int64CountAll);
        }

        ProcessOnce(this->blockLoopNum - 1, this->ubTail, int64CountAll);
    }

private:
    __aicore__ inline void ProcessOnce(int64_t i, int64_t currentCNum, int64_t int64CountAll)
    {
        CopyInSumAndSquare(i * this->ubFormer, currentCNum);
        CopyInRunningMeanVar(i * this->ubFormer, currentCNum);
        LocalTensor<T> sumUb = sumQueue.DeQue<T>();
        LocalTensor<T> squareSumUb = squareSumQueue.DeQue<T>();
        LocalTensor<float> sumAllUb = sumAllBuff.Get<float>();
        LocalTensor<float> squareSumAllUb = squareSumAllBuff.Get<float>();

        int64_t currentCAlignNum = (currentCNum + (BLOCK_SIZE / sizeof(T)) - 1) / (BLOCK_SIZE / sizeof(T)) * (BLOCK_SIZE / sizeof(T));
        uint32_t shape[] = { static_cast<uint32_t>(this->nLen), static_cast<uint32_t>(currentCAlignNum)};
        if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> fp32SumUb = castSumBuff.Get<float>();
            Cast(fp32SumUb, sumUb, RoundMode::CAST_NONE, this->nLen * currentCAlignNum);
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumAllUb, fp32SumUb, shape, false);

            LocalTensor<float> fp32SquareSumUb = castSquareSumBuff.Get<float>();
            Cast(fp32SquareSumUb, squareSumUb, RoundMode::CAST_NONE, this->nLen * currentCAlignNum);
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(squareSumAllUb, fp32SquareSumUb, shape, false);
        } else {
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumAllUb, sumUb, shape, false);
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(squareSumAllUb, squareSumUb, shape, false);
        }

        sumQueue.FreeTensor(sumUb);
        squareSumQueue.FreeTensor(squareSumUb);

        LocalTensor<T> runningMeanUb = runningMeanInQueue.DeQue<T>();
        LocalTensor<T> runningVarUb = runningVarInQueue.DeQue<T>();

        __local_mem__ float* sumAllLocal = (__local_mem__ float*)sumAllUb.GetPhyAddr();
        __local_mem__ float* squareSumAllLocal = (__local_mem__ float*)squareSumAllUb.GetPhyAddr();
        __local_mem__ T* runningMeanLocal = (__local_mem__ T*)runningMeanUb.GetPhyAddr();
        __local_mem__ T* runningVarLocal = (__local_mem__ T*)runningVarUb.GetPhyAddr();

        LocalTensor<T> batchMeanUb = batchMeanQueue.AllocTensor<T>();
        LocalTensor<T> batchVarUb = batchRstdQueue.AllocTensor<T>();
        LocalTensor<T> runningMeanOutUb = runningMeanOutQueue.AllocTensor<T>();
        LocalTensor<T> runningVarOutUb = runningVarOutQueue.AllocTensor<T>();

        __local_mem__ T* batchMeanLocal = (__local_mem__ T*)batchMeanUb.GetPhyAddr();
        __local_mem__ T* batchVarLocal = (__local_mem__ T*)batchVarUb.GetPhyAddr();
        __local_mem__ T* runningMeanOutLocal = (__local_mem__ T*)runningMeanOutUb.GetPhyAddr();
        __local_mem__ T* runningVarOutLocal = (__local_mem__ T*)runningVarOutUb.GetPhyAddr();

        NFullLoadBatchNormVF(this->momentum, this->eps, currentCNum, int64CountAll,
                            sumAllLocal, squareSumAllLocal, runningMeanLocal, runningVarLocal,
                            batchMeanLocal, batchVarLocal, runningMeanOutLocal, runningVarOutLocal);

        runningMeanInQueue.FreeTensor(runningMeanUb);
        runningVarInQueue.FreeTensor(runningVarUb);
        
        batchMeanQueue.EnQue(batchMeanUb);
        batchRstdQueue.EnQue(batchVarUb);
        runningMeanOutQueue.EnQue(runningMeanOutUb);
        runningVarOutQueue.EnQue(runningVarOutUb);
        CopyOut(i * this->ubFormer, currentCNum);
    }

    __aicore__ inline void NFullLoadBatchNormVF(float momentum, float eps, int64_t currentCNum, int64_t int64CountAll,
                                        __local_mem__ float* sumAllLocal, __local_mem__ float* squareSumAllLocal, 
                                        __local_mem__ T* runningMeanLocal, __local_mem__ T* runningVarLocal,
                                        __local_mem__ T* batchMeanLocal, __local_mem__ T* batchVarLocal,
                                        __local_mem__ T* runningMeanOutLocal, __local_mem__ T* runningVarOutLocal)
    {
        float float32CountAll = static_cast<float>(int64CountAll);
        float reciCountAll = float(1.0) / float32CountAll;
        float invCountAllMom = float32CountAll / (float32CountAll - 1) * momentum;
        float oneSubMomentum = float(1.0) - momentum;
        uint16_t cLoopCount = ops::CeilDiv(currentCNum, static_cast<int64_t>(VL_FP32));
        uint32_t sreg = currentCNum;

        __VEC_SCOPE__
        {
            RegTensor<float> sumAll;    
            RegTensor<float> squareSum;
            RegTensor<float> runningMean;
            RegTensor<float> runningVar;

            MaskReg pregLoop;

            for (uint16_t cIndex = 0; cIndex < cLoopCount; cIndex++) {
                pregLoop = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(sumAll, sumAllLocal + cIndex * VL_FP32);
                Muls(sumAll, sumAll, reciCountAll, pregLoop);
                StoreTensorForDtypeT(batchMeanLocal + cIndex * VL_FP32, sumAll, pregLoop);
                Mul(runningVar, sumAll, sumAll, pregLoop);
                LoadTensorForDtypeT(runningMean, runningMeanLocal + cIndex * VL_FP32, pregLoop);
                Muls(sumAll, sumAll, momentum, pregLoop);
                Muls(runningMean, runningMean, oneSubMomentum, pregLoop);
                Add(runningMean, runningMean, sumAll, pregLoop);
                StoreTensorForDtypeT(runningMeanOutLocal + cIndex * VL_FP32, runningMean, pregLoop);
                DataCopy<float, LoadDist::DIST_NORM>(squareSum, squareSumAllLocal + cIndex * VL_FP32);
                Muls(squareSum, squareSum, reciCountAll, pregLoop);
                Sub(runningVar, squareSum, runningVar, pregLoop);
                Adds(sumAll, runningVar, eps, pregLoop);
                Sqrt(sumAll, sumAll, pregLoop);
                Duplicate(squareSum, float(1.0), pregLoop);
                Div(sumAll, squareSum, sumAll, pregLoop);
                StoreTensorForDtypeT(batchVarLocal + cIndex * VL_FP32, sumAll, pregLoop);
                Muls(runningVar, runningVar, invCountAllMom, pregLoop);
                LoadTensorForDtypeT(squareSum, runningVarLocal + cIndex * VL_FP32, pregLoop);
                Muls(squareSum, squareSum, oneSubMomentum, pregLoop);
                Add(runningVar, runningVar, squareSum, pregLoop);
                StoreTensorForDtypeT(runningVarOutLocal + cIndex * VL_FP32, runningVar, pregLoop);
            }  
        }
    }

    __aicore__ inline void LoadTensorForDtypeT(RegTensor<float>& dst, __local_mem__ T* src, MaskReg& preg)
    {
        if constexpr (IsSameType<T, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src);
        } else {  // fp16、bf16
            RegTensor<T> xFp16;
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src));
            Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void StoreTensorForDtypeT(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg)
    {
        if constexpr (IsSameType<T, float>::value) {
            DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst, src, preg);
        } else {
            AscendC::MicroAPI::RegTensor<T> xFp16;
            Cast<T, float, castTraitB32B16>(xFp16, src, preg);
            DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void CopyInN()
    {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = this->nLen * sizeof(int32_t);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> dataCopyPadExtParams {false, 0, 0, 0};

        LocalTensor<int32_t> countUb = countQueue.AllocTensor<int32_t>();
        DataCopyPad(countUb, countGm, dataCopyExtParams, dataCopyPadExtParams);
        countQueue.EnQue(countUb);
    }

    __aicore__ inline void CopyInSumAndSquare(int64_t offset, int64_t currentCNum)
    {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = this->nLen;
        dataCopyExtParams.blockLen = currentCNum * sizeof(T);
        dataCopyExtParams.srcStride = (this->cLen - currentCNum) * sizeof(T);
        dataCopyExtParams.dstStride = 0;
        DataCopyPadExtParams<T> dataCopyPadExtParams {false, 0, 0, 0};

        LocalTensor<T> sumUb = sumQueue.AllocTensor<T>();
        LocalTensor<T> squareSumUb = squareSumQueue.AllocTensor<T>();
        DataCopyPad(sumUb, sumGm[offset], dataCopyExtParams, dataCopyPadExtParams);
        DataCopyPad(squareSumUb, squareSumGm[offset], dataCopyExtParams, dataCopyPadExtParams);
        sumQueue.EnQue(sumUb);
        squareSumQueue.EnQue(squareSumUb);
    }

    __aicore__ inline void CopyInRunningMeanVar(int64_t offset, int64_t currentCNum) {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = currentCNum * sizeof(T);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;
        DataCopyPadExtParams<T> dataCopyPadExtParams {false, 0, 0, 0};

        LocalTensor<T> runningMeanUb = runningMeanInQueue.AllocTensor<T>();
        LocalTensor<T> runningVarUb = runningVarInQueue.AllocTensor<T>();
        DataCopyPad(runningMeanUb, runningMeanInGm[offset], dataCopyExtParams, dataCopyPadExtParams);
        DataCopyPad(runningVarUb, runningVarInGm[offset], dataCopyExtParams, dataCopyPadExtParams);
        runningMeanInQueue.EnQue(runningMeanUb);
        runningVarInQueue.EnQue(runningVarUb);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t currentCNum) 
    {
        LocalTensor<T> batchMeanLocal = batchMeanQueue.DeQue<T>();
        LocalTensor<T> batchVarLocal = batchRstdQueue.DeQue<T>();
        LocalTensor<T> runningMeanOutLocal = runningMeanOutQueue.DeQue<T>();
        LocalTensor<T> runningVarOutLocal = runningVarOutQueue.DeQue<T>();

        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = currentCNum * sizeof(T);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;
        DataCopyPad(batchMeanGm[offset], batchMeanLocal, dataCopyExtParams);
        DataCopyPad(batchRstdGm[offset], batchVarLocal, dataCopyExtParams);
        DataCopyPad(runningMeanOutGm[offset], runningMeanOutLocal, dataCopyExtParams);
        DataCopyPad(runningVarOutGm[offset], runningVarOutLocal, dataCopyExtParams);

        batchMeanQueue.FreeTensor(batchMeanLocal);
        batchRstdQueue.FreeTensor(batchVarLocal);
        runningMeanOutQueue.FreeTensor(runningMeanOutLocal);
        runningVarOutQueue.FreeTensor(runningVarOutLocal);
    }

    /* global memory address */
    GlobalTensor<T> sumGm;
    GlobalTensor<T> squareSumGm;
    GlobalTensor<int32_t> countGm;
    GlobalTensor<T> runningMeanInGm;
    GlobalTensor<T> runningVarInGm;

    GlobalTensor<T> batchMeanGm;
    GlobalTensor<T> batchRstdGm;
    GlobalTensor<T> runningMeanOutGm;
    GlobalTensor<T> runningVarOutGm;

    /* variable */
    int64_t blockDim;
    int64_t blockFormer;
    int64_t blockLoopNum; 
    int64_t nLen;
    int64_t cLen;
    int64_t ubFormer;
    int64_t ubTail;
    float momentum;
    float eps;

    /* ascendc variable */
    TPipe* pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> squareSumQueue;
    TQue<QuePosition::VECIN, 1> countQueue; 
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchRstdQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> castCountBuff;
    TBuf<TPosition::VECCALC> sumAllBuff;
    TBuf<TPosition::VECCALC> squareSumAllBuff;
    TBuf<TPosition::VECCALC> countAllBuff;

    // b16 场景下用临时空间用作转B32
    TBuf<TPosition::VECCALC> castSumBuff;
    TBuf<TPosition::VECCALC> castSquareSumBuff;
};
} // namespace SyncBatchNormGatherStats
#endif

