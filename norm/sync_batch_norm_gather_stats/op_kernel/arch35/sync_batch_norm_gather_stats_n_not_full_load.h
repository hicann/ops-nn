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
 * \file sync_batch_norm_gather_stats_n_not_full_load.h
 * \brief
 */

#ifndef __SYNC_BATCH_NORM_GATHER_STATS_N_NOT_FULL_LOAD__
#define __SYNC_BATCH_NORM_GATHER_STATS_N_NOT_FULL_LOAD__

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
class SyncBatchNormGatherStatsNNotFullLoad
{
public:
    __aicore__ inline SyncBatchNormGatherStatsNNotFullLoad(const SyncBatchNormGatherStatsNNotFullLoadTilingData* tilingData, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->blockDim = tilingData->blockDim;
        this->cLen = tilingData->cLen;
        this->cFactor = tilingData->cFactor;
        this->nFactor = tilingData->nFactor;
        this->nLoop = tilingData->nLoop;
        this->nMainFoldCount = tilingData->nMainFoldCount;
        this->nTail = tilingData->nTail;
        this->cacheBufferCount = tilingData->cacheBufferCount;
        this->resultCacheId = tilingData->resultCacheId;
        this->cLoopMainBlock = tilingData->cLoopMainBlock;
        this->cTileMainBlock = tilingData->cTileMainBlock;
        this->momentum = tilingData->momentum;
        this->eps = tilingData->eps;

        if (GetBlockIdx() == (this->blockDim - 1)) {
            this->cBlockLoop = tilingData->cLoopTailBlock;
            this->cTail = tilingData->cTailTailBlock;
        } else {
            this->cBlockLoop = tilingData->cLoopMainBlock;
            this->cTail = tilingData->cTileMainBlock;
        }
    }

    __aicore__ inline void Init(GM_ADDR total_sum, GM_ADDR total_square_sum, GM_ADDR sample_count, GM_ADDR running_mean, 
                                GM_ADDR running_var, GM_ADDR batch_mean, GM_ADDR batch_invstd, GM_ADDR running_mean_update, 
                                GM_ADDR running_var_update)
    {
        auto blockIdx = GetBlockIdx();

        int64_t cGmOffset = ((this->cLoopMainBlock - 1) * this->cFactor + this->cTileMainBlock) * blockIdx;

        sumGm.SetGlobalBuffer((__gm__ T*)total_sum + cGmOffset);
        squareSumGm.SetGlobalBuffer((__gm__ T*)total_square_sum + cGmOffset);
        countGm.SetGlobalBuffer((__gm__ int32_t*)sample_count);
        runningMeanInGm.SetGlobalBuffer((__gm__ T*)running_mean + cGmOffset);
        runningVarInGm.SetGlobalBuffer((__gm__ T*)running_var + cGmOffset);

        batchMeanGm.SetGlobalBuffer((__gm__ T*)batch_mean + cGmOffset);
        batchRstdGm.SetGlobalBuffer((__gm__ T*)batch_invstd + cGmOffset);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T*)running_mean_update + cGmOffset);
        runningVarOutGm.SetGlobalBuffer((__gm__ T*)running_var_update + cGmOffset);

        if constexpr (std::is_same<T, float>::value) {
            pipe->InitBuffer(sumQueue, DOUBLE_BUFFER, this->nFactor * this->cFactor * sizeof(T));
            pipe->InitBuffer(squareSumQueue, DOUBLE_BUFFER, this->nFactor * this->cFactor * sizeof(T));
        } else {
            pipe->InitBuffer(sumQueue, TRIPLE_BUFFER, this->nFactor * this->cFactor * sizeof(T));
            pipe->InitBuffer(squareSumQueue, TRIPLE_BUFFER, this->nFactor * this->cFactor * sizeof(T));
            pipe->InitBuffer(sumSquareBuff, this->nFactor * this->cFactor * sizeof(float));
        }
        pipe->InitBuffer(countQueue, DOUBLE_BUFFER, this->nFactor * sizeof(int32_t));
        pipe->InitBuffer(runningMeanInQueue, DOUBLE_BUFFER, this->cFactor * sizeof(T));
        pipe->InitBuffer(runningVarInQueue, DOUBLE_BUFFER, this->cFactor * sizeof(T));

        pipe->InitBuffer(batchMeanQueue, 1, this->cFactor * sizeof(T));
        pipe->InitBuffer(batchRstdQueue, 1, this->cFactor * sizeof(T));
        pipe->InitBuffer(runningMeanOutQueue, 1, this->cFactor * sizeof(T));
        pipe->InitBuffer(runningVarOutQueue, 1, this->cFactor * sizeof(T));

        pipe->InitBuffer(countBuff, this->nFactor * sizeof(int64_t));
        pipe->InitBuffer(sumSquareAllCacheBuff, (this->cacheBufferCount + DOUBLE_BUFFER) * this->cFactor * sizeof(float));
        pipe->InitBuffer(countAllCacheBuff, (this->cacheBufferCount + 1) * BLOCK_SIZE);
    }

    __aicore__ inline void Process()
    {
        SampleCountProcess();
        LocalTensor<int64_t> countAllCacheUb = countAllCacheBuff.Get<int64_t>();

        int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDSToV);
        WaitFlag<HardEvent::V_S>(eventIDSToV);
        int64_t int64CountAll = countAllCacheUb[this->resultCacheId * BS_INT64].GetValue(0);
        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDVToS);
        WaitFlag<HardEvent::S_V>(eventIDVToS);

        for (int64_t i = 0; i < this->cBlockLoop - 1; i++) {
            ProcessOnce(i, this->cFactor, int64CountAll);
        }
        ProcessOnce(this->cBlockLoop - 1, this->cTail, int64CountAll);
    }

private:
    __aicore__ inline void ProcessOnce(int64_t ubTimes, int64_t currentCNum, int64_t int64CountAll)
    {
        SumSquareReduceProcess(this->sumQueue, this->sumGm, ubTimes, currentCNum, 1);
        SumSquareReduceProcess(this->squareSumQueue, this->squareSumGm, ubTimes, currentCNum, 0);
        CopyInRunningMeanVar(ubTimes * this->cFactor, currentCNum);
        LocalTensor<float> sumSquareAllCacheUb = sumSquareAllCacheBuff.Get<float>();
        LocalTensor<T> runningMeanUb = runningMeanInQueue.DeQue<T>();
        LocalTensor<T> runningVarUb = runningVarInQueue.DeQue<T>();

        __local_mem__ float* sumAllLocal = (__local_mem__ float*)sumSquareAllCacheUb[(this->cacheBufferCount+1)*this->cFactor].GetPhyAddr();
        __local_mem__ float* squareSumAllLocal = (__local_mem__ float*)sumSquareAllCacheUb[this->cacheBufferCount*this->cFactor].GetPhyAddr();
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

        CopyOut(ubTimes * this->cFactor, currentCNum);
    }

    __aicore__ inline int64_t GetCacheId(const int64_t idx)
    {
        return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
    }
    __aicore__ inline void CopyInN(int64_t offset, int64_t currentNum)
    {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = currentNum * sizeof(int32_t);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> dataCopyPadExtParams {false, 0, 0, 0};

        LocalTensor<int32_t> countUb = countQueue.AllocTensor<int32_t>();
        DataCopyPad(countUb, countGm[offset], dataCopyExtParams, dataCopyPadExtParams);
        countQueue.EnQue(countUb);
    }

    __aicore__ inline void CastCount1VF(__local_mem__ int64_t* dst, __local_mem__ int32_t* src, int32_t num)
    {
        __VEC_SCOPE__
        {
            RegTensor<int32_t> srcReg;
            RegTensor<int64_t> dstReg;
            MaskReg mask;
            uint16_t castTimes = CeilDivision(num, VL_INT64);
            uint32_t width = num;

            for (uint16_t i = 0; i < castTimes; i++) {
                mask = UpdateMask<int64_t>(width);
                auto srcAddr = src + i * VL_INT64;
                auto dstAddr = dst + i * VL_INT64;
                DataCopy<int32_t, LoadDist::DIST_UNPACK_B32>(srcReg, srcAddr);
                Cast<int64_t, int32_t, castTraitB32B64>(dstReg, srcReg, mask);
                DataCopy<int64_t, StoreDist::DIST_NORM>(dstAddr, dstReg, mask);
            }
        }
    }

    __aicore__ inline void FoldCountVF(__local_mem__ int64_t* src1, __local_mem__ int32_t* src2, int32_t num)
    {
        __VEC_SCOPE__
        {
            RegTensor<int64_t> src1Reg;
            RegTensor<int32_t> src2Reg;
            RegTensor<int64_t> tempReg;
            MaskReg mask;
            uint16_t foldTimes = CeilDivision(num, VL_INT64);
            uint32_t width = num;

            for (uint16_t i = 0; i < foldTimes; i++) {
                mask = UpdateMask<int64_t>(width);
                auto src1Addr = src1 + i * VL_INT64;
                auto src2Addr = src2 + i * VL_INT64;
                DataCopy<int64_t, LoadDist::DIST_NORM>(src1Reg, src1Addr);
                DataCopy<int32_t, LoadDist::DIST_UNPACK_B32>(src2Reg, src2Addr);
                Cast<int64_t, int32_t, castTraitB32B64>(tempReg, src2Reg, mask);
                Add(src1Reg, src1Reg, tempReg, mask);
                DataCopy<int64_t, StoreDist::DIST_NORM>(src1Addr, src1Reg, mask);
            }
        }
    }

    template<typename U>
    __aicore__ inline void UpdateCache(const LocalTensor<U>& dstTensor, const LocalTensor<U>& srcTensor,
                                    const int64_t cacheId, const int64_t stride, const int64_t count)
    {
        uint16_t outerLoopTimes =
            ops::CeilDiv(static_cast<int64_t>(count * sizeof(U)), static_cast<int64_t>(VECTOR_LENGTH));
        uint16_t innerLoopTimes = cacheId;
        uint32_t outerLoopStride = VECTOR_LENGTH / sizeof(U);
        uint32_t innerLoopStride = stride;

        __local_mem__ U* dst = (__local_mem__ U*)dstTensor.GetPhyAddr();
        __local_mem__ U* cache = (__local_mem__ U*)dstTensor.GetPhyAddr() + cacheId * stride;
        __local_mem__ U* src = (__local_mem__ U*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(count);
            RegTensor<U> aReg, bReg;
            MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = UpdateMask<U>(sreg);
                DataCopy(aReg, (__local_mem__ U*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    DataCopy(bReg, (__local_mem__ U*)dst + i * outerLoopStride + j * innerLoopStride);
                    Add<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                DataCopy((__local_mem__ U*)cache + i * outerLoopStride, aReg, pMask);
            }
        }
    }

    __aicore__ inline void SampleCountProcess()
    {
        LocalTensor<int64_t> countBuffUb = countBuff.Get<int64_t>();
        __local_mem__ int64_t* countBuffLocal = (__local_mem__ int64_t*)countBuffUb.GetPhyAddr();
        LocalTensor<int64_t> countAllCacheUb = countAllCacheBuff.Get<int64_t>();

        for (int64_t basicBlockIdx = 0; basicBlockIdx < this->nLoop; basicBlockIdx++) {
            CopyInN(basicBlockIdx * this->nFactor, this->nFactor);
            LocalTensor<int32_t> count1Ub = countQueue.DeQue<int32_t>();
            __local_mem__ int32_t* count1Local = (__local_mem__ int32_t*)count1Ub.GetPhyAddr();
            CastCount1VF(countBuffLocal, count1Local, this->nFactor);
            countQueue.FreeTensor(count1Ub);

            if (basicBlockIdx < this->nMainFoldCount) {
                CopyInN((basicBlockIdx + this->nLoop) * this->nFactor, this->nFactor);
                LocalTensor<int32_t> count2Ub = countQueue.DeQue<int32_t>();
                __local_mem__ int32_t* count2Local = (__local_mem__ int32_t*)count2Ub.GetPhyAddr();
                FoldCountVF(countBuffLocal, count2Local, this->nFactor);
                countQueue.FreeTensor(count2Ub);
            } else if ((basicBlockIdx == this->nMainFoldCount) && (this->nTail > 0)) {
                CopyInN((basicBlockIdx + this->nLoop) * this->nFactor, this->nTail);
                LocalTensor<int32_t> count2Ub = countQueue.DeQue<int32_t>();
                __local_mem__ int32_t* count2Local = (__local_mem__ int32_t*)count2Ub.GetPhyAddr();
                FoldCountVF(countBuffLocal, count2Local, this->nTail);
                countQueue.FreeTensor(count2Ub);
            }
            int64_t cacheId = GetCacheId(basicBlockIdx);
            int64_t tempIndex = this->cacheBufferCount * BS_INT64;
            uint32_t shape[] = {1, static_cast<uint32_t>(this->nFactor)};
            ReduceSum<int64_t, AscendC::Pattern::Reduce::AR, true>(countAllCacheUb[tempIndex], countBuffUb, shape, false);
            UpdateCache<int64_t>(countAllCacheUb, countAllCacheUb[tempIndex], cacheId, BS_INT64, 1);
        }
    }

    __aicore__ inline void CopyInSumSquare(TQue<QuePosition::VECIN, 1>& inQueue, GlobalTensor<T>& inGm, 
                                    int64_t offset, int64_t currentNNum, int64_t currentCNum)
    {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = currentNNum;
        dataCopyExtParams.blockLen = currentCNum * sizeof(T);
        dataCopyExtParams.srcStride = (this->cLen - currentCNum) * sizeof(T);
        dataCopyExtParams.dstStride = 0;
        DataCopyPadExtParams<T> dataCopyPadExtParams {false, 0, 0, 0};

        LocalTensor<T> sumUb = inQueue.AllocTensor<T>();
        DataCopyPad(sumUb, inGm[offset], dataCopyExtParams, dataCopyPadExtParams);
        inQueue.EnQue(sumUb);
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

    __aicore__ inline void CastSum1VF(__local_mem__ float* dst, __local_mem__ T* src, int32_t num)
    {
        __VEC_SCOPE__
        {
            RegTensor<T> srcReg;
            RegTensor<float> dstReg;
            MaskReg mask;
            uint16_t castTimes = CeilDivision(num, VL_FP32);
            uint32_t width = num;

            for (uint16_t i = 0; i < castTimes; i++) {
                mask = UpdateMask<float>(width);
                auto srcAddr = src + i * VL_FP32;
                auto dstAddr = dst + i * VL_FP32;
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr);
                Cast<float, T, castTraitB162B32>(dstReg, srcReg, mask);
                DataCopy<float, StoreDist::DIST_NORM>(dstAddr, dstReg, mask);
            }
        }
    }

    __aicore__ inline void FoldSumSquareVF(__local_mem__ float* src1, __local_mem__ T* src2, int32_t num)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> src1Reg;
            RegTensor<T> src2Reg;
            RegTensor<float> tempReg;
            MaskReg mask;
            uint16_t foldTimes = CeilDivision(num, VL_FP32);
            uint32_t width = num;

            for (uint16_t i = 0; i < foldTimes; i++) {
                mask = UpdateMask<float>(width);
                auto src1Addr = src1 + i * VL_FP32;
                auto src2Addr = src2 + i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(src1Reg, src1Addr);
                if constexpr (std::is_same<T, float>::value) {
                    DataCopy<float, LoadDist::DIST_NORM>(src2Reg, src2Addr);
                    Add(src1Reg, src1Reg, src2Reg, mask);
                } else {
                    DataCopy<T, LoadDist::DIST_UNPACK_B16>(src2Reg, src2Addr);
                    Cast<float, T, castTraitB162B32>(tempReg, src2Reg, mask);
                    Add(src1Reg, src1Reg, tempReg, mask);
                }
                DataCopy<float, StoreDist::DIST_NORM>(src1Addr, src1Reg, mask);
            }
        }
    }

    __aicore__ inline void SumSquareReduceProcess(TQue<QuePosition::VECIN, 1>& inQueue, GlobalTensor<T>& inGm,
                                    int64_t ubTimes, int64_t currentCNum, int64_t resOffset)
    {
        int64_t currentCAlignNum = (currentCNum + (BLOCK_SIZE / sizeof(T)) - 1) / (BLOCK_SIZE / sizeof(T)) * (BLOCK_SIZE / sizeof(T));
        LocalTensor<float> sumSquareBuffUb;
        __local_mem__ float* sumSquareBuffLocal;
        if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, bfloat16_t>::value) {
            sumSquareBuffUb = sumSquareBuff.Get<float>();
            sumSquareBuffLocal = (__local_mem__ float*)sumSquareBuffUb.GetPhyAddr();
        }
        LocalTensor<float> sumSquareAllCacheUb = sumSquareAllCacheBuff.Get<float>();

        for (int64_t basicBlockIdx = 0; basicBlockIdx < this->nLoop; basicBlockIdx++) {
            int64_t sum1Offset = ubTimes * this->cFactor + basicBlockIdx * this->nFactor * this->cLen;
            CopyInSumSquare(inQueue, inGm, sum1Offset, this->nFactor, currentCNum);
            LocalTensor<T> sum1Ub = inQueue.DeQue<T>();
            __local_mem__ T* sum1Local = (__local_mem__ T*)sum1Ub.GetPhyAddr();

            if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, bfloat16_t>::value) {
                CastSum1VF(sumSquareBuffLocal, sum1Local, this->nFactor * this->cFactor);
                inQueue.FreeTensor(sum1Ub);
            }
            int64_t sum2Offset = ubTimes * this->cFactor + (basicBlockIdx + this->nLoop) * this->nFactor * this->cLen;
            if (basicBlockIdx < this->nMainFoldCount) {
                CopyInSumSquare(inQueue, inGm, sum2Offset, this->nFactor, currentCNum);
                LocalTensor<T> sum2Ub = inQueue.DeQue<T>();
                __local_mem__ T* sum2Local = (__local_mem__ T*)sum2Ub.GetPhyAddr();
                if constexpr (std::is_same<T, float>::value) {
                    FoldSumSquareVF(sum1Local, sum2Local, this->nFactor * currentCAlignNum);
                } else {
                    FoldSumSquareVF(sumSquareBuffLocal, sum2Local, this->nFactor * currentCAlignNum);
                }
                inQueue.FreeTensor(sum2Ub);
            } else if ((basicBlockIdx == this->nMainFoldCount) && (this->nTail > 0)) {
                CopyInSumSquare(inQueue, inGm, sum2Offset, this->nTail, currentCNum);
                LocalTensor<T> sum2Ub = inQueue.DeQue<T>();
                __local_mem__ T* sum2Local = (__local_mem__ T*)sum2Ub.GetPhyAddr();
                if constexpr (std::is_same<T, float>::value) {
                    FoldSumSquareVF(sum1Local, sum2Local, this->nTail * currentCAlignNum);
                } else {
                    FoldSumSquareVF(sumSquareBuffLocal, sum2Local, this->nTail * currentCAlignNum);
                }
                inQueue.FreeTensor(sum2Ub);
            }

            int64_t cacheId = GetCacheId(basicBlockIdx);
            int64_t tempIndex = this->cacheBufferCount * this->cFactor;
            uint32_t shape[] = {static_cast<uint32_t>(this->nFactor), static_cast<uint32_t>(currentCAlignNum)};
            if constexpr (std::is_same<T, float>::value) {
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumSquareAllCacheUb[tempIndex], sum1Ub, shape, false);
                inQueue.FreeTensor(sum1Ub);
            } else {
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumSquareAllCacheUb[tempIndex], sumSquareBuffUb, shape, false);
            }
            UpdateCache<float>(sumSquareAllCacheUb, sumSquareAllCacheUb[tempIndex], cacheId, this->cFactor, currentCNum);
        }
        Copy(sumSquareAllCacheUb[(this->cacheBufferCount + resOffset) * this->cFactor], sumSquareAllCacheUb[this->resultCacheId * this->cFactor], this->cFactor);
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
    int64_t cLen;
    int64_t cFactor;
    int64_t cBlockLoop;
    int64_t cTail;
    int64_t nFactor;
    int64_t nLoop;
    int64_t nMainFoldCount;
    int64_t nTail;
    int64_t cacheBufferCount;
    int64_t cLoopMainBlock;
    int64_t cTileMainBlock;
    int32_t resultCacheId;
    float momentum;
    float eps;

    /* ascendc variable */
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> sumQueue;
    TQue<QuePosition::VECIN, 1> squareSumQueue;
    TQue<QuePosition::VECIN, 1> countQueue; 
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchRstdQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> countBuff;
    TBuf<TPosition::VECCALC> sumSquareAllCacheBuff;
    TBuf<TPosition::VECCALC> countAllCacheBuff;

    TBuf<TPosition::VECCALC> sumSquareBuff; // B16场景下需要一个buffer
};
} // namespace SyncBatchNormGatherStats
#endif // __SYNC_BATCH_NORM_GATHER_STATS_N_NOT_FULL_LOAD__
