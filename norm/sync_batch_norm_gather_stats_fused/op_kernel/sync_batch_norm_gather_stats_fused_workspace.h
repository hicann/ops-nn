/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused_workspace.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_WORKSPACE_H
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_WORKSPACE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "sync_batch_norm_gather_stats_fused_counts_sum.h"

namespace SyncBatchNormGatherStatsFused {
using namespace AscendC;

template <typename T>
class SyncBatchNormGatherStatsFusedWorkspace {
public:
    __aicore__ inline SyncBatchNormGatherStatsFusedWorkspace(){};

    __aicore__ inline void Init(GM_ADDR mean, GM_ADDR invstd, GM_ADDR counts, GM_ADDR runningMean, GM_ADDR runningVar,
                                GM_ADDR meanAllOut, GM_ADDR invstdAllOut, GM_ADDR runningMeanOut, GM_ADDR runningVarOut,
                                GM_ADDR workspace, const SyncBatchNormGatherStatsFusedTilingDataWorkspace* tilingData,
                                TPipe& pipeIn)
    {
        pipe = pipeIn;
        nLength = tilingData->nLength;
        cLength = tilingData->cLength;
        blockNum = tilingData->blockNum;
        momentum = tilingData->momentum;
        eps = tilingData->eps;

        countsGm.SetGlobalBuffer((__gm__ float*)counts, nLength);
        countsReduceWorkspace.SetGlobalBuffer((__gm__ float*)workspace, GetBlockNum());
        InitOutput<float>(countsReduceWorkspace[GetBlockIdx()], 1, 0.0f);

        if (GetBlockIdx() < tilingData->blockNum) {
            blockCStart = GetBlockIdx() * tilingData->blockFormer;
            curCLength = GetBlockIdx() == (blockNum - 1) ? tilingData->blockTail : tilingData->blockFormer;
            ubFormer = GetBlockIdx() == (blockNum - 1) ? tilingData->ubFormerOfTail : tilingData->ubFormerOfFormer;
            ubLoop = GetBlockIdx() == (blockNum - 1) ? tilingData->ubLoopOfTail : tilingData->ubLoopOfFormer;
            ubTail = GetBlockIdx() == (blockNum - 1) ? tilingData->ubTailOfTail : tilingData->ubTailOfFormer;

            meanGm.SetGlobalBuffer((__gm__ T*)mean, nLength * cLength);
            invstdGm.SetGlobalBuffer((__gm__ T*)invstd, nLength * cLength);
            meanAllOutGm.SetGlobalBuffer((__gm__ T*)meanAllOut, cLength);
            invstdAllOutGm.SetGlobalBuffer((__gm__ T*)invstdAllOut, cLength);
            runningMeanOutGm.SetGlobalBuffer((__gm__ T*)runningMeanOut, cLength);
            runningVarOutGm.SetGlobalBuffer((__gm__ T*)runningVarOut, cLength);
            runningMeanGm.SetGlobalBuffer((__gm__ T*)runningMean, cLength);
            runningVarGm.SetGlobalBuffer((__gm__ T*)runningVar, cLength);

            ////大小待确定，数量待确定
        }
    }

    __aicore__ inline void initBuffer()
    {
        int64_t bufferSize = ubFormer * sizeof(float);
        pipe.InitBuffer(queue0_, 1, bufferSize);
        pipe.InitBuffer(queue1_, 1, bufferSize);
        pipe.InitBuffer(queue2_, 1, bufferSize);
        pipe.InitBuffer(queue3_, 1, bufferSize);
        pipe.InitBuffer(queue4_, 1, bufferSize);
    }

    __aicore__ inline void Process()
    {
        ComputeCountsSum();
        SyncAll();
        pipe.Reset();
        if (GetBlockIdx() < blockNum) {
            initBuffer();
            unbiasCountsSum = countsSumScalar - 1.0f;
            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                buffer2_ = queue2_.AllocTensor<float>();
                Duplicate(buffer2_, static_cast<float>(0.0f), ubFormer);
                buffer3_ = queue3_.AllocTensor<float>();
                Duplicate(buffer3_, static_cast<float>(0.0f), ubFormer);

                int64_t curColNum = (ubIdx == ubLoop - 1) ? ubTail : ubFormer;
                for (int64_t rowIdx = 0; rowIdx < nLength; rowIdx++) {
                    ComputeGlobalMean(rowIdx, ubIdx, curColNum);
                }

                for (int64_t rowIdx = 0; rowIdx < nLength; rowIdx++) {
                    ComputeGlobalVar(rowIdx, ubIdx, curColNum);
                }
                ComputeRunningMean(ubIdx, curColNum);
                meanAllWithRunningMeanCopyOut(ubIdx, curColNum);
                ComputeGlobalInvstdWithRunningVar(ubIdx, curColNum);
                FinalCoyOut(ubIdx, curColNum);
            }
        }
    }
    __aicore__ inline void ComputeCountsSum()
    {
        SyncBatchNormGatherStatsFusedCountsSum<T> op;
        op.initBuffer(pipe, countsGm, countsReduceWorkspace);

        int64_t nAlign = ((nLength + 8 - 1) / 8) * 8;

        op.FinalComputeProcess(nAlign, nLength, countsSumScalar);
    }
    __aicore__ inline void CopyMeanGm(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        buffer0_ = queue0_.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParam;
        intriParam.blockCount = 1;
        intriParam.blockLen = curColNum * sizeof(T);
        intriParam.srcStride = 0;
        intriParam.dstStride = 0;

        int64_t offset = rowIdx * cLength + blockCStart + ubIdx * ubFormer;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0_.ReinterpretCast<T>(), meanGm[offset], intriParam, padParams);
        } else {
            DataCopyPad(buffer0_.ReinterpretCast<T>()[ubFormer], meanGm[offset], intriParam, padParams);
        }

        queue0_.EnQue(buffer0_);
        buffer0_ = queue0_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer0_, buffer0_.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, ubFormer);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyInvstd(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        buffer1_ = queue1_.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curColNum * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        int64_t offset = rowIdx * cLength + blockCStart + ubIdx * ubFormer;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer1_.ReinterpretCast<T>(), invstdGm[offset], intriParams, padParams);
        } else {
            DataCopyPad(buffer1_.ReinterpretCast<T>()[ubFormer], invstdGm[offset], intriParams, padParams);
        }

        queue1_.EnQue(buffer1_);
        buffer1_ = queue1_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer1_, buffer1_.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, ubFormer);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeGlobalMean(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        CopyMeanGm(rowIdx, ubIdx, curColNum);
        float countValue = countsGm.GetValue(rowIdx);
        PipeBarrier<PIPE_V>();

        Muls(buffer0_, buffer0_, countValue, curColNum);
        PipeBarrier<PIPE_V>();

        buffer1_ = queue1_.AllocTensor<float>();
        Duplicate(buffer1_, countsSumScalar, curColNum);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer0_, buffer0_, buffer1_, curColNum);
        PipeBarrier<PIPE_V>();

        Add(buffer2_, buffer2_, buffer0_, curColNum);
        PipeBarrier<PIPE_V>();

        queue0_.FreeTensor(buffer0_);
        queue1_.FreeTensor(buffer1_);
    }

    __aicore__ inline void ComputeGlobalVar(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        CopyInvstd(rowIdx, ubIdx, curColNum);
        float countValue = countsGm.GetValue(rowIdx);

        buffer0_ = queue0_.AllocTensor<float>();
        Duplicate(buffer0_, static_cast<float>(1.0f), curColNum);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer1_, buffer0_, buffer1_, curColNum);
        PipeBarrier<PIPE_V>();
        queue0_.FreeTensor(buffer0_);

        CopyMeanGm(rowIdx, ubIdx, curColNum);

        Mul(buffer1_, buffer1_, buffer1_, curColNum);
        PipeBarrier<PIPE_V>();

        Adds(buffer1_, buffer1_, -eps, curColNum);
        PipeBarrier<PIPE_V>();

        Sub(buffer0_, buffer0_, buffer2_, curColNum);
        PipeBarrier<PIPE_V>();

        Mul(buffer0_, buffer0_, buffer0_, curColNum);
        PipeBarrier<PIPE_V>();

        Add(buffer0_, buffer0_, buffer1_, curColNum);
        PipeBarrier<PIPE_V>();

        Muls(buffer0_, buffer0_, countValue, curColNum);
        PipeBarrier<PIPE_V>();

        Add(buffer3_, buffer3_, buffer0_, curColNum);
        PipeBarrier<PIPE_V>();
        queue0_.FreeTensor(buffer0_);
        queue1_.FreeTensor(buffer1_);
    }

    __aicore__ inline void CopyRunning(const int64_t ubIdx, const int64_t curColNum, const GlobalTensor<T>& runningGm)

    {
        buffer0_ = queue0_.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curColNum * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        int64_t offset = blockCStart + ubIdx * ubFormer;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0_.ReinterpretCast<T>(), runningGm[offset], intriParams, padParams);
        } else {
            DataCopyPad(buffer0_.ReinterpretCast<T>()[ubFormer], runningGm[offset], intriParams, padParams);
        }

        queue0_.EnQue(buffer0_);
        buffer0_ = queue0_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer0_, buffer0_.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, curColNum);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void ComputeRunningMean(const int64_t ubIdx, const int64_t curColNum)
    {
        CopyRunning(ubIdx, curColNum, runningMeanGm);
        buffer4_ = queue4_.AllocTensor<float>();
        Muls(buffer0_, buffer0_, 1.0f - momentum, curColNum);
        PipeBarrier<PIPE_V>();

        Muls(buffer4_, buffer2_, momentum, curColNum);
        PipeBarrier<PIPE_V>();

        Add(buffer4_, buffer4_, buffer0_, curColNum);
        PipeBarrier<PIPE_V>();
        queue0_.FreeTensor(buffer0_);
    }
    __aicore__ inline void meanAllWithRunningMeanCopyOut(const int64_t ubIdx, const int64_t curColNum)
    {
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curColNum * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer2_.ReinterpretCast<T>(), buffer2_, RoundMode::CAST_RINT, curColNum);
            PipeBarrier<PIPE_V>();
            Cast(buffer4_.ReinterpretCast<T>(), buffer4_, RoundMode::CAST_RINT, curColNum);
            PipeBarrier<PIPE_V>();
        }
        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event0);
        WaitFlag<HardEvent::V_MTE3>(event0);
        DataCopyPad(meanAllOutGm[blockCStart + ubIdx * ubFormer], buffer2_.ReinterpretCast<T>(), intriParams);
        DataCopyPad(runningMeanOutGm[blockCStart + ubIdx * ubFormer], buffer4_.ReinterpretCast<T>(), intriParams);

        queue2_.FreeTensor(buffer2_);
        queue4_.FreeTensor(buffer4_);
    }

    __aicore__ inline void ComputeGlobalInvstdWithRunningVar(const int64_t ubIdx, const int64_t curColNum)
    {
        buffer1_ = queue1_.AllocTensor<float>();

        CopyRunning(ubIdx, curColNum, runningVarGm);

        Muls(buffer0_, buffer0_, 1.0f - momentum, curColNum);
        PipeBarrier<PIPE_V>();

        Duplicate(buffer1_, unbiasCountsSum, curColNum);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer1_, buffer3_, buffer1_, curColNum);
        PipeBarrier<PIPE_V>();

        Muls(buffer1_, buffer1_, momentum, curColNum);
        PipeBarrier<PIPE_V>();

        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event0);
        WaitFlag<HardEvent::MTE3_V>(event0);

        buffer2_ = queue2_.AllocTensor<float>();
        Add(buffer2_, buffer1_, buffer0_, curColNum); // runningVarOut
        PipeBarrier<PIPE_V>();

        Duplicate(buffer1_, countsSumScalar, curColNum);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer3_, buffer3_, buffer1_, curColNum);
        PipeBarrier<PIPE_V>();

        Adds(buffer3_, buffer3_, eps, curColNum);
        PipeBarrier<PIPE_V>();

        Sqrt(buffer3_, buffer3_, curColNum); // invastdAll
        PipeBarrier<PIPE_V>();

        Duplicate(buffer1_, static_cast<float>(1.0), curColNum);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer3_, buffer1_, buffer3_, curColNum);
        PipeBarrier<PIPE_V>();
        queue0_.FreeTensor(buffer0_);
        queue1_.FreeTensor(buffer1_);
    }

    __aicore__ inline void FinalCoyOut(const int64_t ubIdx, const int64_t curColNum) // 补齐Cast
    {
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer2_.ReinterpretCast<T>(), buffer2_, RoundMode::CAST_RINT, curColNum);
            PipeBarrier<PIPE_V>();
            Cast(buffer3_.ReinterpretCast<T>(), buffer3_, RoundMode::CAST_RINT, curColNum);
            PipeBarrier<PIPE_V>();
        }
        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event0);
        WaitFlag<HardEvent::V_MTE3>(event0);

        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curColNum * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        int64_t offset = blockCStart + ubIdx * ubFormer;
        DataCopyPad(runningVarOutGm[offset], buffer2_.ReinterpretCast<T>(), intriParams);
        DataCopyPad(invstdAllOutGm[offset], buffer3_.ReinterpretCast<T>(), intriParams);

        queue2_.FreeTensor(buffer2_);
        queue3_.FreeTensor(buffer3_);
    }

private:
    TPipe pipe;
    static constexpr DivConfig divConfig = {DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;
    int64_t nLength;
    int64_t cLength;
    int64_t blockNum;
    int64_t ubFormer;
    int64_t ubLoop;
    int64_t ubTail;
    int64_t curCLength;
    float momentum;
    float countsSumScalar;
    float eps;
    float unbiasCountsSum;
    int64_t blockCStart;

    GlobalTensor<T> meanGm;
    GlobalTensor<T> invstdGm;
    GlobalTensor<T> runningMeanGm;
    GlobalTensor<T> runningVarGm;
    GlobalTensor<float> countsGm;
    GlobalTensor<T> runningMeanOutGm;
    GlobalTensor<T> runningVarOutGm;
    GlobalTensor<T> meanAllOutGm;
    GlobalTensor<T> invstdAllOutGm;
    GlobalTensor<float> countsReduceWorkspace;

    LocalTensor<float> buffer0_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;

    TQue<QuePosition::VECIN, 1> queue0_;
    TQue<QuePosition::VECIN, 1> queue1_;
    TQue<QuePosition::VECOUT, 1> queue2_;
    TQue<QuePosition::VECOUT, 1> queue3_;
    TQue<QuePosition::VECOUT, 1> queue4_;
};
} // namespace SyncBatchNormGatherStatsFused
#endif // SYNC_BATCH_NORM_GATHER_STATS_FUSED_WORKSPACE_H