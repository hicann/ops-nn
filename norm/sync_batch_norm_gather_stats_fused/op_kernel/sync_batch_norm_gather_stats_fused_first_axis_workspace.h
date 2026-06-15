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
 * \file sync_batch_norm_gather_stats_fused_first_axis_workspace.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_FIRST_AXIS_WORKSPACE_H
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_FIRST_AXIS_WORKSPACE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "sync_batch_norm_gather_stats_fused_deterministic.h"
#include "sync_batch_norm_gather_stats_fused_compute.h"
#include "sync_batch_norm_gather_stats_fused_counts_sum.h"

namespace SyncBatchNormGatherStatsFused {
using namespace AscendC;

template <typename T>
class SyncBatchNormGatherStatsFusedFirstAxisWorkspace {
public:
    __aicore__ inline SyncBatchNormGatherStatsFusedFirstAxisWorkspace(){};

    __aicore__ inline void Init(
        GM_ADDR mean, GM_ADDR invstd, GM_ADDR counts, GM_ADDR runningMean, GM_ADDR runningVar, GM_ADDR meanAllOut,
        GM_ADDR invstdAllOut, GM_ADDR runningMeanOut, GM_ADDR runningVarOut, GM_ADDR workspace,
        const SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace* tilingData, TPipe& pipeIn)
    {
        pipe = pipeIn;
        nLength = tilingData->nLength;
        cLength = tilingData->cLength;
        blockNum = tilingData->blockNum;
        momentum = tilingData->momentum;
        eps = tilingData->eps;
        cAlign = tilingData->cAlignV;

        invstdAllOutGm.SetGlobalBuffer((__gm__ T*)invstdAllOut, cLength);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T*)runningMeanOut, cLength);
        runningVarOutGm.SetGlobalBuffer((__gm__ T*)runningVarOut, cLength);
        meanAllOutGm.SetGlobalBuffer((__gm__ T*)meanAllOut, cLength);
        runningMeanGm.SetGlobalBuffer((__gm__ T*)runningMean, cLength);
        runningVarGm.SetGlobalBuffer((__gm__ T*)runningVar, cLength);
        countsGm.SetGlobalBuffer((__gm__ float*)counts, nLength);

        int64_t wsLenPerBlock = cAlign * WORKSPACE_NUM;
        meanAllResultworkspaceGM.SetGlobalBuffer((__gm__ float*)workspace, cAlign);
        varAllResultworkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + cAlign, cAlign);
        workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace + 2 * cAlign, wsLenPerBlock * blockNum);
        countsReduceWorkspace.SetGlobalBuffer(
            (__gm__ float*)workspace + (2 + blockNum * WORKSPACE_NUM) * cAlign, GetBlockNum());
        InitOutput<float>(countsReduceWorkspace[GetBlockIdx()], 1, 0.0f);

        if (GetBlockIdx() < tilingData->blockNum) {
            blockNStart = GetBlockIdx() * tilingData->blockFormer;
            curNLength = GetBlockIdx() == (blockNum - 1) ? tilingData->blockTail : tilingData->blockFormer;
            ubFormer = tilingData->ubFormer;
            ubLoop = tilingData->ubLoop;
            ubTail = tilingData->ubTail;

            meanGm.SetGlobalBuffer((__gm__ T*)mean, nLength * cLength);
            invstdGm.SetGlobalBuffer((__gm__ T*)invstd, nLength * cLength);

            reduceWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + (2 + GetBlockIdx()) * cAlign, cAlign);
            InitOutput<float>(reduceWorkspaceGM, cAlign, 0.0f);
            initBuffer();
        }
    }

    __aicore__ inline void initBuffer()
    {
        int64_t bufferSize = ubFormer * sizeof(float);
        pipe.InitBuffer(queue0, 1, bufferSize);
        pipe.InitBuffer(queue1, 1, bufferSize);
        pipe.InitBuffer(queue2, 1, bufferSize);
    }

    __aicore__ inline void Process()
    {
        ComputeCountsSum();
        SyncAll();
        pipe.Reset();
        if (GetBlockIdx() < blockNum) {
            initBuffer();
            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                int64_t curColNum = (ubIdx == ubLoop - 1) ? ubTail : ubFormer;
                for (int64_t rowIdx = 0; rowIdx < curNLength; rowIdx++) {
                    ComputeGlobalMean(rowIdx, ubIdx, curColNum);
                }
            }
        }
        SyncAll();
        pipe.Reset();

        ComputeGlobalMeanDterministicFun();
        SyncAll();
        pipe.Reset();
        if (GetBlockIdx() < blockNum) {
            initBuffer();
            unbiasCountsSum = countsSumScalar - 1.0f;
            InitOutput<float>(reduceWorkspaceGM, cAlign, 0.0f);
            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                int64_t curColNum = (ubIdx == ubLoop - 1) ? ubTail : ubFormer;
                CopyReduceWorkspaceResult(ubIdx, curColNum);
                for (int64_t rowIdx = 0; rowIdx < curNLength; rowIdx++) {
                    ComputeGlobalVar(rowIdx, ubIdx, curColNum);
                }
                queue2.FreeTensor(buffer2);
            }
        }
        SyncAll();
        pipe.Reset();
        ComputeGlobalVarDterministic();
        SyncAll();
        pipe.Reset();
        ComputeGlobalInvstdWithRunningVarFun();
    }

    __aicore__ inline void ComputeCountsSum()
    {
        SyncBatchNormGatherStatsFusedCountsSum<T> op;
        op.initBuffer(pipe, countsGm, countsReduceWorkspace);

        int64_t nAlign = ((nLength + 7) / 8) * 8;

        op.FinalComputeProcess(nAlign, nLength, countsSumScalar);
    }
    __aicore__ inline void CopyReduceWorkspaceResult(const int64_t ubIdx, const int64_t curColNum)
    {
        buffer2 = queue2.AllocTensor<float>();
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParams;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        intriParams.blockCount = 1;
        intriParams.blockLen = curColNum * sizeof(float);

        DataCopyPad(buffer2, meanAllResultworkspaceGM[ubIdx * ubFormer], intriParams, padParams);

        queue2.EnQue(buffer2);
        buffer2 = queue2.DeQue<float>();
    }

    __aicore__ inline void CopyMeanData(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        buffer0 = queue0.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        intriParams.blockLen = curColNum * sizeof(T);

        int64_t offset = (blockNStart + rowIdx) * cLength + ubIdx * ubFormer;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0.ReinterpretCast<T>(), meanGm[offset], intriParams, padParams);
        } else {
            DataCopyPad(buffer0.ReinterpretCast<T>()[ubFormer], meanGm[offset], intriParams, padParams);
        }

        queue0.EnQue(buffer0);
        buffer0 = queue0.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer0, buffer0.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, ubFormer);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyInvstdData(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        buffer1 = queue1.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        intriParams.blockLen = curColNum * sizeof(T);
        int64_t offset = (blockNStart + rowIdx) * cLength + ubIdx * ubFormer;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer1.ReinterpretCast<T>(), invstdGm[offset], intriParams, padParams);
        } else {
            DataCopyPad(buffer1.ReinterpretCast<T>()[ubFormer], invstdGm[offset], intriParams, padParams);
        }

        queue1.EnQue(buffer1);
        buffer1 = queue1.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer1, buffer1.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, ubFormer);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeGlobalMean(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        CopyMeanData(rowIdx, ubIdx, curColNum);
        float countValue = countsGm.GetValue(blockNStart + rowIdx);
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(curColNum * sizeof(float)), 0, 0, 0};

        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event0);
        WaitFlag<HardEvent::MTE2_V>(event0);

        Muls(buffer0, buffer0, countValue, curColNum);
        PipeBarrier<PIPE_V>();

        buffer1 = queue1.AllocTensor<float>();
        Duplicate(buffer1, countsSumScalar, curColNum);
        PipeBarrier<PIPE_V>();

        Div(buffer0, buffer0, buffer1, curColNum);
        PipeBarrier<PIPE_V>();

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event1);
        WaitFlag<HardEvent::V_MTE3>(event1);

        SetAtomicAdd<float>();
        DataCopyPad(reduceWorkspaceGM[ubIdx * ubFormer], buffer0, intriParams); // workspace part sum meanAll
        SetAtomicNone();

        queue0.FreeTensor(buffer0);
        queue1.FreeTensor(buffer1);
    }

    __aicore__ inline void ComputeGlobalVar(const int64_t rowIdx, const int64_t ubIdx, const int64_t curColNum)
    {
        CopyInvstdData(rowIdx, ubIdx, curColNum);
        float countValue = countsGm.GetValue(blockNStart + rowIdx);
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(cAlign * sizeof(float)), 0, 0, 0};

        buffer0 = queue0.AllocTensor<float>();
        Duplicate(buffer0, static_cast<float>(1.0f), curColNum);
        PipeBarrier<PIPE_V>();

        Div(buffer1, buffer0, buffer1, curColNum);
        PipeBarrier<PIPE_V>();
        queue0.FreeTensor(buffer0);

        CopyMeanData(rowIdx, ubIdx, curColNum);

        Mul(buffer1, buffer1, buffer1, curColNum);
        PipeBarrier<PIPE_V>();

        Adds(buffer1, buffer1, -eps, curColNum);
        PipeBarrier<PIPE_V>();

        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);

        Sub(buffer0, buffer0, buffer2, curColNum);
        PipeBarrier<PIPE_V>();

        Mul(buffer0, buffer0, buffer0, curColNum);
        PipeBarrier<PIPE_V>();

        Add(buffer0, buffer0, buffer1, curColNum);
        PipeBarrier<PIPE_V>();

        Muls(buffer0, buffer0, countValue, curColNum);
        PipeBarrier<PIPE_V>();

        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event0);
        WaitFlag<HardEvent::V_MTE3>(event0);
        SetAtomicAdd<float>();
        DataCopyPad(reduceWorkspaceGM[ubIdx * ubFormer], buffer0, intriParams);
        SetAtomicNone();
        queue0.FreeTensor(buffer0);
        queue1.FreeTensor(buffer1);
    }

    __aicore__ inline void ComputeGlobalVarDterministic()
    {
        SyncBatchNormGatherStatsFusedDeterminsticCompute<T> op;
        op.initBuffer1(pipe, varAllResultworkspaceGM, workspaceGMOri, WORKSPACE_NUM);
        op.InvstdAllDeterministic(cAlign, blockNum, cLength);
    }

    __aicore__ inline void ComputeGlobalMeanDterministicFun()
    {
        SyncBatchNormGatherStatsFusedDeterminsticCompute<T> op;
        op.initBuffer2(
            pipe, meanAllResultworkspaceGM, meanAllOutGm, runningMeanGm, runningMeanOutGm, workspaceGMOri,
            WORKSPACE_NUM, momentum);
        op.MeanAllDeterministicWithRunningMeanUpdate(cAlign, blockNum, cLength);
    }

    __aicore__ inline void ComputeGlobalInvstdWithRunningVarFun()
    {
        SyncBatchNormGatherStatsFusedRunningCompute<T> op;
        op.initBuffer(
            pipe, runningVarOutGm, invstdAllOutGm, runningVarGm, varAllResultworkspaceGM, countsSumScalar, momentum,
            eps, 1);
        op.FinalComputeProcess(cAlign, 1, cLength);
    }

private:
    TPipe pipe;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;
    constexpr static uint16_t WORKSPACE_NUM = 1;
    int64_t nLength;
    int64_t cLength;
    int64_t blockNum;
    int64_t ubFormer;
    int64_t ubLoop;
    int64_t ubTail;
    int64_t curNLength;
    float momentum;
    float countsSumScalar;
    float eps;
    float unbiasCountsSum;
    int64_t blockNStart;
    int64_t cAlign;

    GlobalTensor<T> meanGm;
    GlobalTensor<T> invstdGm;
    GlobalTensor<float> countsGm;
    GlobalTensor<T> runningMeanGm;
    GlobalTensor<T> runningVarGm;
    GlobalTensor<T> meanAllOutGm;
    GlobalTensor<T> invstdAllOutGm;
    GlobalTensor<T> runningMeanOutGm;
    GlobalTensor<T> runningVarOutGm;

    GlobalTensor<float> workspaceGMOri;
    GlobalTensor<float> meanAllResultworkspaceGM;
    GlobalTensor<float> varAllResultworkspaceGM;
    GlobalTensor<float> reduceWorkspaceGM;
    GlobalTensor<float> countsReduceWorkspace;

    LocalTensor<float> buffer0;
    LocalTensor<float> buffer1;
    LocalTensor<float> buffer2;

    TQue<QuePosition::VECIN, 1> queue0;
    TQue<QuePosition::VECIN, 1> queue1;
    TQue<QuePosition::VECIN, 1> queue2;
};
} // namespace SyncBatchNormGatherStatsFused
#endif // SYNC_BATCH_NORM_GATHER_STATS_FUSED_FIRST_AXIS_WORKSPACE_H