/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused_first_axis_common.h
 * \brief First axis common implementation for SyncBatchNormGatherStatsFused
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_FIRST_AXIS_COMMON_H
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_FIRST_AXIS_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "sync_batch_norm_gather_stats_fused_deterministic.h"
#include "sync_batch_norm_gather_stats_fused_compute.h"
#include "sync_batch_norm_gather_stats_fused_counts_sum.h"

namespace SyncBatchNormGatherStatsFused {
using namespace AscendC;

constexpr int32_t FA_COMMON_B32_BLOCK_SIZE = 8;
constexpr int32_t FA_COMMON_B16_BLOCK_SIZE = 16;
constexpr int32_t FA_COMMON_B32_REPEAT_SIZE = 64;
constexpr int32_t FA_COMMON_CONSTANT_EIGHT = 8;
constexpr int32_t FA_COMMON_MAX_REPEAT_BRCB = 255;
constexpr int32_t FA_COMMON_VC_MAX_REPEAT_BRCB = 248;

template <typename T>
class SyncBatchNormGatherStatsFusedFirstAxisCommon {
public:
    __aicore__ inline SyncBatchNormGatherStatsFusedFirstAxisCommon(){};

    __aicore__ inline void Init(GM_ADDR mean, GM_ADDR invstd, GM_ADDR counts, GM_ADDR runningMean, GM_ADDR runningVar,
                                GM_ADDR meanAllOut, GM_ADDR invstdAllOut, GM_ADDR runningMeanOut, GM_ADDR runningVarOut,
                                GM_ADDR workspace,
                                const SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon* tilingData, TPipe& pipeIn)
    {
        pipe = pipeIn;
        nLength = tilingData->nLength;
        cLength = tilingData->cLength;
        blockNum = tilingData->blockNum;
        momentum = tilingData->momentum;
        eps = tilingData->eps;
        cAlign = tilingData->cAlignM;
        blockNStart = GetBlockIdx() * tilingData->blockFormer;
        curNLength = (GetBlockIdx() == blockNum - 1) ? tilingData->blockTail : tilingData->blockFormer;
        wholeBufferElemNums = tilingData->wholeBufferElemNums;

        countsGm.SetGlobalBuffer((__gm__ float*)counts, nLength);
        meanAllOutGm.SetGlobalBuffer((__gm__ T*)meanAllOut, cLength);
        invstdAllOutGm.SetGlobalBuffer((__gm__ T*)invstdAllOut, cLength);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T*)runningMeanOut, cLength);
        runningVarOutGm.SetGlobalBuffer((__gm__ T*)runningVarOut, cLength);
        runningMeanGm.SetGlobalBuffer((__gm__ T*)runningMean, cLength);
        runningVarGm.SetGlobalBuffer((__gm__ T*)runningVar, cLength);

        int64_t wsLenPerBlock = cAlign * WORKSPACE_NUM;
        meanAllResultworkspaceGM.SetGlobalBuffer((__gm__ float*)workspace, cAlign);
        varAllResultworkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + cAlign, cAlign);
        workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace + 2 * cAlign, wsLenPerBlock * blockNum);
        countsReduceWorkspace.SetGlobalBuffer((__gm__ float*)workspace + (2 + blockNum * WORKSPACE_NUM) * cAlign,
                                              GetBlockNum());
        InitOutput<float>(countsReduceWorkspace[GetBlockIdx()], 1, 0.0f);

        if (GetBlockIdx() < tilingData->blockNum) {
            ubFormer = tilingData->ubFormer;
            ubLoop = (GetBlockIdx() == blockNum - 1) ? tilingData->ubLoopOfTailBlock : tilingData->ubLoopOfFormerBlock;
            ubTail = (GetBlockIdx() == blockNum - 1) ? tilingData->ubTailOfTailBlock : tilingData->ubTailOfFormerBlock;
            meanGm.SetGlobalBuffer((__gm__ T*)mean, nLength * cLength);
            invstdGm.SetGlobalBuffer((__gm__ T*)invstd, nLength * cLength);
            reduceWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + (2 + GetBlockIdx()) * cAlign, cAlign);
            InitOutput<float>(reduceWorkspaceGM, cAlign, 0.0f);
        }
    }

    __aicore__ inline void initBuffer(const SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon* tilingData)
    {
        pipe.InitBuffer(queue0_, 1, tilingData->wholeBufferByteSize);
        pipe.InitBuffer(queue1_, 1, tilingData->wholeBufferByteSize);
        pipe.InitBuffer(queue2_, 1, tilingData->nBufferByteSize);
        pipe.InitBuffer(queue3_, 1, tilingData->cBufferByteSize);
        pipe.InitBuffer(queue4_, 1, tilingData->cBufferByteSize);
        pipe.InitBuffer(tmpTensor0_, tilingData->nBrcbBufferByteSize);
    }

    __aicore__ inline void Process(const SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon* tilingData)
    {
        ComputeCountsSum();
        SyncAll();
        pipe.Reset();
        if (GetBlockIdx() < blockNum) {
            initBuffer(tilingData);

            buffer3_ = queue3_.AllocTensor<float>();
            Duplicate(buffer3_, static_cast<float>(0.0), cAlign);
            buffer5_ = tmpTensor0_.Get<float>();
            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                int64_t curRowNum = ubIdx == ubLoop - 1 ? ubTail : ubFormer;
                ComputeGlobalMean(ubIdx, curRowNum);
            }
        }
        SyncAll();
        pipe.Reset();
        ComputeGlobalMeanDterministic();
        SyncAll();
        pipe.Reset();
        if (GetBlockIdx() < blockNum) {
            unbiasCountsSum = countsSumScalar - 1.0f;
            initBuffer(tilingData);

            buffer4_ = queue4_.AllocTensor<float>();
            Duplicate(buffer4_, static_cast<float>(0.0), cAlign);
            buffer5_ = tmpTensor0_.Get<float>();

            CopyReduceWorkspaceResult();
            InitOutput<float>(reduceWorkspaceGM, cAlign, 0.0f);
            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                int64_t curRowNum = ubIdx == ubLoop - 1 ? ubTail : ubFormer;
                ComputeGlobalVar(ubIdx, curRowNum);
            }
        }
        SyncAll();
        pipe.Reset();
        ComputeGlobalVarDterministic();
        SyncAll();
        pipe.Reset();
        ComputeGlobalInvstdWithRunningVar();
    }

private:
    __aicore__ inline void ComputeCountsSum()
    {
        SyncBatchNormGatherStatsFusedCountsSum<T> op;
        op.initBuffer(pipe, countsGm, countsReduceWorkspace);
        int64_t nAlign;

        nAlign = ((nLength + 8 - 1) / 8) * 8;

        op.FinalComputeProcess(nAlign, nLength, countsSumScalar);
    }
    __aicore__ inline void CopyReduceWorkspaceResult()
    {
        buffer3_ = queue3_.AllocTensor<float>();
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParams;

        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        intriParams.blockCount = 1;
        intriParams.blockLen = cAlign * sizeof(float);

        DataCopyPad(buffer3_, meanAllResultworkspaceGM, intriParams, padParams);

        queue3_.EnQue(buffer3_);
        buffer3_ = queue3_.DeQue<float>();
    }

    __aicore__ inline void ComputeGlobalMeanDterministic()
    {
        SyncBatchNormGatherStatsFusedDeterminsticCompute<T> op;
        op.initBuffer2(pipe, meanAllResultworkspaceGM, meanAllOutGm, runningMeanGm, runningMeanOutGm, workspaceGMOri,
                       WORKSPACE_NUM, momentum);
        op.MeanAllDeterministicWithRunningMeanUpdate(cAlign, blockNum, cLength);
    }

    __aicore__ inline void ComputeGlobalVarDterministic()
    {
        SyncBatchNormGatherStatsFusedDeterminsticCompute<T> op;
        op.initBuffer1(pipe, varAllResultworkspaceGM, workspaceGMOri, WORKSPACE_NUM);
        op.InvstdAllDeterministic(cAlign, blockNum, cLength);
    }

    __aicore__ inline void ComputeGlobalInvstdWithRunningVar()
    {
        SyncBatchNormGatherStatsFusedRunningCompute<T> op;
        op.initBuffer(pipe, runningVarOutGm, invstdAllOutGm, runningVarGm, varAllResultworkspaceGM, countsSumScalar,
                      momentum, eps, 1);
        op.FinalComputeProcess(cAlign, 1, cLength);
    }

private:
    __aicore__ inline void CopyMeanData(const int64_t ubIdx, const int64_t curRowNum);
    __aicore__ inline void CopyInvstdData(const int64_t ubIdx, const int64_t curRowNum);
    __aicore__ inline void CopyCount(const int64_t ubIdx, const int64_t curRowNum);

    __aicore__ inline void ComputeGlobalMean(const int64_t ubIdx, const int64_t curRowNum);
    __aicore__ inline void ComputeGlobalVar(const int64_t ubIdx, const int64_t curRowNum);

    template <typename dType>
    __aicore__ inline void BlockBroadcast(const LocalTensor<dType>& dst, const LocalTensor<dType>& src,
                                          const int64_t curRowsNum);

    __aicore__ inline void BinElemWithInlinedNLastBrcFP32(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                          const LocalTensor<float>& src1, const int64_t curRowsNum,
                                                          void (*func)(const LocalTensor<float>&,
                                                                       const LocalTensor<float>&,
                                                                       const LocalTensor<float>&, uint64_t, uint8_t,
                                                                       const BinaryRepeatParams&));

    __aicore__ inline void NlastReduceSumLargeStride(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                     const int64_t curRowsNum);
    __aicore__ inline void LastBrcFP32LargeStride(const LocalTensor<float>& output, const LocalTensor<float>& input0,
                                                  const LocalTensor<float>& input1, const int64_t curRowsNum,
                                                  void (*func)(const LocalTensor<float>&, const LocalTensor<float>&,
                                                               const LocalTensor<float>&, uint64_t, uint8_t,
                                                               const BinaryRepeatParams&));
    __aicore__ inline void NlastReduceSum(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                          const int64_t curRowsNum);

    __aicore__ inline void BinElemNLastBrcLargeStride(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                      const LocalTensor<float>& src1, const int64_t curRowsNum,
                                                      void (*func)(const LocalTensor<float>&, const LocalTensor<float>&,
                                                                   const LocalTensor<float>&, uint64_t, uint8_t,
                                                                   const BinaryRepeatParams&));

    __aicore__ inline void BinElemWithInlinedLastBrcFP32(
        const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
        const int64_t rows,
        void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
                     const BinaryRepeatParams&));

    __aicore__ inline void LastReduceSum(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                         const LocalTensor<float>& tmp, const int64_t curRowsNum,
                                         const int64_t curColNum);

    __aicore__ inline void LastReduceSumLargeStride(const LocalTensor<float>& tmp, const LocalTensor<float>& src,
                                                    const int64_t curRowsNum, const int64_t curColNum);

private:
    TPipe pipe;
    static constexpr DivConfig divConfig = {DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;
    constexpr static int64_t WORKSPACE_NUM = 1;
    int64_t nLength;
    int64_t cLength;
    int64_t cAlign;
    int64_t blockNum;
    int64_t ubFormer;
    int64_t ubLoop;
    int64_t ubTail;
    int64_t wholeBufferElemNums;
    float momentum;
    float countsSumScalar;
    float eps;
    float unbiasCountsSum;
    int64_t blockNStart;
    int64_t curNLength;

    GlobalTensor<T> meanGm;
    GlobalTensor<T> invstdGm;
    GlobalTensor<float> countsGm;
    GlobalTensor<T> runningVarGm;
    GlobalTensor<T> runningMeanGm;
    GlobalTensor<T> meanAllOutGm;
    GlobalTensor<T> invstdAllOutGm;
    GlobalTensor<T> runningMeanOutGm;
    GlobalTensor<T> runningVarOutGm;
    GlobalTensor<float> workspaceGMOri;
    GlobalTensor<float> meanAllResultworkspaceGM;
    GlobalTensor<float> varAllResultworkspaceGM;
    GlobalTensor<float> reduceWorkspaceGM;
    GlobalTensor<float> countsReduceWorkspace;

    LocalTensor<float> buffer0_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;
    LocalTensor<float> buffer5_;

    TQue<QuePosition::VECIN, 1> queue0_;
    TQue<QuePosition::VECIN, 1> queue1_;
    TQue<QuePosition::VECIN, 1> queue2_;
    TQue<QuePosition::VECOUT, 1> queue3_;
    TQue<QuePosition::VECOUT, 1> queue4_;
    TBuf<> tmpTensor0_;
};

} // namespace SyncBatchNormGatherStatsFused

#include "sync_batch_norm_gather_stats_fused_first_axis_common.inl"

#endif // SYNC_BATCH_NORM_GATHER_STATS_FUSED_FIRST_AXIS_COMMON_H