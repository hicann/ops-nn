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
 * \file sync_batch_norm_gather_stats_fused_common.h
 * \brief Common implementation for SyncBatchNormGatherStatsFused (non-first axis)
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_COMMON_H
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "sync_batch_norm_gather_stats_fused_counts_sum.h"

namespace SyncBatchNormGatherStatsFused {
using namespace AscendC;

constexpr int32_t COMMON_B32_BLOCK_SIZE = 8;
constexpr int32_t COMMON_B16_BLOCK_SIZE = 16;
constexpr int32_t COMMON_B32_REPEAT_SIZE = 64;
constexpr int32_t COMMON_CONSTANT_EIGHT = 8;
constexpr int32_t COMMON_MAX_REPEAT = 255;
constexpr int32_t COMMON_VC_MAX_REPEAT = 248;

template <typename T>
class SyncBatchNormGatherStatsFusedCommon {
public:
    __aicore__ inline SyncBatchNormGatherStatsFusedCommon(){};

    __aicore__ inline void Init(
        GM_ADDR mean, GM_ADDR invstd, GM_ADDR counts, GM_ADDR runningMean, GM_ADDR runningVar, GM_ADDR meanAllOut,
        GM_ADDR invstdAllOut, GM_ADDR runningMeanOut, GM_ADDR runningVarOut, GM_ADDR workspace,
        const SyncBatchNormGatherStatsFusedTilingDataCommon* tilingData, TPipe& pipeIn)
    {
        pipe = pipeIn;
        nLength = tilingData->nLength;
        cLength = tilingData->cLength;
        blockNum = tilingData->blockNum;
        momentum = tilingData->momentum;
        eps = tilingData->eps;
        cAlign = (GetBlockIdx() == blockNum - 1) ? tilingData->cTailAlignM_ : tilingData->cFormerAlignM_;
        cAlignM = (GetBlockIdx() == blockNum - 1) ? tilingData->cTailAlignM_ : tilingData->cFormerAlignM_;
        curCLength = (GetBlockIdx() == blockNum - 1) ? tilingData->blockTail : tilingData->blockFormer;
        wholeBufferElemNums = tilingData->wholeBufferElemNums;

        countsGm.SetGlobalBuffer((__gm__ float*)counts, nLength);
        countsReduceWorkspace.SetGlobalBuffer((__gm__ float*)workspace, GetBlockNum());
        InitOutput<float>(countsReduceWorkspace[GetBlockIdx()], 1, 0.0f);
        if (GetBlockIdx() < tilingData->blockNum) {
            blockCStart = GetBlockIdx() * tilingData->blockFormer;
            ubFormer = tilingData->ubFormer;
            ubLoop = tilingData->ubLoop;
            ubTail = tilingData->ubTail;

            meanGm.SetGlobalBuffer((__gm__ T*)mean, nLength * cLength);
            invstdGm.SetGlobalBuffer((__gm__ T*)invstd, nLength * cLength);
            runningMeanGm.SetGlobalBuffer((__gm__ T*)runningMean, cLength);
            runningVarGm.SetGlobalBuffer((__gm__ T*)runningVar, cLength);
            meanAllOutGm.SetGlobalBuffer((__gm__ T*)meanAllOut, cLength);
            invstdAllOutGm.SetGlobalBuffer((__gm__ T*)invstdAllOut, cLength);
            runningMeanOutGm.SetGlobalBuffer((__gm__ T*)runningMeanOut, cLength);
            runningVarOutGm.SetGlobalBuffer((__gm__ T*)runningVarOut, cLength);
        }
    }

    __aicore__ inline void initBuffer(const SyncBatchNormGatherStatsFusedTilingDataCommon* tilingData)
    {
        pipe.InitBuffer(queue0_, 1, tilingData->wholeBufferByteSize);
        pipe.InitBuffer(queue1_, 1, tilingData->wholeBufferByteSize);
        pipe.InitBuffer(queue2_, 1, tilingData->nBufferByteSize);
        pipe.InitBuffer(queue3_, 1, tilingData->cBufferByteSize);
        pipe.InitBuffer(queue4_, 1, tilingData->cBufferByteSize);
        pipe.InitBuffer(queue5_, 1, tilingData->cBufferByteSize);
        pipe.InitBuffer(queue6_, 1, tilingData->cBufferByteSize);
        pipe.InitBuffer(tmpTensor0_, tilingData->nBrcbBufferByteSize);
    }

    __aicore__ inline void Process(const SyncBatchNormGatherStatsFusedTilingDataCommon* tilingData)
    {
        ComputeCountsSum();
        SyncAll();
        pipe.Reset();

        if (GetBlockIdx() < blockNum) {
            initBuffer(tilingData);
            unbiasCountsSum = countsSumScalar - 1.0f;

            buffer3_ = queue3_.AllocTensor<float>();
            Duplicate(buffer3_, static_cast<float>(0.0f), cAlign);
            buffer4_ = queue4_.AllocTensor<float>();
            Duplicate(buffer4_, static_cast<float>(0.0f), cAlign);
            buffer7_ = tmpTensor0_.Get<float>();

            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                int64_t curRowNum = ubIdx == ubLoop - 1 ? ubTail : ubFormer;
                ComputeGlobalMean(ubIdx, curRowNum);
            }

            for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                int64_t curRowNum = ubIdx == ubLoop - 1 ? ubTail : ubFormer;
                ComputeGlobalVar(ubIdx, curRowNum);
            }
            ComputeGlobalInvstdWithRunningOut();
            ResultCoyOut();
        }
    }

    __aicore__ inline void ComputeCountsSum()
    {
        SyncBatchNormGatherStatsFusedCountsSum<T> op;
        op.initBuffer(pipe, countsGm, countsReduceWorkspace);
        int64_t nAlign;

        nAlign = ((nLength + 7) / 8) * 8;

        op.FinalComputeProcess(nAlign, nLength, countsSumScalar);
    }

    __aicore__ inline void CopyMean(const int64_t ubIdx, const int64_t curRowNum)
    {
        buffer0_ = queue0_.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParamMean;
        intriParamMean.blockCount = curRowNum;
        intriParamMean.srcStride = (cLength - curCLength) * sizeof(T);
        intriParamMean.dstStride = 0;

        if (likely(cAlign == curCLength)) {
            intriParamMean.blockLen = cAlign * sizeof(T);
        } else {
            intriParamMean.blockLen = curCLength * sizeof(T);
            padParams.isPad = true;
            padParams.rightPadding = cAlignM - curCLength;
        }
        int64_t offset = blockCStart + ubIdx * ubFormer * cLength;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0_.ReinterpretCast<T>(), meanGm[offset], intriParamMean, padParams);
        } else {
            DataCopyPad(buffer0_.ReinterpretCast<T>()[wholeBufferElemNums], meanGm[offset], intriParamMean, padParams);
        }

        queue0_.EnQue(buffer0_);
        buffer0_ = queue0_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(
                buffer0_, buffer0_.ReinterpretCast<T>()[wholeBufferElemNums], RoundMode::CAST_NONE,
                wholeBufferElemNums);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyInvstd(const int64_t ubIdx, const int64_t curRowNum)
    {
        buffer1_ = queue1_.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParam;
        intriParam.blockCount = curRowNum;
        intriParam.srcStride = (cLength - curCLength) * sizeof(T);
        intriParam.dstStride = 0;

        if (likely(cAlign == curCLength)) {
            intriParam.blockLen = cAlign * sizeof(T);
        } else {
            intriParam.blockLen = curCLength * sizeof(T);
            padParams.isPad = true;
            padParams.rightPadding = cAlignM - curCLength;
        }

        int64_t offset = blockCStart + ubIdx * ubFormer * cLength;
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer1_.ReinterpretCast<T>(), invstdGm[offset], intriParam, padParams);
        } else {
            DataCopyPad(buffer1_.ReinterpretCast<T>()[wholeBufferElemNums], invstdGm[offset], intriParam, padParams);
        }

        queue1_.EnQue(buffer1_);
        buffer1_ = queue1_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(
                buffer1_, buffer1_.ReinterpretCast<T>()[wholeBufferElemNums], RoundMode::CAST_NONE,
                wholeBufferElemNums);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyCount(const int64_t ubIdx, const int64_t curRowNum)
    {
        buffer2_ = queue2_.AllocTensor<float>();
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyExtParams IntriParam;
        IntriParam.blockCount = 1;
        IntriParam.blockLen = curRowNum * sizeof(float);
        IntriParam.srcStride = 0;
        IntriParam.dstStride = 0;

        DataCopyPad(buffer2_, countsGm[ubIdx * ubFormer], IntriParam, padParams);
        queue2_.EnQue(buffer2_);
        buffer2_ = queue2_.DeQue<float>();
    }

    __aicore__ inline void ComputeGlobalMean(const int64_t ubIdx, const int64_t curRowNum)
    {
        CopyMean(ubIdx, curRowNum);
        CopyCount(ubIdx, curRowNum);
        BlockBroadcast<float>(buffer7_, buffer2_, curRowNum);
        PipeBarrier<PIPE_V>();

        BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer7_, curRowNum, Mul);
        PipeBarrier<PIPE_V>();

        buffer1_ = queue1_.AllocTensor<float>();
        Duplicate(buffer1_, countsSumScalar, curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer0_, buffer0_, buffer1_, curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        NlastReduceSum(buffer3_, buffer0_, curRowNum);
        PipeBarrier<PIPE_V>();

        queue0_.FreeTensor(buffer0_);
        queue1_.FreeTensor(buffer1_);
        queue2_.FreeTensor(buffer2_);
    }

    __aicore__ inline void ComputeGlobalVar(const int64_t ubIdx, const int64_t curRowNum)
    {
        CopyCount(ubIdx, curRowNum);
        CopyInvstd(ubIdx, curRowNum);

        buffer0_ = queue0_.AllocTensor<float>();
        Duplicate(buffer0_, static_cast<float>(1.0), curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer1_, buffer0_, buffer1_, curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        queue0_.FreeTensor(buffer0_);
        CopyMean(ubIdx, curRowNum);

        Mul(buffer1_, buffer1_, buffer1_, curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        Adds(buffer1_, buffer1_, -eps, curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        BinElemWithInlinedNLastBrcFP32(buffer0_, buffer0_, buffer3_, curRowNum, Sub);
        PipeBarrier<PIPE_V>();

        Mul(buffer0_, buffer0_, buffer0_, curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        Add(buffer0_, buffer0_, buffer1_, curRowNum * cAlign);
        PipeBarrier<PIPE_V>();

        BlockBroadcast<float>(buffer7_, buffer2_, curRowNum);
        PipeBarrier<PIPE_V>();

        BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer7_, curRowNum, Mul);
        PipeBarrier<PIPE_V>();

        NlastReduceSum(buffer4_, buffer0_, curRowNum);
        PipeBarrier<PIPE_V>();

        queue0_.FreeTensor(buffer0_);
        queue1_.FreeTensor(buffer1_);
        queue2_.FreeTensor(buffer2_);
    }

    __aicore__ inline void CopyRunning(const GlobalTensor<T>& runningGm)
    {
        buffer0_ = queue0_.AllocTensor<float>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        if (likely(cAlignM == curCLength)) {
            intriParams.blockLen = cAlign * sizeof(T);
        } else {
            intriParams.blockLen = curCLength * sizeof(T);
            padParams.isPad = true;
            padParams.rightPadding = cAlignM - curCLength;
        }

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0_.ReinterpretCast<T>(), runningGm[blockCStart], intriParams, padParams);
        } else {
            DataCopyPad(buffer0_.ReinterpretCast<T>()[cAlignM], runningGm[blockCStart], intriParams, padParams);
        }

        queue0_.EnQue(buffer0_);
        buffer0_ = queue0_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer0_, buffer0_.ReinterpretCast<T>()[cAlignM], RoundMode::CAST_NONE, cAlign);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeGlobalInvstdWithRunningOut()
    {
        buffer5_ = queue5_.AllocTensor<float>();
        buffer6_ = queue6_.AllocTensor<float>();
        CopyRunning(runningVarGm);

        Muls(buffer0_, buffer0_, 1.0f - momentum, cAlign);
        PipeBarrier<PIPE_V>();

        buffer1_ = queue1_.AllocTensor<float>();
        Duplicate(buffer1_, unbiasCountsSum, cAlign);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer6_, buffer4_, buffer1_, cAlign);
        PipeBarrier<PIPE_V>();

        Muls(buffer6_, buffer6_, momentum, cAlign);
        PipeBarrier<PIPE_V>();

        Add(buffer6_, buffer6_, buffer0_, cAlign);
        PipeBarrier<PIPE_V>();

        queue0_.FreeTensor(buffer0_);
        CopyRunning(runningMeanGm);

        Muls(buffer1_, buffer3_, momentum, cAlign);
        PipeBarrier<PIPE_V>();

        Muls(buffer5_, buffer0_, 1.0f - momentum, cAlign);
        PipeBarrier<PIPE_V>();

        Add(buffer5_, buffer5_, buffer1_, cAlign);
        PipeBarrier<PIPE_V>();

        Duplicate(buffer0_, countsSumScalar, cAlign);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer4_, buffer4_, buffer0_, cAlign);
        PipeBarrier<PIPE_V>();

        Adds(buffer4_, buffer4_, eps, cAlign);
        PipeBarrier<PIPE_V>();

        Sqrt(buffer4_, buffer4_, cAlign);
        PipeBarrier<PIPE_V>();

        Duplicate(buffer0_, static_cast<float>(1.0), cAlign);
        PipeBarrier<PIPE_V>();

        Div<float, divConfig>(buffer4_, buffer0_, buffer4_, cAlign);
        PipeBarrier<PIPE_V>();

        queue0_.FreeTensor(buffer0_);
        queue1_.FreeTensor(buffer1_);
    }

    __aicore__ inline void ResultCoyOut()
    {
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer3_.ReinterpretCast<T>(), buffer3_, RoundMode::CAST_RINT, cAlign);
            PipeBarrier<PIPE_V>();
            Cast(buffer4_.ReinterpretCast<T>(), buffer4_, RoundMode::CAST_RINT, cAlign);
            PipeBarrier<PIPE_V>();
            Cast(buffer5_.ReinterpretCast<T>(), buffer5_, RoundMode::CAST_RINT, cAlign);
            PipeBarrier<PIPE_V>();
            Cast(buffer6_.ReinterpretCast<T>(), buffer6_, RoundMode::CAST_RINT, cAlign);
            PipeBarrier<PIPE_V>();
        }
        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event0);
        WaitFlag<HardEvent::V_MTE3>(event0);

        DataCopyExtParams intriParamsT;
        intriParamsT.blockCount = 1;
        intriParamsT.blockLen = curCLength * sizeof(T);
        intriParamsT.srcStride = 0;
        intriParamsT.dstStride = 0;

        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);

        DataCopyPad(meanAllOutGm[blockCStart], buffer3_.ReinterpretCast<T>(), intriParamsT);
        DataCopyPad(invstdAllOutGm[blockCStart], buffer4_.ReinterpretCast<T>(), intriParamsT);
        DataCopyPad(runningMeanOutGm[blockCStart], buffer5_.ReinterpretCast<T>(), intriParamsT);
        DataCopyPad(runningVarOutGm[blockCStart], buffer6_.ReinterpretCast<T>(), intriParamsT);
        queue3_.FreeTensor(buffer3_);
        queue4_.FreeTensor(buffer4_);
        queue5_.FreeTensor(buffer5_);
        queue6_.FreeTensor(buffer6_);
    }

private:
    template <typename dType>
    __aicore__ inline void BlockBroadcast(
        const LocalTensor<dType>& dst, const LocalTensor<dType>& src, const int64_t curRowsNum);

    __aicore__ inline void BinElemNLastBrcLargeStride(
        const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
        const int64_t curRowsNum,
        void (*func)(
            const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
            const BinaryRepeatParams&));

    __aicore__ inline void BinElemWithInlinedNLastBrcFP32(
        const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
        const int64_t curRowsNum,
        void (*func)(
            const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
            const BinaryRepeatParams&));

    __aicore__ inline void LastBrcFP32LargeStride(
        const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
        const int64_t curRowsNum,
        void (*func)(
            const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
            const BinaryRepeatParams&));

    __aicore__ inline void BinElemWithInlinedLastBrcFP32(
        const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
        const int64_t rows,
        void (*func)(
            const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
            const BinaryRepeatParams&));

    __aicore__ inline void NlastReduceSumLargeStride(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum);

    __aicore__ inline void NlastReduceSum(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum);

    __aicore__ inline void LastReduceSumLargeStride(
        const LocalTensor<float>& tmp, const LocalTensor<float>& src, const int64_t curRowsNum,
        const int64_t curColNum);

    __aicore__ inline void LastReduceSum(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmp,
        const int64_t curRowsNum, const int64_t curColNum);

private:
    TPipe pipe;
    static constexpr DivConfig divConfig = {DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;
    int64_t nLength;
    int64_t cLength;
    int64_t cAlign;
    int64_t cAlignM;
    int64_t blockNum;
    int64_t ubFormer;
    int64_t ubLoop;
    int64_t ubTail;
    int64_t wholeBufferElemNums;
    float momentum;
    float countsSumScalar;
    float eps;
    float unbiasCountsSum;
    int64_t blockCStart;
    int64_t curCLength;

    GlobalTensor<T> meanGm;
    GlobalTensor<T> invstdGm;
    GlobalTensor<float> countsGm;
    GlobalTensor<T> runningMeanGm;
    GlobalTensor<T> runningVarGm;
    GlobalTensor<T> meanAllOutGm;
    GlobalTensor<T> invstdAllOutGm;
    GlobalTensor<T> runningMeanOutGm;
    GlobalTensor<T> runningVarOutGm;
    GlobalTensor<float> countsReduceWorkspace;

    LocalTensor<float> buffer0_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;
    LocalTensor<float> buffer5_;
    LocalTensor<float> buffer6_;
    LocalTensor<float> buffer7_;

    TQue<QuePosition::VECIN, 1> queue0_;
    TQue<QuePosition::VECIN, 1> queue1_;
    TQue<QuePosition::VECIN, 1> queue2_;
    TQue<QuePosition::VECOUT, 1> queue3_;
    TQue<QuePosition::VECOUT, 1> queue4_;
    TQue<QuePosition::VECOUT, 1> queue5_;
    TQue<QuePosition::VECOUT, 1> queue6_;
    TBuf<> tmpTensor0_;
};

} // namespace SyncBatchNormGatherStatsFused

#include "sync_batch_norm_gather_stats_fused_common.inl"

#endif // SYNC_BATCH_NORM_GATHER_STATS_FUSED_COMMON_H