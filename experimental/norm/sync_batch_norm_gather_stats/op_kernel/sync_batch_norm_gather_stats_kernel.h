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
 * \file sync_batch_norm_gather_stats_kernel.h
 * \brief 910B(__CCE_AICORE__ == 220) 向量实现；310P 标量见 arch20/。
 */

#include <type_traits>
#include "kernel_operator.h"
#include "common/sync_batch_norm_gather_stats_tiling_key.h"
#include "common/sync_batch_norm_gather_stats_kernel_base.h"

using namespace AscendC;

namespace SyncBatchNormGatherStats {
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t OUT_QUEUE_DEPTH = 4;
constexpr uint32_t SAMPLE_COUNT_TILE_N = 256;
constexpr uint32_t FP32_C_FIRST_GROUP_N = 16;

template <typename T>
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return CeilDiv<T>(value, align) * align;
}

template <typename T, typename CountT>
class SyncBatchNormGatherStatsVector : public SyncBatchNormGatherStatsKernelBase<T, CountT> {
    SYNC_BATCH_NORM_GATHER_STATS_USE_BASE_MEMBERS;

public:
    __aicore__ inline SyncBatchNormGatherStatsVector(TPipe* pipeIn) : pipe(pipeIn) {}

    __aicore__ inline void InitFull(const SyncBatchNormGatherStatsTilingData* tilingData)
    {
        SetFullTiling(tilingData);
        InitBuffers(nFactor, cFactor);
    }

    __aicore__ inline void InitNotFull(const SyncBatchNormGatherStatsNNotFullLoadTilingData* tilingData)
    {
        SetNotFullTiling(tilingData);
        InitBuffers(nFactor, cFactor);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockDim || nTotal == 0 || cLen == 0 || cLoop == 0) {
            return;
        }

        // 910B(220) 向量实现：cLoop==1 且 N 小，走单 tile 快路径；否则通用 C tile 循环。
        if (cLoop == 1 && nTotal <= nFactor && nTotal <= SCALAR_COUNT_N_MAX) {
            ProcessSingleTileFast();
            return;
        }

        const float countAll = ReduceSampleCount();
        for (uint64_t cLoopIdx = 0; cLoopIdx < cLoop; ++cLoopIdx) {
            const uint64_t currentCNum = (cLoopIdx == cLoop - 1) ? cTail : cFactor;
            if (currentCNum == 0) {
                continue;
            }
            const uint64_t cOffset = cBase + cLoopIdx * cFactor;
            ProcessOneCTile(cOffset, currentCNum, countAll);
        }
    }

private:
    __aicore__ inline void ProcessSingleTileFast()
    {
        const uint64_t currentCNum = cTail;
        const uint64_t cAlignT = AlignUp<T>(currentCNum, BLOCK_BYTES / sizeof(T));

        DataCopyExtParams mParams;
        mParams.blockCount = static_cast<uint16_t>(nTotal);
        mParams.blockLen = static_cast<uint32_t>(currentCNum * sizeof(T));
        mParams.srcStride = static_cast<uint32_t>((cLen - currentCNum) * sizeof(T));
        mParams.dstStride = 0;
        DataCopyExtParams vParams;
        vParams.blockCount = 1;
        vParams.blockLen = static_cast<uint32_t>(currentCNum * sizeof(T));
        vParams.srcStride = 0;
        vParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> sumLocal = sumQueue.AllocTensor<T>();
        LocalTensor<T> squareLocal = squareSumQueue.AllocTensor<T>();
        LocalTensor<T> runningMeanLocal = runningMeanQueue.AllocTensor<T>();
        LocalTensor<T> runningVarLocal = runningVarQueue.AllocTensor<T>();
        DataCopyPad(sumLocal, totalSumGm[cBase], mParams, padParams);
        DataCopyPad(squareLocal, totalSquareSumGm[cBase], mParams, padParams);
        DataCopyPad(runningMeanLocal, runningMeanGm[cBase], vParams, padParams);
        DataCopyPad(runningVarLocal, runningVarGm[cBase], vParams, padParams);
        sumQueue.EnQue(sumLocal);
        squareSumQueue.EnQue(squareLocal);
        runningMeanQueue.EnQue(runningMeanLocal);
        runningVarQueue.EnQue(runningVarLocal);

        float countAll = 0.0f;
        for (uint64_t i = 0; i < nTotal; ++i) {
            countAll += static_cast<float>(sampleCountGm.GetValue(i));
        }

        LocalTensor<float> sumAcc = sumAccBuf.Get<float>();
        LocalTensor<float> squareAcc = squareAccBuf.Get<float>();
        LocalTensor<T> sumIn = sumQueue.DeQue<T>();
        LocalTensor<T> squareIn = squareSumQueue.DeQue<T>();
        ReduceMatrixInto(sumIn, squareIn, sumAcc, squareAcc, nTotal, currentCNum, cAlignT);
        sumQueue.FreeTensor(sumIn);
        squareSumQueue.FreeTensor(squareIn);

        LocalTensor<T> runningMeanIn = runningMeanQueue.DeQue<T>();
        LocalTensor<T> runningVarIn = runningVarQueue.DeQue<T>();
        CastRunningToFp32(runningMeanIn, runningVarIn, currentCNum);

        ComputeAndCopyOut(cBase, currentCNum, countAll);
    }

    __aicore__ inline void ReduceMatrixInto(LocalTensor<T>& sumIn, LocalTensor<T>& squareIn, LocalTensor<float>& sumAcc,
                                            LocalTensor<float>& squareAcc, uint64_t currentN, uint64_t currentCNum,
                                            uint64_t cAlignT)
    {
        const uint64_t reduceCount = currentN * cAlignT;
        uint32_t shape[] = {static_cast<uint32_t>(currentN), static_cast<uint32_t>(cAlignT)};
        if constexpr (std::is_same<T, float>::value) {
            ReduceSum<float, Pattern::Reduce::RA, true>(sumAcc, sumIn, shape, true);
            ReduceSum<float, Pattern::Reduce::RA, true>(squareAcc, squareIn, shape, true);
        } else {
            LocalTensor<float> sumFp32 = sumCastBuf.Get<float>();
            LocalTensor<float> squareFp32 = squareSumCastBuf.Get<float>();
            Cast(sumFp32, sumIn, RoundMode::CAST_NONE, static_cast<uint32_t>(reduceCount));
            Cast(squareFp32, squareIn, RoundMode::CAST_NONE, static_cast<uint32_t>(reduceCount));
            ReduceSum<float, Pattern::Reduce::RA, true>(sumAcc, sumFp32, shape, true);
            ReduceSum<float, Pattern::Reduce::RA, true>(squareAcc, squareFp32, shape, true);
        }
    }

    __aicore__ inline void InitBuffers(uint64_t nTile, uint64_t cTile)
    {
        countFactor = nTile;
        if constexpr (std::is_same<T, float>::value) {
            if (nTile == 1 || nTile == FP32_C_FIRST_GROUP_N) {
                countFactor = Min(nTotal, static_cast<uint64_t>(SAMPLE_COUNT_TILE_N));
            }
        }
        const uint64_t nAlignCount = AlignUp<T>(countFactor, BLOCK_BYTES / sizeof(CountT));
        const uint64_t cAlignT = AlignUp<T>(cTile, BLOCK_BYTES / sizeof(T));
        const uint64_t matrixTCount = nTile * cAlignT;
        const uint64_t matrixFp32Count = nTile * cAlignT;

        pipe->InitBuffer(sumQueue, 1, matrixTCount * sizeof(T));
        pipe->InitBuffer(squareSumQueue, 1, matrixTCount * sizeof(T));
        pipe->InitBuffer(sampleCountQueue, 1, nAlignCount * sizeof(CountT));
        pipe->InitBuffer(runningMeanQueue, 1, cAlignT * sizeof(T));
        pipe->InitBuffer(runningVarQueue, 1, cAlignT * sizeof(T));
        pipe->InitBuffer(outQueue, 1, OUT_QUEUE_DEPTH * cAlignT * sizeof(T));

        pipe->InitBuffer(sumAccBuf, cAlignT * sizeof(float));
        pipe->InitBuffer(squareAccBuf, cAlignT * sizeof(float));
        pipe->InitBuffer(runningMeanFp32Buf, cAlignT * sizeof(float));
        pipe->InitBuffer(runningVarFp32Buf, cAlignT * sizeof(float));
        pipe->InitBuffer(tmp1Buf, cAlignT * sizeof(float));
        pipe->InitBuffer(tmp2Buf, cAlignT * sizeof(float));

        if constexpr (!std::is_same<T, float>::value) {
            pipe->InitBuffer(sumCastBuf, matrixFp32Count * sizeof(float));
            pipe->InitBuffer(squareSumCastBuf, matrixFp32Count * sizeof(float));
        }
    }

    __aicore__ inline float ReduceSampleCount()
    {
        float countAll = 0.0f;
        if (nTotal <= SCALAR_COUNT_N_MAX) {
            for (uint64_t i = 0; i < nTotal; ++i) {
                countAll += static_cast<float>(sampleCountGm.GetValue(i));
            }
            return countAll;
        }
        for (uint64_t nOffset = 0; nOffset < nTotal; nOffset += countFactor) {
            const uint64_t currentN = Min(countFactor, nTotal - nOffset);
            countAll += ReduceSampleCountChunk(nOffset, currentN);
        }
        return countAll;
    }

    __aicore__ inline float ReduceSampleCountChunk(uint64_t nOffset, uint64_t currentN)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentN * sizeof(CountT));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<CountT> padParams{false, 0, 0, 0};

        LocalTensor<CountT> countLocal = sampleCountQueue.AllocTensor<CountT>();
        DataCopyPad(countLocal, sampleCountGm[nOffset], copyParams, padParams);
        sampleCountQueue.EnQue(countLocal);
        LocalTensor<CountT> countIn = sampleCountQueue.DeQue<CountT>();

        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        float countSum = 0.0f;
        for (uint64_t i = 0; i < currentN; ++i) {
            countSum += static_cast<float>(countIn.GetValue(i));
        }
        sampleCountQueue.FreeTensor(countIn);
        return countSum;
    }

    __aicore__ inline void ProcessOneCTile(uint64_t cOffset, uint64_t currentCNum, float countAll)
    {
        const uint64_t cAlignT = AlignUp<T>(currentCNum, BLOCK_BYTES / sizeof(T));
        LocalTensor<float> sumAcc = sumAccBuf.Get<float>();
        LocalTensor<float> squareAcc = squareAccBuf.Get<float>();
        Duplicate(sumAcc, 0.0f, static_cast<uint32_t>(cAlignT));
        Duplicate(squareAcc, 0.0f, static_cast<uint32_t>(cAlignT));

        for (uint64_t nOffset = 0; nOffset < nTotal; nOffset += nFactor) {
            const uint64_t currentN = Min(nFactor, nTotal - nOffset);
            ReduceSumSquareChunk(cOffset, nOffset, currentN, currentCNum, sumAcc, squareAcc);
        }

        CopyInRunning(cOffset, currentCNum);
        ComputeAndCopyOut(cOffset, currentCNum, countAll);
    }

    __aicore__ inline void ReduceSumSquareChunk(uint64_t cOffset, uint64_t nOffset, uint64_t currentN,
                                                uint64_t currentCNum, LocalTensor<float>& sumAcc,
                                                LocalTensor<float>& squareAcc)
    {
        const uint64_t cAlignT = AlignUp<T>(currentCNum, BLOCK_BYTES / sizeof(T));
        const uint64_t reduceCount = currentN * cAlignT;

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(currentN);
        copyParams.blockLen = static_cast<uint32_t>(currentCNum * sizeof(T));
        copyParams.srcStride = static_cast<uint32_t>((cLen - currentCNum) * sizeof(T));
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        const uint64_t gmOffset = nOffset * cLen + cOffset;
        LocalTensor<T> sumLocal = sumQueue.AllocTensor<T>();
        LocalTensor<T> squareLocal = squareSumQueue.AllocTensor<T>();
        DataCopyPad(sumLocal, totalSumGm[gmOffset], copyParams, padParams);
        DataCopyPad(squareLocal, totalSquareSumGm[gmOffset], copyParams, padParams);
        sumQueue.EnQue(sumLocal);
        squareSumQueue.EnQue(squareLocal);
        LocalTensor<T> sumIn = sumQueue.DeQue<T>();
        LocalTensor<T> squareIn = squareSumQueue.DeQue<T>();
        if constexpr (std::is_same<T, float>::value) {
            if (nFactor == FP32_C_FIRST_GROUP_N) {
                for (uint64_t row = 0; row < currentN; ++row) {
                    const uint64_t rowOffset = row * cAlignT;
                    Add(sumAcc, sumAcc, sumIn[rowOffset], static_cast<uint32_t>(currentCNum));
                    Add(squareAcc, squareAcc, squareIn[rowOffset], static_cast<uint32_t>(currentCNum));
                }
                sumQueue.FreeTensor(sumIn);
                squareSumQueue.FreeTensor(squareIn);
                return;
            }
            if (currentN == 1) {
                Add(sumAcc, sumAcc, sumIn, static_cast<uint32_t>(currentCNum));
                Add(squareAcc, squareAcc, squareIn, static_cast<uint32_t>(currentCNum));
                sumQueue.FreeTensor(sumIn);
                squareSumQueue.FreeTensor(squareIn);
                return;
            }
        }

        LocalTensor<float> sumPart = tmp1Buf.Get<float>();
        LocalTensor<float> squarePart = tmp2Buf.Get<float>();
        uint32_t shape[] = {static_cast<uint32_t>(currentN), static_cast<uint32_t>(cAlignT)};
        if constexpr (std::is_same<T, float>::value) {
            ReduceSum<float, Pattern::Reduce::RA, true>(sumPart, sumIn, shape, true);
            ReduceSum<float, Pattern::Reduce::RA, true>(squarePart, squareIn, shape, true);
        } else {
            LocalTensor<float> sumFp32 = sumCastBuf.Get<float>();
            LocalTensor<float> squareFp32 = squareSumCastBuf.Get<float>();
            Cast(sumFp32, sumIn, RoundMode::CAST_NONE, static_cast<uint32_t>(reduceCount));
            Cast(squareFp32, squareIn, RoundMode::CAST_NONE, static_cast<uint32_t>(reduceCount));
            ReduceSum<float, Pattern::Reduce::RA, true>(sumPart, sumFp32, shape, true);
            ReduceSum<float, Pattern::Reduce::RA, true>(squarePart, squareFp32, shape, true);
        }
        Add(sumAcc, sumAcc, sumPart, static_cast<uint32_t>(cAlignT));
        Add(squareAcc, squareAcc, squarePart, static_cast<uint32_t>(cAlignT));

        sumQueue.FreeTensor(sumIn);
        squareSumQueue.FreeTensor(squareIn);
    }

    __aicore__ inline void CopyInRunning(uint64_t cOffset, uint64_t currentCNum)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentCNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> runningMeanLocal = runningMeanQueue.AllocTensor<T>();
        LocalTensor<T> runningVarLocal = runningVarQueue.AllocTensor<T>();
        DataCopyPad(runningMeanLocal, runningMeanGm[cOffset], copyParams, padParams);
        DataCopyPad(runningVarLocal, runningVarGm[cOffset], copyParams, padParams);
        runningMeanQueue.EnQue(runningMeanLocal);
        runningVarQueue.EnQue(runningVarLocal);
        LocalTensor<T> runningMeanIn = runningMeanQueue.DeQue<T>();
        LocalTensor<T> runningVarIn = runningVarQueue.DeQue<T>();
        CastRunningToFp32(runningMeanIn, runningVarIn, currentCNum);
    }

    // running mean/var 拷入 fp32 buffer 后释放输入队列，single-tile 与通用 C tile 路径共用。
    __aicore__ inline void CastRunningToFp32(LocalTensor<T>& runningMeanIn, LocalTensor<T>& runningVarIn,
                                             uint64_t currentCNum)
    {
        LocalTensor<float> runningMeanFp32 = runningMeanFp32Buf.Get<float>();
        LocalTensor<float> runningVarFp32 = runningVarFp32Buf.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(runningMeanFp32, runningMeanIn, 0.0f, static_cast<uint32_t>(currentCNum));
            Adds(runningVarFp32, runningVarIn, 0.0f, static_cast<uint32_t>(currentCNum));
        } else {
            Cast(runningMeanFp32, runningMeanIn, RoundMode::CAST_NONE, static_cast<uint32_t>(currentCNum));
            Cast(runningVarFp32, runningVarIn, RoundMode::CAST_NONE, static_cast<uint32_t>(currentCNum));
        }
        runningMeanQueue.FreeTensor(runningMeanIn);
        runningVarQueue.FreeTensor(runningVarIn);
    }

    __aicore__ inline void ComputeAndCopyOut(uint64_t cOffset, uint64_t currentCNum, float countAll)
    {
        LocalTensor<float> batchMean = sumAccBuf.Get<float>();
        LocalTensor<float> batchVar = squareAccBuf.Get<float>();
        LocalTensor<float> batchInvstd = tmp1Buf.Get<float>();
        LocalTensor<float> tmp = tmp2Buf.Get<float>();
        LocalTensor<float> runningMeanOut = runningMeanFp32Buf.Get<float>();
        LocalTensor<float> runningVarOut = runningVarFp32Buf.Get<float>();

        const float reciprocalCount = (countAll == 0.0f) ? 0.0f : (1.0f / countAll);
        const float oneSubMomentum = 1.0f - momentum;
        // countAll<=1 退化为有偏估计，防 (countAll-1)=0 产生 NaN 写入 running_var
        const float unbiasedMomentum = (countAll > 1.0f) ? (countAll / (countAll - 1.0f) * momentum) : momentum;

        Muls(batchMean, batchMean, reciprocalCount, static_cast<uint32_t>(currentCNum));
        Muls(batchVar, batchVar, reciprocalCount, static_cast<uint32_t>(currentCNum));
        Mul(tmp, batchMean, batchMean, static_cast<uint32_t>(currentCNum));
        Sub(batchVar, batchVar, tmp, static_cast<uint32_t>(currentCNum));

        Adds(batchInvstd, batchVar, eps, static_cast<uint32_t>(currentCNum));
        Sqrt(batchInvstd, batchInvstd, static_cast<uint32_t>(currentCNum));
        Duplicate(tmp, 1.0f, static_cast<uint32_t>(currentCNum));
        Div(batchInvstd, tmp, batchInvstd, static_cast<uint32_t>(currentCNum));

        Muls(runningMeanOut, runningMeanOut, oneSubMomentum, static_cast<uint32_t>(currentCNum));
        Muls(tmp, batchMean, momentum, static_cast<uint32_t>(currentCNum));
        Add(runningMeanOut, runningMeanOut, tmp, static_cast<uint32_t>(currentCNum));

        Muls(runningVarOut, runningVarOut, oneSubMomentum, static_cast<uint32_t>(currentCNum));
        Muls(tmp, batchVar, unbiasedMomentum, static_cast<uint32_t>(currentCNum));
        Add(runningVarOut, runningVarOut, tmp, static_cast<uint32_t>(currentCNum));

        CopyOutResults(cOffset, currentCNum, batchMean, batchInvstd, runningMeanOut, runningVarOut);
    }

    // 输出结果：fp32 直写，fp16 打包 cast 后写出
    __aicore__ inline void CopyOutResults(uint64_t cOffset, uint64_t currentCNum, LocalTensor<float>& batchMean,
                                          LocalTensor<float>& batchInvstd, LocalTensor<float>& runningMeanOut,
                                          LocalTensor<float>& runningVarOut)
    {
        if constexpr (std::is_same<T, float>::value) {
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint32_t>(currentCNum * sizeof(T));
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            const int32_t vToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopyPad(batchMeanGm[cOffset], batchMean, outParams);
            DataCopyPad(batchInvstdGm[cOffset], batchInvstd, outParams);
            DataCopyPad(runningMeanUpdateGm[cOffset], runningMeanOut, outParams);
            DataCopyPad(runningVarUpdateGm[cOffset], runningVarOut, outParams);
            const int32_t mte3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(mte3ToV);
            WaitFlag<HardEvent::MTE3_V>(mte3ToV);
        } else {
            const uint64_t cAlignT = AlignUp<T>(currentCNum, BLOCK_BYTES / sizeof(T));
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint32_t>(currentCNum * sizeof(T));
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
            const uint32_t cnt = static_cast<uint32_t>(currentCNum);
            Cast(outLocal, batchMean, RoundMode::CAST_RINT, cnt);
            Cast(outLocal[cAlignT], batchInvstd, RoundMode::CAST_RINT, cnt);
            Cast(outLocal[cAlignT * 2], runningMeanOut, RoundMode::CAST_RINT, cnt);
            Cast(outLocal[cAlignT * 3], runningVarOut, RoundMode::CAST_RINT, cnt);
            outQueue.EnQue(outLocal);
            LocalTensor<T> out = outQueue.DeQue<T>();
            DataCopyPad(batchMeanGm[cOffset], out, outParams);
            DataCopyPad(batchInvstdGm[cOffset], out[cAlignT], outParams);
            DataCopyPad(runningMeanUpdateGm[cOffset], out[cAlignT * 2], outParams);
            DataCopyPad(runningVarUpdateGm[cOffset], out[cAlignT * 3], outParams);
            outQueue.FreeTensor(out);
        }
    }

private:
    TPipe* pipe;
    uint64_t countFactor = 0;

    TQue<QuePosition::VECIN, 1> sumQueue;
    TQue<QuePosition::VECIN, 1> squareSumQueue;
    TQue<QuePosition::VECIN, 1> sampleCountQueue;
    TQue<QuePosition::VECIN, 1> runningMeanQueue;
    TQue<QuePosition::VECIN, 1> runningVarQueue;
    TQue<QuePosition::VECOUT, OUT_QUEUE_DEPTH> outQueue;

    TBuf<TPosition::VECCALC> sumAccBuf;
    TBuf<TPosition::VECCALC> squareAccBuf;
    TBuf<TPosition::VECCALC> runningMeanFp32Buf;
    TBuf<TPosition::VECCALC> runningVarFp32Buf;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    TBuf<TPosition::VECCALC> sumCastBuf;
    TBuf<TPosition::VECCALC> squareSumCastBuf;
};
} // namespace SyncBatchNormGatherStats

using namespace SyncBatchNormGatherStats;

SYNC_BATCH_NORM_GATHER_STATS_KERNEL_ENTRY(SyncBatchNormGatherStatsVector)
