/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_v2.h
 * \brief SmoothL1LossV2 kernel operator
 */
#ifndef SMOOTH_L1_LOSS_V2_H
#define SMOOTH_L1_LOSS_V2_H
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "smooth_l1_loss_v2_tiling_data.h"
#include "smooth_l1_loss_v2_tiling_key.h"

namespace MySmoothL1LossV2 {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t SYNC_WORKSPACE_REGION_NUM = 2;

__aicore__ inline uint64_t AlignUpValue(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

__aicore__ inline uint64_t AlignDownValue(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return (value / align) * align;
}

template <typename SRC>
__aicore__ inline float ToFloatValue(SRC val)
{
    if constexpr (std::is_same_v<SRC, bfloat16_t>) {
        uint16_t bf16Bits = 0;
        __builtin_memcpy(&bf16Bits, &val, sizeof(bf16Bits));
        uint32_t fp32Bits = static_cast<uint32_t>(bf16Bits) << 16;
        float out = 0.0f;
        __builtin_memcpy(&out, &fp32Bits, sizeof(out));
        return out;
    }
    return static_cast<float>(val);
}

template <typename DST>
__aicore__ inline DST FromFloatValue(float val)
{
    if constexpr (std::is_same_v<DST, bfloat16_t>) {
        uint32_t fp32Bits = 0;
        __builtin_memcpy(&fp32Bits, &val, sizeof(fp32Bits));
        const uint32_t exponentMask = 0x7F800000U;
        const uint32_t mantissaMask = 0x007FFFFFU;

        uint16_t bf16Bits;
        // Keep Inf/NaN payload class stable instead of feeding cast with exceptional values.
        if ((fp32Bits & exponentMask) == exponentMask) {
            bf16Bits = static_cast<uint16_t>(fp32Bits >> 16);
            if ((fp32Bits & mantissaMask) != 0) {
                bf16Bits |= 0x0040U; // force qNaN payload bit
            }
        } else {
            uint32_t lsb = (fp32Bits >> 16) & 0x1U;
            uint32_t roundingBias = 0x00007FFFU + lsb;
            bf16Bits = static_cast<uint16_t>((fp32Bits + roundingBias) >> 16);
        }

        bfloat16_t out;
        __builtin_memcpy(&out, &bf16Bits, sizeof(bf16Bits));
        return out;
    }
    return static_cast<DST>(val);
}

// 支持三种数据类型，三输入（predict, label, loss）
template <typename TYPE_PREDICT, typename TYPE_LABEL, typename TYPE_LOSS>
class KernelSmoothL1LossV2 {
public:
    __aicore__ inline KernelSmoothL1LossV2(){};

    __aicore__ inline void Init(
        GM_ADDR predict, GM_ADDR label, GM_ADDR loss, GM_ADDR workspace, const SmoothL1LossV2TilingData& tiling)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        (void)tiling.finalBigTileNum;
        (void)tiling.finalSmallTileNum;
        (void)tiling.smallCoreDataNum;
        (void)tiling.smallTailDataNum;
        (void)tiling.bigTailDataNum;
        (void)tiling.tailBlockNum;
        this->blockBytes = tiling.blockBytes;
        uint64_t coreId = AscendC::GetBlockIdx();
        uint64_t globalBufferIndex = tiling.bigCoreDataNum * coreId;

        InitTileAndAlign(tiling.tileDataNum, tiling.reduction);
        InitReductionAndWorkspace(workspace, tiling.totalDataNum, tiling.sigma, tiling.reduction);
        InitCorePartition(coreId, globalBufferIndex);
        InitGlobalBuffers(predict, label, loss, globalBufferIndex, tiling.reduction);
        InitPipeBuffers();
    }

    __aicore__ inline void Process()
    {
        if (this->coreDataNum == 0) {
            if (this->reduction != 0) {
                GlobalReduction(0.0f, 0.0f);
            }
            return;
        }

        int32_t loopCount = static_cast<int32_t>(this->tileNum);
        float totalSum = 0.0f;
        float totalCompensation = 0.0f;

        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; ++i) {
            CopyIn(i);
            if (this->reduction == 0) {
                Compute(i);
                CopyOut(i);
            } else {
                ComputeReduceAccumulate(i, totalSum, totalCompensation);
            }
        }

        this->processDataNum = this->tailDataNum;
        if (this->processDataNum > 0) {
            CopyIn(loopCount - 1);
            if (this->reduction == 0) {
                Compute(loopCount - 1);
                CopyOut(loopCount - 1);
            } else {
                ComputeReduceAccumulate(loopCount - 1, totalSum, totalCompensation);
            }
        }

        if (this->reduction != 0) {
            GlobalReduction(totalSum, totalCompensation);
        }
    }

private:
    __aicore__ inline void InitTileAndAlign(uint64_t inputTileDataNum, int reduction)
    {
        constexpr uint64_t kAlignBytes = 32;
        const uint64_t alignPredict = kAlignBytes / sizeof(TYPE_PREDICT);
        const uint64_t alignLabel = kAlignBytes / sizeof(TYPE_LABEL);
        const uint64_t alignLoss = kAlignBytes / sizeof(TYPE_LOSS);
        const uint64_t alignFloat = kAlignBytes / sizeof(float);
        uint64_t maxPair = (alignPredict > alignLabel) ? alignPredict : alignLabel;
        uint64_t maxPair2 = (alignLoss > alignFloat) ? alignLoss : alignFloat;
        this->alignUnit = (maxPair > maxPair2) ? maxPair : maxPair2;

        constexpr uint64_t ubPoolBytes = 196608;
        constexpr uint64_t ubSafetyBytes = 512;
        const uint64_t usableUbBytes = (ubPoolBytes > ubSafetyBytes) ? (ubPoolBytes - ubSafetyBytes) : ubPoolBytes;
        uint64_t bytesPerElem = static_cast<uint64_t>(BUFFER_NUM) * (sizeof(TYPE_PREDICT) + sizeof(TYPE_LABEL)) +
                                sizeof(float) + sizeof(float);
        if (reduction == 0) {
            bytesPerElem += static_cast<uint64_t>(BUFFER_NUM) * sizeof(TYPE_LOSS);
        }

        uint64_t maxTileDataNum = (bytesPerElem == 0) ? 0 : (usableUbBytes / bytesPerElem);
        if (maxTileDataNum == 0) {
            maxTileDataNum = 1;
        }

        this->tileDataNum = inputTileDataNum;
        if (this->tileDataNum == 0 || this->tileDataNum > maxTileDataNum) {
            this->tileDataNum = maxTileDataNum;
        }

        // Keep all vector buffers 32B-friendly to avoid runtime allocator corruption.
        if (this->alignUnit > 1) {
            uint64_t alignedTile = AlignDownValue(this->tileDataNum, this->alignUnit);
            if (alignedTile == 0) {
                alignedTile = this->alignUnit;
            }
            if (alignedTile > maxTileDataNum) {
                alignedTile = AlignDownValue(maxTileDataNum, this->alignUnit);
                if (alignedTile == 0) {
                    alignedTile = maxTileDataNum;
                }
            }
            this->tileDataNum = alignedTile;
        }
    }

    __aicore__ inline void InitReductionAndWorkspace(
        GM_ADDR workspace, uint64_t totalDataNum, float sigma, int reduction)
    {
        this->sigma = sigma;
        this->reduction = reduction;
        this->totalDataNum = totalDataNum;
        this->coreNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        this->alignedCoreNum = ((this->coreNum + 7ULL) / 8ULL) * 8ULL;
        this->totalReduceCount = static_cast<uint32_t>(this->coreNum * 8ULL);
        this->workspaceFloatOffset = (this->coreNum * this->blockBytes * SYNC_WORKSPACE_REGION_NUM) / sizeof(float);
        if (this->reduction != 0) {
            workspaceGm.SetGlobalBuffer((__gm__ float*)workspace + this->workspaceFloatOffset, this->totalReduceCount);
        }
    }

    __aicore__ inline void InitCorePartition(uint64_t coreId, uint64_t& globalBufferIndex)
    {
        // 按元素总数严格均分，避免不同 dtype/对齐组合下出现跨核覆盖空洞。
        uint64_t baseCoreDataNum = (this->coreNum == 0) ? 0 : (this->totalDataNum / this->coreNum);
        uint64_t extraCoreNum = (this->coreNum == 0) ? 0 : (this->totalDataNum % this->coreNum);
        this->coreDataNum = baseCoreDataNum + ((coreId < extraCoreNum) ? 1ULL : 0ULL);
        globalBufferIndex = coreId * baseCoreDataNum + ((coreId < extraCoreNum) ? coreId : extraCoreNum);

        if (this->coreDataNum > 0 && this->tileDataNum > this->coreDataNum) {
            this->tileDataNum = this->coreDataNum;
        }
        if (this->tileDataNum == 0 || this->coreDataNum == 0) {
            this->tileNum = 0;
            this->tailDataNum = 0;
        } else {
            this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
            this->tailDataNum = this->coreDataNum - this->tileDataNum * (this->tileNum - 1);
        }
    }

    __aicore__ inline void InitGlobalBuffers(
        GM_ADDR predict, GM_ADDR label, GM_ADDR loss, uint64_t globalBufferIndex, int reduction)
    {
        predictGm.SetGlobalBuffer((__gm__ TYPE_PREDICT*)predict + globalBufferIndex, this->coreDataNum);
        labelGm.SetGlobalBuffer((__gm__ TYPE_LABEL*)label + globalBufferIndex, this->coreDataNum);
        if (reduction == 0) {
            lossGm.SetGlobalBuffer((__gm__ TYPE_LOSS*)loss + globalBufferIndex, this->coreDataNum);
        } else {
            // Reduced output is a scalar; keeping length at 1 prevents reduction-path overwrite.
            lossGm.SetGlobalBuffer((__gm__ TYPE_LOSS*)loss, 1);
        }
    }

    __aicore__ inline void InitPipeBuffers()
    {
        pipe.InitBuffer(inQueuePredict, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_PREDICT));
        pipe.InitBuffer(inQueueLabel, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_LABEL));
        if (this->reduction == 0) {
            pipe.InitBuffer(outQueueLoss, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_LOSS));
        } else {
            pipe.InitBuffer(reduceAccumBuf, sizeof(float) * 8U);
            pipe.InitBuffer(crossCoreReduceBuf, sizeof(float) * this->totalReduceCount);
        }
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
    }

    template <typename T>
    __aicore__ inline void CopyGmToLocalAligned(
        AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyLocalToGmAligned(
        AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    template <typename T>
    __aicore__ inline void CopyLocalToLocalAligned(
        AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        if (this->alignUnit != 0 && count < this->alignUnit) {
            for (uint64_t i = 0; i < count; ++i) {
                dst.SetValue(i, src.GetValue(i));
            }
            return;
        }
        uint64_t alignedDown = (this->alignUnit == 0) ? count : (count / this->alignUnit) * this->alignUnit;
        if (alignedDown > 0) {
            AscendC::DataCopy(dst, src, alignedDown);
        }
        if (count > alignedDown) {
            for (uint64_t i = alignedDown; i < count; ++i) {
                dst.SetValue(i, src.GetValue(i));
            }
        }
    }

    template <typename DST, typename SRC>
    __aicore__ inline void ConvertLocalByScalar(
        AscendC::LocalTensor<DST> dst, AscendC::LocalTensor<SRC> src, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        if constexpr (std::is_same_v<DST, SRC>) {
            CopyLocalToLocalAligned(dst, src, count);
        } else if constexpr (std::is_same_v<DST, float>) {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<SRC, float>) {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                dst.SetValue(i, static_cast<DST>(src.GetValue(i)));
            }
        }
    }

    __aicore__ inline void ComputeSmoothL1FloatCore(
        AscendC::LocalTensor<TYPE_PREDICT> predictLocal, AscendC::LocalTensor<TYPE_LABEL> labelLocal,
        AscendC::LocalTensor<float> diff, AscendC::LocalTensor<float> absDiff)
    {
        float sigmaVal = this->sigma;
        float invSigma = (sigmaVal == 0.0f) ? 0.0f : 1.0f / sigmaVal;

        if constexpr (std::is_same_v<TYPE_PREDICT, float>) {
            CopyLocalToLocalAligned(diff, predictLocal, this->processDataNum);
        } else {
            ConvertLocalByScalar(diff, predictLocal, this->processDataNum);
        }
        if constexpr (std::is_same_v<TYPE_LABEL, float>) {
            CopyLocalToLocalAligned(absDiff, labelLocal, this->processDataNum);
        } else {
            ConvertLocalByScalar(absDiff, labelLocal, this->processDataNum);
        }

        AscendC::Sub(diff, diff, absDiff, this->processDataNum);
        AscendC::Abs(absDiff, diff, this->processDataNum);
        AscendC::Mins(diff, absDiff, sigmaVal, this->processDataNum);
        AscendC::Sub(absDiff, absDiff, diff, this->processDataNum);
        AscendC::Mul(diff, diff, diff, this->processDataNum);
        AscendC::Muls(diff, diff, 0.5f * invSigma, this->processDataNum);
        AscendC::Add(diff, diff, absDiff, this->processDataNum);
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_PREDICT> predictLocal = inQueuePredict.AllocTensor<TYPE_PREDICT>();
        AscendC::LocalTensor<TYPE_LABEL> labelLocal = inQueueLabel.AllocTensor<TYPE_LABEL>();
        CopyGmToLocalAligned(predictLocal, predictGm[progress * this->tileDataNum], this->processDataNum);
        CopyGmToLocalAligned(labelLocal, labelGm[progress * this->tileDataNum], this->processDataNum);
        inQueuePredict.EnQue(predictLocal);
        inQueueLabel.EnQue(labelLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_LOSS> lossLocal = outQueueLoss.DeQue<TYPE_LOSS>();
        CopyLocalToGmAligned(lossGm[progress * this->tileDataNum], lossLocal, this->processDataNum);
        outQueueLoss.FreeTensor(lossLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_PREDICT> predictLocal = inQueuePredict.DeQue<TYPE_PREDICT>();
        AscendC::LocalTensor<TYPE_LABEL> labelLocal = inQueueLabel.DeQue<TYPE_LABEL>();
        AscendC::LocalTensor<TYPE_LOSS> lossLocal = outQueueLoss.AllocTensor<TYPE_LOSS>();
        AscendC::LocalTensor<float> diff = tmp1.Get<float>();
        AscendC::LocalTensor<float> absDiff = tmp2.Get<float>();

        ComputeSmoothL1FloatCore(predictLocal, labelLocal, diff, absDiff);

        if constexpr (std::is_same_v<TYPE_LOSS, float>) {
            CopyLocalToLocalAligned(lossLocal, diff, this->processDataNum);
        } else {
            ConvertLocalByScalar(lossLocal, diff, this->processDataNum);
        }
        outQueueLoss.EnQue<TYPE_LOSS>(lossLocal);
        inQueuePredict.FreeTensor(predictLocal);
        inQueueLabel.FreeTensor(labelLocal);
    }

    __aicore__ inline float ReduceSumTile(AscendC::LocalTensor<float>& val, uint64_t processLen)
    {
        if (processLen == 0) {
            return 0.0f;
        }
        uint32_t alignLen = static_cast<uint32_t>(AlignUpValue(processLen, 8));
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t i = static_cast<uint32_t>(processLen); i < alignLen; ++i) {
            val.SetValue(i, 0.0f);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        uint64_t currentLen = alignLen;
        while (currentLen > 8) {
            uint64_t half = ((currentLen + 15) >> 4) << 3;
            uint64_t rightLen = currentLen - half;
            AscendC::Add<float>(val, val, val[half], rightLen);
            currentLen = half;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        float tileSum = 0.0f;
        for (uint64_t i = 0; i < 8; ++i) {
            tileSum += val.GetValue(i);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return tileSum;
    }

    __aicore__ inline void ComputeReduceAccumulate(int32_t progress, float& sum, float& compensation)
    {
        (void)progress;
        AscendC::LocalTensor<TYPE_PREDICT> predictLocal = inQueuePredict.DeQue<TYPE_PREDICT>();
        AscendC::LocalTensor<TYPE_LABEL> labelLocal = inQueueLabel.DeQue<TYPE_LABEL>();
        AscendC::LocalTensor<float> diff = tmp1.Get<float>();
        AscendC::LocalTensor<float> absDiff = tmp2.Get<float>();

        ComputeSmoothL1FloatCore(predictLocal, labelLocal, diff, absDiff);

        float tileSum = ReduceSumTile(diff, this->processDataNum);
        // Compensated accumulation reduces reduction drift on large reductions.
        float y = tileSum - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;

        inQueuePredict.FreeTensor(predictLocal);
        inQueueLabel.FreeTensor(labelLocal);
    }

    __aicore__ inline void StoreReductionResult(float totalSum)
    {
        TYPE_LOSS scalarVal;
        if constexpr (std::is_same_v<TYPE_LOSS, float>) {
            scalarVal = totalSum;
        } else {
            scalarVal = FromFloatValue<TYPE_LOSS>(totalSum);
        }
        lossGm.SetValue(0, scalarVal);
    }

    __aicore__ inline void GlobalReduction(float totalSum, float totalCompensation)
    {
        (void)totalCompensation;
        if (this->coreNum == 1) {
            if (this->reduction == 2 && this->totalDataNum != 0) {
                totalSum = totalSum / static_cast<float>(static_cast<int64_t>(this->totalDataNum));
            }
            StoreReductionResult(totalSum);
            return;
        }

        AscendC::LocalTensor<float> accum = reduceAccumBuf.Get<float>();
        for (uint32_t i = 0; i < 8U; ++i) {
            accum.SetValue(i, 0.0f);
        }
        accum.SetValue(0, totalSum);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopy(workspaceGm[AscendC::GetBlockIdx() * 8ULL], accum, 8U);
        AscendC::SyncAll();

        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        AscendC::LocalTensor<float> allCoreSums = crossCoreReduceBuf.Get<float>();
        AscendC::DataCopy(allCoreSums, workspaceGm, this->totalReduceCount);
        AscendC::PipeBarrier<PIPE_ALL>();

        float globalSum = 0.0f;
        float globalCompensation = 0.0f;
        for (uint64_t i = 0; i < this->coreNum; ++i) {
            float y = allCoreSums.GetValue(i * 8ULL) - globalCompensation;
            float t = globalSum + y;
            globalCompensation = (t - globalSum) - y;
            globalSum = t;
        }
        if (this->reduction == 2 && this->totalDataNum != 0) {
            globalSum = globalSum / static_cast<float>(static_cast<int64_t>(this->totalDataNum));
        }
        StoreReductionResult(globalSum);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuePredict;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueLabel;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueLoss;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceAccumBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> crossCoreReduceBuf;
    AscendC::GlobalTensor<TYPE_PREDICT> predictGm;
    AscendC::GlobalTensor<TYPE_LABEL> labelGm;
    AscendC::GlobalTensor<TYPE_LOSS> lossGm;
    AscendC::GlobalTensor<float> workspaceGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    uint64_t totalDataNum = 0;
    uint64_t coreNum = 0;
    uint64_t alignedCoreNum = 0;
    uint32_t totalReduceCount = 0;
    uint64_t workspaceFloatOffset = 0;
    float sigma = 0.0f;
    int reduction = 0;
    uint64_t alignUnit = 0;
    uint32_t blockBytes = 0;
};

} // namespace MySmoothL1LossV2
#endif // SMOOTH_L1_LOSS_V2_H
