/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_NORM_GROUP_NORM_KERNEL_IMPL_H_
#define EXPERIMENTAL_NORM_GROUP_NORM_KERNEL_IMPL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_tiling_data.h"
#include "group_norm_tiling_key.h"

namespace GroupNormKernel {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename T, uint32_t schMode>
class GroupNorm {
public:
    __aicore__ inline GroupNorm() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        GroupNormTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSingleElementGroups();
    __aicore__ inline void ProcessOneGroup(uint32_t groupIndex);
    __aicore__ inline void CalcMeanAndVariance(
        uint64_t groupOffset, uint64_t dataCount, float& meanVal, float& meanOutVal, float& rstdVal);
    __aicore__ inline void NormalizeAndCopyOut(
        uint64_t groupOffset, uint32_t channelBase, uint64_t dataCount, float meanVal, float rstdVal);
    __aicore__ inline void NormalizeFloatTile(
        LocalTensor<float>& srcLocal, LocalTensor<float>& dstLocal, uint64_t offset, uint32_t channelBase,
        uint32_t currentCount, float meanVal, float rstdVal);
    __aicore__ inline float LoadScalarAsFloat(GlobalTensor<T>& gm, uint32_t index);
    __aicore__ inline void CopyInPad(LocalTensor<T>& dstLocal, uint64_t gmOffset, uint32_t count);
    __aicore__ inline void CopyOutPad(GlobalTensor<T>& gm, uint64_t gmOffset, LocalTensor<T>& srcLocal, uint32_t count);
    __aicore__ inline void StoreScalar(GlobalTensor<T>& gm, uint32_t index, float value);
    __aicore__ inline uint64_t GetGroupOffset(uint32_t groupIndex) const;
    __aicore__ inline uint32_t GetChannelBase(uint32_t groupIndex) const;

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> meanGm;
    GlobalTensor<T> rstdGm;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> yFloatBuf;

    GroupNormTilingData tilingData;
    uint32_t startGroup{0};
    uint32_t groupCount{0};
};

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
    GroupNormTilingData* tiling)
{
    (void)workspace;
    tilingData = *tiling;

    uint32_t blockIdx = GetBlockIdx();
    groupCount = tilingData.groupPerCore + (blockIdx < tilingData.groupTailCore ? 1 : 0);
    startGroup = blockIdx * tilingData.groupPerCore +
                 (blockIdx < tilingData.groupTailCore ? blockIdx : tilingData.groupTailCore);

    uint64_t totalNum = static_cast<uint64_t>(tilingData.n) * tilingData.c * tilingData.hxw;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);
    gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma), tilingData.c);
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), tilingData.c);
    meanGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mean), tilingData.groupNum);
    rstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rstd), tilingData.groupNum);

    pipe.InitBuffer(inQueue, BUFFER_NUM, tilingData.tileLength * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, tilingData.tileLength * sizeof(T));
    pipe.InitBuffer(xFloatBuf, tilingData.tileLength * sizeof(float));
    pipe.InitBuffer(yFloatBuf, tilingData.tileLength * sizeof(float));
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::Process()
{
    if (tilingData.elementsPerGroup == 1) {
        ProcessSingleElementGroups();
        return;
    }
    if constexpr (schMode == GROUP_NORM_SCH_SMALL_GROUP) {
        for (uint32_t i = 0; i < groupCount; ++i) {
            ProcessOneGroup(startGroup + i);
        }
    } else {
        for (uint32_t i = 0; i < groupCount; ++i) {
            ProcessOneGroup(startGroup + i);
        }
    }
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::ProcessSingleElementGroups()
{
    uint32_t start = startGroup;
    uint32_t end = startGroup + groupCount;
    float rstdVal = 1.0f / sqrt(tilingData.eps);
    float yVal = tilingData.hasBeta == 0 ? 0.0f : LoadScalarAsFloat(betaGm, 0);
    for (uint32_t offset = start; offset < end; offset += tilingData.tileLength) {
        uint32_t currentCount = tilingData.tileLength;
        if (offset + currentCount > end) {
            currentCount = end - offset;
        }
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        CopyInPad(xLocal, offset, currentCount);
        CopyOutPad(meanGm, offset, xLocal, currentCount);
        if constexpr (IsSameType<T, float>::value) {
            Duplicate(yLocal, rstdVal, currentCount);
        } else {
            LocalTensor<float> yFloat = yFloatBuf.Get<float>();
            Duplicate(yFloat, rstdVal, currentCount);
            Cast(yLocal, yFloat, RoundMode::CAST_ROUND, currentCount);
        }
        CopyOutPad(rstdGm, offset, yLocal, currentCount);
        if constexpr (IsSameType<T, float>::value) {
            if (tilingData.c == 1 || tilingData.hasBeta == 0) {
                Duplicate(yLocal, yVal, currentCount);
            } else {
                for (uint32_t i = 0; i < currentCount; ++i) {
                    uint32_t channel = (offset + i) % tilingData.c;
                    yLocal.SetValue(i, LoadScalarAsFloat(betaGm, channel));
                }
            }
        } else {
            LocalTensor<float> yFloat = yFloatBuf.Get<float>();
            if (tilingData.c == 1 || tilingData.hasBeta == 0) {
                Duplicate(yFloat, yVal, currentCount);
            } else {
                for (uint32_t i = 0; i < currentCount; ++i) {
                    uint32_t channel = (offset + i) % tilingData.c;
                    yFloat.SetValue(i, LoadScalarAsFloat(betaGm, channel));
                }
            }
            Cast(yLocal, yFloat, RoundMode::CAST_ROUND, currentCount);
        }
        CopyOutPad(yGm, offset, yLocal, currentCount);
        inQueue.FreeTensor(xLocal);
        outQueue.FreeTensor(yLocal);
    }
}

template <typename T, uint32_t schMode>
__aicore__ inline uint64_t GroupNorm<T, schMode>::GetGroupOffset(uint32_t groupIndex) const
{
    uint32_t nIndex = groupIndex / tilingData.numGroups;
    uint32_t gIndex = groupIndex % tilingData.numGroups;
    return static_cast<uint64_t>(nIndex) * tilingData.c * tilingData.hxw +
           static_cast<uint64_t>(gIndex) * tilingData.channelsPerGroup * tilingData.hxw;
}

template <typename T, uint32_t schMode>
__aicore__ inline uint32_t GroupNorm<T, schMode>::GetChannelBase(uint32_t groupIndex) const
{
    return (groupIndex % tilingData.numGroups) * tilingData.channelsPerGroup;
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::ProcessOneGroup(uint32_t groupIndex)
{
    float meanVal = 0.0f;
    float meanOutVal = 0.0f;
    float rstdVal = 0.0f;
    uint64_t groupOffset = GetGroupOffset(groupIndex);
    CalcMeanAndVariance(groupOffset, tilingData.elementsPerGroup, meanVal, meanOutVal, rstdVal);
    StoreScalar(meanGm, groupIndex, meanOutVal);
    StoreScalar(rstdGm, groupIndex, rstdVal);
    NormalizeAndCopyOut(groupOffset, GetChannelBase(groupIndex), tilingData.elementsPerGroup, meanVal, rstdVal);
}

template <typename T, uint32_t schMode>
__aicore__ inline float GroupNorm<T, schMode>::LoadScalarAsFloat(GlobalTensor<T>& gm, uint32_t index)
{
    LocalTensor<T> scalarLocal = inQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(scalarLocal, gm[index], copyParams, padParams);
    inQueue.EnQue(scalarLocal);
    scalarLocal = inQueue.DeQue<T>();

    float value = 0.0f;
    if constexpr (IsSameType<T, float>::value) {
        value = scalarLocal.GetValue(0);
    } else {
        LocalTensor<float> scalarFloat = xFloatBuf.Get<float>();
        Cast(scalarFloat, scalarLocal, RoundMode::CAST_NONE, 1);
        value = scalarFloat.GetValue(0);
    }
    inQueue.FreeTensor(scalarLocal);
    return value;
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::StoreScalar(GlobalTensor<T>& gm, uint32_t index, float value)
{
    LocalTensor<T> scalarLocal = outQueue.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        scalarLocal.SetValue(0, value);
    } else {
        LocalTensor<float> scalarFloat = yFloatBuf.Get<float>();
        scalarFloat.SetValue(0, value);
        Cast(scalarLocal, scalarFloat, RoundMode::CAST_ROUND, 1);
    }
    outQueue.EnQue(scalarLocal);
    scalarLocal = outQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(gm[index], scalarLocal, copyParams);
    outQueue.FreeTensor(scalarLocal);
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::CopyInPad(LocalTensor<T>& dstLocal, uint64_t gmOffset, uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dstLocal, xGm[gmOffset], copyParams, padParams);
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::CopyOutPad(
    GlobalTensor<T>& gm, uint64_t gmOffset, LocalTensor<T>& srcLocal, uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(gm[gmOffset], srcLocal, copyParams);
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::CalcMeanAndVariance(
    uint64_t groupOffset, uint64_t dataCount, float& meanVal, float& meanOutVal, float& rstdVal)
{
    float sum = 0.0f;
    float sumComp = 0.0f;
    float squareSum = 0.0f;
    float squareSumComp = 0.0f;
    for (uint64_t offset = 0; offset < dataCount; offset += tilingData.tileLength) {
        uint64_t currentCount64 = tilingData.tileLength;
        if (offset + currentCount64 > dataCount) {
            currentCount64 = dataCount - offset;
        }
        uint32_t currentCount = static_cast<uint32_t>(currentCount64);
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        CopyInPad(xLocal, groupOffset + offset, currentCount);
        inQueue.EnQue(xLocal);
        xLocal = inQueue.DeQue<T>();

        if constexpr (IsSameType<T, float>::value) {
            for (uint32_t i = 0; i < currentCount; ++i) {
                float v = xLocal.GetValue(i);
                float sumInput = v - sumComp;
                float sumNext = sum + sumInput;
                sumComp = (sumNext - sum) - sumInput;
                sum = sumNext;
                float squareInput = v * v - squareSumComp;
                float squareNext = squareSum + squareInput;
                squareSumComp = (squareNext - squareSum) - squareInput;
                squareSum = squareNext;
            }
        } else {
            LocalTensor<float> xFloat = xFloatBuf.Get<float>();
            Cast(xFloat, xLocal, RoundMode::CAST_NONE, currentCount);
            for (uint32_t i = 0; i < currentCount; ++i) {
                float v = xFloat.GetValue(i);
                float sumInput = v - sumComp;
                float sumNext = sum + sumInput;
                sumComp = (sumNext - sum) - sumInput;
                sum = sumNext;
                float squareInput = v * v - squareSumComp;
                float squareNext = squareSum + squareInput;
                squareSumComp = (squareNext - squareSum) - squareInput;
                squareSum = squareNext;
            }
        }
        inQueue.FreeTensor(xLocal);
    }
    meanVal = sum * tilingData.invElementsPerGroup;
    meanOutVal = meanVal;
    float variance = squareSum * tilingData.invElementsPerGroup - meanVal * meanVal;
    variance = variance < 0.0f ? 0.0f : variance;
    rstdVal = 1.0f / sqrt(variance + tilingData.eps);
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::NormalizeFloatTile(
    LocalTensor<float>& srcLocal, LocalTensor<float>& dstLocal, uint64_t offset, uint32_t channelBase,
    uint32_t currentCount, float meanVal, float rstdVal)
{
    for (uint32_t i = 0; i < currentCount;) {
        uint32_t channelInGroup = static_cast<uint32_t>((offset + i) / tilingData.hxw);
        uint32_t channel = channelBase + channelInGroup;
        uint32_t nextChannelOffset = (channelInGroup + 1) * tilingData.hxw;
        uint32_t segmentEnd = nextChannelOffset > offset + currentCount ?
                                  currentCount :
                                  nextChannelOffset - static_cast<uint32_t>(offset);
        uint32_t segmentCount = segmentEnd - i;
        float gammaVal = tilingData.hasGamma == 0 ? 1.0f : LoadScalarAsFloat(gammaGm, channel);
        float betaVal = tilingData.hasBeta == 0 ? 0.0f : LoadScalarAsFloat(betaGm, channel);
        float scaleVal = rstdVal * gammaVal;
        Adds(dstLocal[i], srcLocal[i], -meanVal, segmentCount);
        Muls(dstLocal[i], dstLocal[i], scaleVal, segmentCount);
        Adds(dstLocal[i], dstLocal[i], betaVal, segmentCount);
        i = segmentEnd;
    }
}

template <typename T, uint32_t schMode>
__aicore__ inline void GroupNorm<T, schMode>::NormalizeAndCopyOut(
    uint64_t groupOffset, uint32_t channelBase, uint64_t dataCount, float meanVal, float rstdVal)
{
    for (uint64_t offset = 0; offset < dataCount; offset += tilingData.tileLength) {
        uint64_t currentCount64 = tilingData.tileLength;
        if (offset + currentCount64 > dataCount) {
            currentCount64 = dataCount - offset;
        }
        uint32_t currentCount = static_cast<uint32_t>(currentCount64);
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        CopyInPad(xLocal, groupOffset + offset, currentCount);
        inQueue.EnQue(xLocal);
        xLocal = inQueue.DeQue<T>();

        if constexpr (IsSameType<T, float>::value) {
            NormalizeFloatTile(xLocal, yLocal, offset, channelBase, currentCount, meanVal, rstdVal);
        } else {
            LocalTensor<float> xFloat = xFloatBuf.Get<float>();
            LocalTensor<float> yFloat = yFloatBuf.Get<float>();
            Cast(xFloat, xLocal, RoundMode::CAST_NONE, currentCount);
            NormalizeFloatTile(xFloat, yFloat, offset, channelBase, currentCount, meanVal, rstdVal);
            Cast(yLocal, yFloat, RoundMode::CAST_ROUND, currentCount);
        }
        CopyOutPad(yGm, groupOffset + offset, yLocal, currentCount);
        inQueue.FreeTensor(xLocal);
        outQueue.FreeTensor(yLocal);
    }
}

} // namespace GroupNormKernel

#endif // EXPERIMENTAL_NORM_GROUP_NORM_KERNEL_IMPL_H_
