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
 * \file max_pool_with_argmax_v3_big_kernel_mul_core.h
 * \brief
 */

#ifndef MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_MUL_CORE_H_
#define MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_MUL_CORE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

namespace MaxPoolWithArgmaxV3BigKernelMulCore {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t REPEAT_DATA = 256;
constexpr uint16_t FLOAT16_NEG_INF = 64512;       // -inf 0xFC00
constexpr uint16_t FLOAT16_INF = 31744;           // inf 0x7C00
constexpr uint16_t FLOAT16_NAN_END = 32768;       // 0x8000
constexpr int32_t FLOAT32_NEG_INF = -2139095040;  // -inf 0xFF800000
constexpr int32_t FLOAT32_INF = 2139095040;       // inf 0x7F800000
constexpr int32_t FLOAT32_NEG_ZERO = -2147483648; // -0
constexpr uint32_t VALUE_WORKSPACE_SIZE = 64 * 4;
constexpr uint32_t INDEX_WORKSPACE_SIZE = 64 * 8;
constexpr int64_t MASK_RATIO = 8;

template <typename T>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        const LocalTensor<T>& xLocal, LocalTensor<float>& castToFP32, TBuf<>& maxUB, TBuf<>& workLocalUB,
        uint32_t dataCount)
    {
        LocalTensor<T> maxOutLocal = maxUB.Get<T>();
        ReduceMax<T>(maxOutLocal, xLocal, xLocal, dataCount, true);
        // pipev
    }

    __aicore__ inline void GetMask(
        const LocalTensor<T>& xLocal, LocalTensor<float>& castToFP32, LocalTensor<uint8_t>& mask, uint32_t dataCount)
    {
        uint32_t dataCountAlign = (dataCount + REPEAT_DATA - 1) / REPEAT_DATA * REPEAT_DATA;
        if (dataCountAlign > dataCount) {
            Duplicate(xLocal[dataCount], T(0), dataCountAlign - dataCount);
            // pipev
        }
        Compare(mask, xLocal, xLocal, CMPMODE::EQ, dataCountAlign);
        // pipev
        Not(mask, mask, dataCountAlign / MASK_RATIO);
        // pipev
    }
};

template <>
class InnerComputer<bfloat16_t> {
public:
    __aicore__ inline void Compute(
        const LocalTensor<bfloat16_t>& xLocal, LocalTensor<float>& castToFP32, TBuf<>& maxUB, TBuf<>& workLocalUB,
        uint32_t dataCount)
    {
        LocalTensor<float> maxOutLocal = maxUB.Get<float>();
        Cast(castToFP32, xLocal, RoundMode::CAST_NONE, dataCount);
        // pipev
        ReduceMax<float>(maxOutLocal, castToFP32, castToFP32, dataCount, true);
        // pipev
    }

    __aicore__ inline void GetMask(
        const LocalTensor<bfloat16_t>& xLocal, LocalTensor<float>& castToFP32, LocalTensor<uint8_t>& mask,
        uint32_t dataCount)
    {
        uint32_t dataCountAlign = (dataCount + REPEAT_DATA - 1) / REPEAT_DATA * REPEAT_DATA;
        if (dataCountAlign > dataCount) {
            Duplicate(castToFP32[dataCount], float(0), dataCountAlign - dataCount);
            // pipev
        }
        Compare(mask, castToFP32, castToFP32, CMPMODE::EQ, dataCountAlign);
        // pipev
        Not(mask, mask, dataCountAlign / MASK_RATIO);
        // pipev
    }
};

template <typename T1, typename T2, typename TINDEX>
class MaxPoolWithArgmaxV3BigKernelMulCore {
public:
    __aicore__ inline MaxPoolWithArgmaxV3BigKernelMulCore(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe_in,
        const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Prepare(int64_t curIdx, int64_t innerBlockIdx);
    __aicore__ inline void BaseCompute(int64_t curIdx);
    __aicore__ inline int64_t HwCopyInput(
        int64_t offset, int64_t blockCount, int64_t blockLen, int64_t blockLenAlign, int64_t srcStride);
    __aicore__ inline int32_t Compute(int64_t dataCount);
    __aicore__ inline int64_t RestoreIndex(int32_t index, int64_t hLen, int64_t wLen);
    __aicore__ inline void CopyMaxOut(int64_t curIdx);
    __aicore__ inline void CopyIndicesOut(int64_t maxIndex, int64_t curIdx);
    __aicore__ inline void NaNIndicesInit(LocalTensor<float> indicesLocal);
    __aicore__ inline void GetIndexWithLastNan(
        LocalTensor<float> indicesMaxLocal, LocalTensor<uint8_t> maskNanLocal, int64_t dataCount, int32_t& index);
    __aicore__ inline int64_t AllWInKernelProcess();
    __aicore__ inline void UpdateMax(int64_t curMaxIndex, T2& maxValue, int64_t& maxIndice, bool first);
    __aicore__ inline int32_t KernelRealIndex(int32_t index, int64_t blockLen, int64_t blockLenAlign);
    __aicore__ inline void CopyOut(int64_t idx, int32_t index);
    __aicore__ inline void ComputeMulCore(int32_t& index);
    __aicore__ inline void CopyInMulCore(int64_t startIdx);
    __aicore__ inline void SplitW(
        int64_t blockLen, int64_t alignBlockLen, int64_t strStride, int64_t& maxValueIndex, T2& value);
    __aicore__ inline int64_t CeilValue(int64_t inputValue, int64_t upperValue)
    {
        if (upperValue == 0) {
            return inputValue;
        }
        return (inputValue + upperValue - 1) / upperValue * upperValue;
    }

    __aicore__ inline int64_t Min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    __aicore__ inline bool IsNan(T2 value)
    {
        if (std::is_same<T2, half>::value) {
            uint16_t nan = *reinterpret_cast<uint16_t*>(&value);
            if ((nan > FLOAT16_INF && nan < FLOAT16_NAN_END) || nan > FLOAT16_NEG_INF) {
                return true;
            }
        } else {
            int32_t nan = *reinterpret_cast<int32_t*>(&value);
            if ((nan != FLOAT32_NEG_ZERO) && (nan > FLOAT32_INF || nan < FLOAT32_NEG_INF)) {
                return true;
            }
        }
        return false;
    }

    TPipe* pipe;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
    // 最大值ub
    TBuf<> maxUB;
    TBuf<> maxUBOutput;
    // indices初始下标
    TBuf<> indicesInitUB;
    // Compare结果mask
    TBuf<> maskNanUB;
    // nan场景最大值和下标
    TBuf<> nanMaxIndexUB;
    TBuf<> nanMaxIndexUBOutput;
    TBuf<> castBuff;

    GlobalTensor<T1> xGm, maxGm;
    GlobalTensor<TINDEX> indicesGm;
    GlobalTensor<T2> maxValueWorkspaceGm;
    GlobalTensor<TINDEX> maxValueIndexWorkspaceGm;

    const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* tilingData;

    uint32_t cBlockIdx = 0;

    int64_t inHW = 1;
    int64_t outHW = 1;
    int64_t curNc = 0;
    int64_t curOriginH = 0;
    int64_t curOriginW = 0;
    int64_t curOriginIndex = 0;
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    T1 minT1 = 0;
    T2 minT2 = 0;
    int32_t inputXQueOffset = 0;
    int64_t curKernelBlockFactorH = 0;
    int64_t curWSplitSize = 0;

    constexpr static int64_t BYTE_T1 = sizeof(T1);
    constexpr static int64_t BLOCK_DATA = platform::GetUbBlockSize();
    constexpr static int64_t BLOCK_NUM_T1 = BLOCK_DATA / sizeof(T1);
    constexpr static int64_t REPEAT_NUM_T1 = REPEAT_DATA / sizeof(T1);
    constexpr static int64_t REPEAT_NUM_T2 = REPEAT_DATA / sizeof(T2);
    constexpr static int64_t BLOCK_NUM_T2 = BLOCK_DATA / sizeof(T2);
};

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe_in,
    const MaxPoolWithArgmaxV3BigKernelMulCoreTilingData* __restrict tiling)
{
    pipe = pipe_in;
    tilingData = tiling;
    // base info
    cBlockIdx = GetBlockIdx();
    inHW = tilingData->hInDim * tilingData->wInDim;
    outHW = tilingData->hOutDim * tilingData->wOutDim;

    uint32_t MIN_FLOAT32 = 0xFF800000; // -inf
    uint16_t MIN_FLOAT16 = 0xFC00;     // -inf
    uint16_t MIN_BFLOAT16 = 0xFF80;    // -inf
    if (std::is_same<T1, float>::value) {
        minT1 = *reinterpret_cast<T1*>(&MIN_FLOAT32);
        minT2 = *reinterpret_cast<T2*>(&MIN_FLOAT32);
    } else if (std::is_same<T1, half>::value) {
        minT1 = *reinterpret_cast<T1*>(&MIN_FLOAT16);
        minT2 = *reinterpret_cast<T2*>(&MIN_FLOAT16);
    } else if (std::is_same<T1, bfloat16_t>::value) {
        minT1 = *reinterpret_cast<T1*>(&MIN_BFLOAT16);
        minT2 = *reinterpret_cast<T2*>(&MIN_FLOAT32);
    }

    // GM
    xGm.SetGlobalBuffer((__gm__ T1*)x);
    maxGm.SetGlobalBuffer((__gm__ T1*)y);
    indicesGm.SetGlobalBuffer((__gm__ TINDEX*)indices);
    maxValueWorkspaceGm.SetGlobalBuffer((__gm__ T2*)workspace);
    maxValueIndexWorkspaceGm.SetGlobalBuffer((__gm__ TINDEX*)(workspace + VALUE_WORKSPACE_SIZE));

    pipe->InitBuffer(
        inputQue, BUFFER_NUM,
        tilingData->maxCountLength * sizeof(float));              // 原地cast 并复用为 nan index 的列表
    pipe->InitBuffer(maxUB, tilingData->valueBufferLength);       // next do 256  参数化
    pipe->InitBuffer(maxUBOutput, tilingData->valueBufferLength); // next do 256  参数化
    pipe->InitBuffer(indicesInitUB, tilingData->maxCountLength * sizeof(float));
    pipe->InitBuffer(maskNanUB, tilingData->maxCountLength / MASK_RATIO); // 复用为reducemax时候的临时空间 worklocal
    pipe->InitBuffer(nanMaxIndexUB, tilingData->valueBufferLength);
    pipe->InitBuffer(nanMaxIndexUBOutput, tilingData->indexBufferLength);
    pipe->InitBuffer(castBuff, tilingData->indexBufferLength);

    if (std::is_same<T1, bfloat16_t>::value) {
        inputXQueOffset = tilingData->maxCountLength; // inputQue的偏移 默认是0，bf16要做原地cast，输入放到后半部分
    }
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::Process()
{
    if (cBlockIdx >= tilingData->multiCoreNum * tilingData->coreNums) {
        return;
    }
    // init indices
    LocalTensor<float> indicesLocal = indicesInitUB.Get<float>();
    NaNIndicesInit(indicesLocal);

    int64_t idx = cBlockIdx / tilingData->multiCoreNum;
    int64_t innerBlockIdx = cBlockIdx % tilingData->multiCoreNum;
    int64_t startIdx = idx * tilingData->multiCoreNum;

    if (tilingData->splitW == 0) {
        if ((innerBlockIdx + 1) == tilingData->multiCoreNum) {
            curKernelBlockFactorH = tilingData->tailKernelBlockFactorH;
        } else {
            curKernelBlockFactorH = tilingData->kernelBlockFactorH;
        }
    } else {
        if ((innerBlockIdx + 1) % tilingData->splitSlice == 0) {
            curWSplitSize = tilingData->tailWSplitSize;
        } else {
            curWSplitSize = tilingData->wSplitSize;
        }
    }
    int32_t index = 0;
    Prepare(idx, innerBlockIdx);
    BaseCompute(idx);
    SyncAll();
    CopyInMulCore(startIdx);
    ComputeMulCore(index);
    CopyOut(idx, index);
}
template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::CopyOut(int64_t idx, int32_t index)
{
    if (cBlockIdx % tilingData->multiCoreNum != 0) {
        return;
    }
    LocalTensor<TINDEX> indicesResult = nanMaxIndexUBOutput.Get<TINDEX>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = 1 * sizeof(T1);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    if (std::is_same<T1, bfloat16_t>::value) {
        LocalTensor<float> maxOutLocal = maxUBOutput.Get<float>();
        LocalTensor<T1> castBuffLocal = castBuff.Get<T1>();
        Cast(castBuffLocal, maxOutLocal, RoundMode::CAST_RINT, MASK_RATIO);
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        DataCopyPad(maxGm[idx], castBuffLocal[0], extParams);
    } else {
        LocalTensor<T1> maxOutLocal = maxUBOutput.Get<T1>();
        DataCopyPad(maxGm[idx], maxOutLocal[0], extParams);
    }
    extParams.blockLen = 1 * sizeof(TINDEX);
    indicesResult.SetValue(0, indicesResult.GetValue(index));

    event_t eventIdStoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdStoMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdStoMTE3);

    DataCopyPad(indicesGm[idx], indicesResult[0], extParams);
    return;
}
template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::ComputeMulCore(int32_t& index)
{
    if (cBlockIdx % tilingData->multiCoreNum != 0) {
        return;
    }
    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    LocalTensor<T2> maxUBLocal = maxUB.Get<T2>();
    LocalTensor<uint8_t> maskNanLocal = maskNanUB.Get<uint8_t>();
    LocalTensor<float> castBuffLocal = castBuff.Get<float>();
    LocalTensor<T2> maxOutLocal = maxUBOutput.Get<T2>();
    ReduceMax<T2>(maxOutLocal, maxUBLocal, maxUBLocal, tilingData->multiCoreNum, true);

    event_t eventIdVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVtoS);
    WaitFlag<HardEvent::V_S>(eventIdVtoS);

    T2 maxIndex = maxOutLocal.GetValue(1);
    T2 maxValue = maxOutLocal.GetValue(0);
    if (std::is_same<T2, half>::value) {
        int16_t indexInt16 = *reinterpret_cast<int16_t*>(&maxIndex);
        index = static_cast<int32_t>(indexInt16);
    } else {
        index = *reinterpret_cast<int32_t*>(&maxIndex);
    }

    if (IsNan(maxValue)) {
        uint32_t dataCountAlign = CeilValue(tilingData->multiCoreNum, REPEAT_NUM_T2);
        uint32_t alignLen = CeilValue(tilingData->multiCoreNum, BLOCK_NUM_T2);
        if (dataCountAlign > alignLen) {
            Duplicate(maxUBLocal[alignLen], T2(0), dataCountAlign - alignLen);
        }
        Compare(maskNanLocal, maxUBLocal, maxUBLocal, CMPMODE::EQ, dataCountAlign);
        Not(maskNanLocal, maskNanLocal, dataCountAlign / MASK_RATIO);

        LocalTensor<float> indicesLocal = indicesInitUB.Get<float>();
        LocalTensor<float> nanMaxIndex = nanMaxIndexUB.Get<float>();
        Select(
            castBuffLocal, maskNanLocal, indicesLocal, float(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
            tilingData->multiCoreNum);
        ReduceMax<float>(nanMaxIndex, castBuffLocal, castBuffLocal, tilingData->multiCoreNum, false);

        event_t eventIdVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVtoS);
        WaitFlag<HardEvent::V_S>(eventIdVtoS);
        index = ScalarCast<float, int32_t, RoundMode::CAST_ROUND>(nanMaxIndex.GetValue(0));
    }
    return;
}
template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::CopyInMulCore(int64_t startIdx)
{
    if (cBlockIdx % tilingData->multiCoreNum != 0) {
        return;
    }
    int32_t alignValue = CeilValue(tilingData->multiCoreNum, BLOCK_NUM_T2);
    LocalTensor<T2> maxValue = maxUB.Get<T2>();
    DataCopyPadExtParams<T2> padExtParams;
    padExtParams.isPad = true;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = alignValue - tilingData->multiCoreNum;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = tilingData->multiCoreNum * sizeof(T2);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    extParams.rsv = 0;
    DataCopyPad(maxValue, maxValueWorkspaceGm[startIdx], extParams, padExtParams);

    LocalTensor<TINDEX> maxValueIndices = nanMaxIndexUBOutput.Get<TINDEX>();
    DataCopyExtParams indicesExtParams;
    indicesExtParams.blockCount = 1;
    indicesExtParams.blockLen = tilingData->multiCoreNum * sizeof(TINDEX);
    indicesExtParams.srcStride = 0;
    indicesExtParams.dstStride = 0;
    indicesExtParams.rsv = 0;

    DataCopyPadExtParams<TINDEX> indicesPadExtParams;
    indicesPadExtParams.isPad = false;
    indicesPadExtParams.leftPadding = 0;
    indicesPadExtParams.rightPadding = 0;
    indicesPadExtParams.paddingValue = 0;
    DataCopyPad(maxValueIndices, maxValueIndexWorkspaceGm[startIdx], indicesExtParams, indicesPadExtParams);
    return;
}
template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::Prepare(
    int64_t curIdx, int64_t innerBlockIdx)
{
    int64_t cur2D = curIdx % outHW;
    int64_t curNc = curIdx / outHW;
    int64_t curHo = cur2D / tilingData->wOutDim;
    int64_t curWo = cur2D % tilingData->wOutDim;

    curOriginH = tilingData->sH * curHo - tilingData->pH;
    if (curOriginH < 0) {
        curkH = Min(tilingData->kH + curOriginH, tilingData->hInDim);
        curOriginH = 0;
    } else {
        curkH = Min(tilingData->hInDim - curOriginH, tilingData->kH);
    }

    curOriginW = tilingData->sW * curWo - tilingData->pW;
    if (curOriginW < 0) {
        curkW = Min(tilingData->kW + curOriginW, tilingData->wInDim);
        curOriginW = 0;
    } else {
        curkW = Min(tilingData->wInDim - curOriginW, tilingData->kW);
    }

    if (tilingData->splitW == 0) {
        curOriginIndex =
            (curOriginH + innerBlockIdx * tilingData->kernelBlockFactorH) * tilingData->wInDim + curOriginW;
    } else {
        curOriginIndex = (curOriginH + innerBlockIdx / tilingData->splitSlice) * tilingData->wInDim + curOriginW +
                         innerBlockIdx % tilingData->splitSlice * tilingData->wSplitSize;
    }
    curInOffset = curNc * inHW + curOriginIndex;
    return;
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::BaseCompute(int64_t curIdx)
{
    int64_t realIndex = AllWInKernelProcess();
    CopyMaxOut(curIdx);
    CopyIndicesOut(realIndex, curIdx);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline int64_t MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::HwCopyInput(
    int64_t offset, int64_t blockCount, int64_t blockLen, int64_t blockLenAlign, int64_t srcStride)
{
    LocalTensor<T1> xLocal = inputQue.AllocTensor<T1>();
    int64_t alignNum = blockLenAlign - blockLen;
    DataCopyPadExtParams<T1> padExtParams;
    padExtParams.isPad = alignNum != 0;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = padExtParams.isPad ? alignNum : 0;
    padExtParams.paddingValue = minT1;

    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T1);
    extParams.srcStride = srcStride * sizeof(T1);
    extParams.dstStride = 0;

    DataCopyPad(xLocal[inputXQueOffset], xGm[offset], extParams, padExtParams);
    inputQue.EnQue(xLocal);
    return blockCount * blockLenAlign;
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline int32_t MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::Compute(int64_t dataCount)
{
    LocalTensor<T1> xLocal = inputQue.DeQue<T1>();
    LocalTensor<float> castToFP32 = xLocal.template ReinterpretCast<float>();
    InnerComputer<T1> computer;

    computer.Compute(xLocal[inputXQueOffset], castToFP32, maxUB, maskNanUB, dataCount);

    event_t eventIdVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVtoS);
    WaitFlag<HardEvent::V_S>(eventIdVtoS);

    int32_t index = 0;
    LocalTensor<uint8_t> maskNanLocal = maskNanUB.Get<uint8_t>();
    // 输入为fp16时
    if (std::is_same<T1, half>::value) {
        LocalTensor<half> maxOutLocal = maxUB.Get<half>();
        half maxIndex = maxOutLocal.GetValue(1);
        int16_t indexInt16 = *reinterpret_cast<int16_t*>(&maxIndex);
        index = static_cast<int32_t>(indexInt16);

        half maxValue = maxOutLocal.GetValue(0);
        if (IsNan(maxValue)) {
            computer.GetMask(xLocal[inputXQueOffset], castToFP32, maskNanLocal, dataCount);
            GetIndexWithLastNan(castToFP32, maskNanLocal, dataCount, index);
        }
    } else {
        LocalTensor<float> maxOutLocal = maxUB.Get<float>();
        float maxIndex = maxOutLocal.GetValue(1);
        index = *reinterpret_cast<int32_t*>(&maxIndex);
        float maxValue = maxOutLocal.GetValue(0);
        if (IsNan(maxValue)) {
            computer.GetMask(xLocal[inputXQueOffset], castToFP32, maskNanLocal, dataCount);
            GetIndexWithLastNan(castToFP32, maskNanLocal, dataCount, index);
        }
    }
    inputQue.FreeTensor<T1>(xLocal);

    return index;
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::GetIndexWithLastNan(
    LocalTensor<float> indicesMaxLocal, LocalTensor<uint8_t> maskNanLocal, int64_t dataCount, int32_t& index)
{
    LocalTensor<float> indicesLocal = indicesInitUB.Get<float>();
    Select(indicesMaxLocal, maskNanLocal, indicesLocal, float(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);

    LocalTensor<float> nanMaxIndex = nanMaxIndexUB.Get<float>();

    ReduceMax<float>(nanMaxIndex, indicesMaxLocal, indicesMaxLocal, dataCount, false);

    event_t eventIdVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVtoS);
    WaitFlag<HardEvent::V_S>(eventIdVtoS);

    index = ScalarCast<float, int32_t, RoundMode::CAST_ROUND>(nanMaxIndex.GetValue(0));
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline int64_t MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::RestoreIndex(
    int32_t index, int64_t hLen, int64_t wLen)
{
    int64_t realIndex = 0;
    if (tilingData->splitW == 0) {
        realIndex = curOriginIndex + index / curkW * tilingData->wInDim + index % curkW;
    } else {
        realIndex = curOriginIndex + index;
    }
    return realIndex;
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::CopyMaxOut(int64_t curIdx)
{
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = 1 * sizeof(T2);
    extParams.srcStride = 0;
    extParams.dstStride = 0;

    LocalTensor<T2> maxValueResult = maxUB.Get<T2>();

    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);

    DataCopyPad(maxValueWorkspaceGm[cBlockIdx], maxValueResult[0], extParams);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::CopyIndicesOut(
    int64_t maxIndex, int64_t curIdx)
{
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = sizeof(int32_t);
    extParams.srcStride = 0;
    extParams.dstStride = 0;

    LocalTensor<TINDEX> indexTensor = nanMaxIndexUBOutput.Get<TINDEX>();

    if (std::is_same<TINDEX, int32_t>::value) {
        indexTensor.SetValue(0, maxIndex);
    } else {
        extParams.blockLen = sizeof(int64_t);
        indexTensor.SetValue(0, maxIndex);
        indexTensor.SetValue(1, maxIndex >> 32);
    }

    event_t eventIdStoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdStoMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdStoMTE3);
    DataCopyPad(maxValueIndexWorkspaceGm[cBlockIdx], indexTensor, extParams);
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::NaNIndicesInit(
    LocalTensor<float> indicesLocal)
{
    int64_t hAlignkW = CeilValue(tilingData->kW, BLOCK_NUM_T1) * tilingData->kH;
    int32_t InitIndicesNum = (hAlignkW > tilingData->maxCountLength) ? tilingData->maxCountLength : hAlignkW;
    CreateVecIndex(indicesLocal, 0.0f, InitIndicesNum);
}
template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::SplitW(
    int64_t blockLen, int64_t alignBlockLen, int64_t strStride, int64_t& maxValueIndex, T2& value)
{
    int64_t kernelOffset = 0;
    int64_t inputOffset = curInOffset;
    int64_t maxIndex = 0;
    T2 maxValue = 0;
    if (alignBlockLen <= tilingData->maxCountLength) {
        int64_t eachLoopLine = tilingData->maxCountLength / alignBlockLen;
        int64_t loop = (curKernelBlockFactorH + eachLoopLine - 1) / eachLoopLine;
        int64_t tailLoopLine = curKernelBlockFactorH - (loop - 1) * eachLoopLine;
        for (int64_t hwLoop = 0; hwLoop < loop; hwLoop++) {
            int64_t blockCount = (hwLoop == loop - 1 ? tailLoopLine : eachLoopLine);
            int32_t dataCount = HwCopyInput(inputOffset, blockCount, blockLen, alignBlockLen, strStride);
            int32_t index = Compute(dataCount);
            index = KernelRealIndex(index, blockLen, alignBlockLen);
            bool first = (hwLoop == 0);
            UpdateMax(kernelOffset + index, maxValue, maxIndex, first);
            inputOffset += blockCount * tilingData->wInDim;
            kernelOffset += blockCount * blockLen;
        }
    } else {
        int64_t loopWCount = (blockLen + tilingData->maxCountLength - 1) / tilingData->maxCountLength;
        int64_t tailLoopWSize = blockLen - (loopWCount - 1) * tilingData->maxCountLength;
        for (int64_t w = 0; w < curKernelBlockFactorH; w++) {
            for (int64_t eachLoopW = 0; eachLoopW < loopWCount; eachLoopW++) {
                int64_t wBlockLen = (eachLoopW == (loopWCount - 1) ? tailLoopWSize : tilingData->maxCountLength);
                int64_t alignWBlockLen =
                    (eachLoopW == (loopWCount - 1) ? CeilValue(tailLoopWSize, BLOCK_NUM_T1) :
                                                     tilingData->maxCountLength);
                int32_t dataCount = HwCopyInput(inputOffset, 1, wBlockLen, alignWBlockLen, 0);
                int32_t index = Compute(dataCount);
                bool first = (eachLoopW == 0 && w == 0);
                UpdateMax(kernelOffset + index, maxValue, maxIndex, first);
                inputOffset += wBlockLen;
                kernelOffset += wBlockLen;
            }
            inputOffset += tilingData->wInDim - blockLen;
        }
    }
    maxValueIndex = maxIndex;
    value = maxValue;
}
template <typename T1, typename T2, typename TINDEX>
__aicore__ inline int64_t MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::AllWInKernelProcess()
{
    int64_t realIndex = 0;
    int64_t blockCount = 0;
    int64_t alignBlockLen = 0;
    int64_t blockLen = 0;
    int64_t strStride = 0;
    if (tilingData->splitW == 0) {
        blockCount = curKernelBlockFactorH;
        blockLen = curkW;
        alignBlockLen = CeilValue(curkW, BLOCK_NUM_T1);
        strStride = (tilingData->wInDim - blockLen);
    } else {
        blockCount = 1;
        blockLen = curWSplitSize;
        alignBlockLen = CeilValue(curWSplitSize, BLOCK_NUM_T1);
        strStride = 0;
    }
    int64_t inputOffset = curInOffset;
    int64_t kernelOffset = 0;
    T2 maxValue = 0;
    int64_t maxIndex = 0;
    if (blockCount * alignBlockLen <= tilingData->maxCountLength) {
        int32_t dataCount = HwCopyInput(curInOffset, blockCount, blockLen, alignBlockLen, strStride);
        int32_t index = Compute(dataCount);
        index = KernelRealIndex(index, blockLen, alignBlockLen);
        realIndex = RestoreIndex(index, curkH, curkW);
    } else {
        if (tilingData->splitW == 0) {
            SplitW(blockLen, alignBlockLen, strStride, maxIndex, maxValue);
        } else {
            int64_t loop = (curWSplitSize + tilingData->maxCountLength - 1) / tilingData->maxCountLength;
            int64_t eachBlockLen = tilingData->maxCountLength;
            int64_t tailBlockLen = curWSplitSize - (loop - 1) * eachBlockLen;
            for (int64_t hwLoop = 0; hwLoop < loop; hwLoop++) {
                blockLen = (hwLoop == loop - 1 ? tailBlockLen : eachBlockLen);
                alignBlockLen = (hwLoop == loop - 1 ? CeilValue(tailBlockLen, BLOCK_NUM_T1) : eachBlockLen);
                int32_t dataCount = HwCopyInput(inputOffset, blockCount, blockLen, alignBlockLen, strStride);
                int32_t index = Compute(dataCount);
                bool first = (hwLoop == 0);
                UpdateMax(kernelOffset + index, maxValue, maxIndex, first);
                inputOffset += blockLen;
                kernelOffset += blockLen;
            }
        }
        realIndex = RestoreIndex(maxIndex, curkH, curkW);
        LocalTensor<T2> maxOutLocal = maxUB.Get<T2>();
        maxOutLocal.SetValue(0, maxValue);
    }
    return realIndex;
}
template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::UpdateMax(
    int64_t curMaxIndex, T2& maxValue, int64_t& maxIndex, bool first)
{
    LocalTensor<T2> maxOutLocal = maxUB.Get<T2>();
    T2 curMaxValue = maxOutLocal.GetValue(0);
    if (first) {
        maxIndex = curMaxIndex;
        maxValue = curMaxValue;
        return;
    }
    if (IsNan(curMaxValue)) {
        maxIndex = curMaxIndex;
        maxValue = curMaxValue;
    } else if (curMaxValue > maxValue) {
        maxIndex = curMaxIndex;
        maxValue = curMaxValue;
    }
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline int32_t MaxPoolWithArgmaxV3BigKernelMulCore<T1, T2, TINDEX>::KernelRealIndex(
    int32_t index, int64_t blockLen, int64_t blockLenAlign)
{
    int64_t alignNum = blockLenAlign - blockLen;
    if (alignNum != 0) {
        return index - alignNum * (index / blockLenAlign);
    } else {
        return index;
    }
}

} // namespace MaxPoolWithArgmaxV3BigKernelMulCore
#endif // MAX_POOL_WITH_ARGMAX_V3_BIG_KERNEL_MUL_CORE_H_