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
 * \file max_pool_grad_with_argmax_v3_nchw_scalar.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_V3_SCALAR_KERNEL_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_V3_SCALAR_KERNEL_H_
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool_grad_with_argmax_v3_base.h"
namespace MaxPoolGradWithArgmaxV3NCHWScalarNameSpace {
constexpr int32_t INVALID_INDEX_VALUE = -1;
template <typename T1, typename T2>
class MaxPoolGradWithArgmaxV3NCHWScalar {
public:
    __aicore__ inline MaxPoolGradWithArgmaxV3NCHWScalar(
        const MaxPoolGradWithArgmaxV3NCHWScalarTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute(
        LocalTensor<computeType>& yLocal, int64_t argmaxNcActual, int64_t argmaxHActual, int64_t argmaxWActual,
        int64_t argmaxLoopIndex);
    __aicore__ inline uint32_t ConvertIndexToUBIndex(int64_t indexValue);
    __aicore__ inline void ComputeActualOffset(
        int64_t loopIndex, int64_t& argmaxNcActual, int64_t& argmaxHActual, int64_t& argmaxWActual,
        int64_t& argmaxGmOffset, int64_t& argmaxNcIndex);
    __aicore__ inline void CopyInArgmaxGrad(
        int64_t& argmaxNcActual, int64_t& argmaxHActual, int64_t& argmaxWActual, int64_t& argmaxGmOffset);
    __aicore__ inline uint32_t ConvertIndexToUBIndex(int64_t indexValue, int64_t innerNcIndex, int64_t argmaxLoopIndex);

private:
    const MaxPoolGradWithArgmaxV3NCHWScalarTilingData& tilingData_;
    TPipe& pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;
    int64_t blockIdx_ = 0;
    int64_t curCoreProcessNum_ = 1;
    int64_t argmaxPlaneSize_ = 1;
    int64_t highAxisIndex_ = 0;
    int64_t highAxisActual_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t hOutputActual_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t wOutputActual_ = 0;
    int64_t highAxisArgmaxOffset_ = 0;
    int64_t hAxisArgmaxOffset_ = 0;
    int64_t wAxisArgmaxOffset_ = 0;
};
template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    argmaxPlaneSize_ = tilingData_.hArgmax * tilingData_.wArgmax;
    curCoreProcessNum_ =
        (blockIdx_ + 1 == tilingData_.usedCoreNum) ? tilingData_.tailCoreProcessNum : tilingData_.normalCoreProcessNum;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);

    pipe_.InitBuffer(outputQue_, BUFFER_NUM, tilingData_.outputBufferSize);
    pipe_.InitBuffer(gradQue_, BUFFER_NUM, tilingData_.gradBufferSize);
    pipe_.InitBuffer(argmaxQue_, BUFFER_NUM, tilingData_.argmaxBufferSize);
    return;
}

template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
    return;
}
template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_.normalCoreProcessNum + loopNum;
    highAxisIndex_ = baseBlockIdx / (tilingData_.hOutputOuter * tilingData_.wOutputOuter);
    highAxisActual_ =
        (highAxisIndex_ == (tilingData_.highAxisOuter - 1) ? tilingData_.highAxisTail : tilingData_.highAxisInner);

    int64_t tempTail = baseBlockIdx - highAxisIndex_ * tilingData_.hOutputOuter * tilingData_.wOutputOuter;
    hAxisIndex_ = tempTail / tilingData_.wOutputOuter;
    hOutputActual_ =
        (hAxisIndex_ == (tilingData_.hOutputOuter - 1) ? tilingData_.hOutputTail : tilingData_.hOutputInner);

    wAxisIndex_ = tempTail - hAxisIndex_ * tilingData_.wOutputOuter;
    wOutputActual_ =
        (wAxisIndex_ == (tilingData_.wOutputOuter - 1) ? tilingData_.wOutputTail : tilingData_.wOutputInner);

    int64_t hArgmaxActualStart = PStart(
        hAxisIndex_ * tilingData_.hOutputInner, tilingData_.padH, tilingData_.hKernel, tilingData_.dilationH,
        tilingData_.hStride);
    int64_t wArgmaxActualStart = PStart(
        wAxisIndex_ * tilingData_.wOutputInner, tilingData_.padW, tilingData_.wKernel, tilingData_.dilationW,
        tilingData_.wStride);

    highAxisArgmaxOffset_ = highAxisIndex_ * tilingData_.highAxisInner * argmaxPlaneSize_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * tilingData_.wArgmax;
    wAxisArgmaxOffset_ = wArgmaxActualStart;
    return;
}
template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::ComputeActualOffset(
    int64_t loopIndex, int64_t& argmaxNcActual, int64_t& argmaxHActual, int64_t& argmaxWActual, int64_t& argmaxGmOffset,
    int64_t& argmaxNcIndex)
{
    argmaxNcIndex = loopIndex / (tilingData_.argmaxHOuter * tilingData_.argmaxWOuter);
    argmaxNcActual =
        (argmaxNcIndex == (tilingData_.argmaxNcOuter - 1) ? tilingData_.argmaxNcTail : tilingData_.argmaxNcInner);
    int64_t remain = loopIndex - argmaxNcIndex * tilingData_.argmaxHOuter * tilingData_.argmaxWOuter;
    int64_t argmaxHIndex = remain / tilingData_.argmaxWOuter;
    argmaxHActual =
        (argmaxHIndex == (tilingData_.argmaxHOuter - 1) ? tilingData_.argmaxHTail : tilingData_.argmaxHInner);
    int64_t argmaxWIndex = remain - argmaxHIndex * tilingData_.argmaxWOuter;
    argmaxWActual =
        (argmaxWIndex == (tilingData_.argmaxWOuter - 1) ? tilingData_.argmaxWTail : tilingData_.argmaxWInner);
    argmaxGmOffset = highAxisArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_ +
                     argmaxNcIndex * tilingData_.argmaxNcInner * tilingData_.hArgmax * tilingData_.wArgmax +
                     argmaxHIndex * tilingData_.argmaxHInner * tilingData_.wArgmax +
                     argmaxWIndex * tilingData_.argmaxWInner;
    return;
}
template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::ProcessPerLoop()
{
    uint32_t calCount = static_cast<uint32_t>(tilingData_.outputBufferSize) / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);
    int64_t argmaxNcActual = 0;
    int64_t argmaxHActual = 0;
    int64_t argmaxWActual = 0;
    int64_t argmaxGmOffset = 0;
    int64_t argmaxNcIndex = 0;
    for (int64_t i = 0; i < tilingData_.argmaxInnerLoop; i++) {
        ComputeActualOffset(i, argmaxNcActual, argmaxHActual, argmaxWActual, argmaxGmOffset, argmaxNcIndex);
        CopyInArgmaxGrad(argmaxNcActual, argmaxHActual, argmaxWActual, argmaxGmOffset);
        Compute(yLocal, argmaxNcActual, argmaxHActual, argmaxWActual, argmaxNcIndex);
    }
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}
template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::CopyInArgmaxGrad(
    int64_t& argmaxNcActual, int64_t& argmaxHActual, int64_t& argmaxWActual, int64_t& argmaxGmOffset)
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    DataCopyExtParams copyInParamT1 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(argmaxNcActual * argmaxHActual * argmaxWActual * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyInParamT1, paramsT1);

    DataCopyPadExtParams<T2> paramsT2 = {false, 0, 0, 0};
    DataCopyExtParams copyInParamT2 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(argmaxNcActual * argmaxHActual * argmaxWActual * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(argmaxLocal, argmaxGm_[argmaxGmOffset], copyInParamT2, paramsT2);
    gradQue_.EnQue(gradLocal);
    argmaxQue_.EnQue(argmaxLocal);
    return;
}
template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    int64_t outputPlaneSize = tilingData_.hOutput * tilingData_.wOutput;
    int64_t highOutputAxisOffset = highAxisIndex_ * tilingData_.highAxisInner * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.wOutput;
    int64_t wOutputAxisOffset = wAxisIndex_ * tilingData_.wOutputInner;
    int64_t outputGmOffset = highOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset;

    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(highAxisActual_ * hOutputActual_ * wOutputActual_ * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    outputQue_.FreeTensor(yLocal);
}
template <typename T1, typename T2>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::Compute(
    LocalTensor<computeType>& yLocal, int64_t argmaxNcActual, int64_t argmaxHActual, int64_t argmaxWActual,
    int64_t argmaxLoopIndex)
{
    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    computeType gradValue;
    int64_t argmaxCountInner = argmaxHActual * argmaxWActual;
    for (int64_t i = 0; i < argmaxNcActual; i++) {
        int64_t argmaxOffsetHigh = i * argmaxCountInner;
        for (int64_t j = 0; j < argmaxCountInner; j++) {
            int64_t indexValue = argmaxLocal.GetValue(argmaxOffsetHigh + j);
            int32_t outputIndexInUB = ConvertIndexToUBIndex(indexValue, i, argmaxLoopIndex);
            if (outputIndexInUB == INVALID_INDEX_VALUE) {
                continue;
            }
            if constexpr (std::is_same<T1, bfloat16_t>::value) {
                gradValue = ToFloat(gradLocal.GetValue(argmaxOffsetHigh + j));
            } else {
                gradValue = static_cast<computeType>(gradLocal.GetValue(argmaxOffsetHigh + j));
            }
            computeType ubValue = yLocal.GetValue(outputIndexInUB);
            yLocal.SetValue(outputIndexInUB, gradValue + ubValue);
        }
    }
    gradQue_.FreeTensor(gradLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
    return;
}
template <typename T1, typename T2>
__aicore__ inline uint32_t MaxPoolGradWithArgmaxV3NCHWScalar<T1, T2>::ConvertIndexToUBIndex(
    int64_t indexValue, int64_t innerNcIndex, int64_t argmaxLoopIndex)
{
    int64_t curHStartIndex = hAxisIndex_ * tilingData_.hOutputInner;
    int64_t curWStartIndex = wAxisIndex_ * tilingData_.wOutputInner;

    int64_t curHEndIndex = curHStartIndex + hOutputActual_;
    int64_t curWEndIndex = curWStartIndex + wOutputActual_;

    int64_t relativeHIndex = indexValue / tilingData_.wOutput;
    int64_t relativeWIndex = indexValue - relativeHIndex * tilingData_.wOutput;

    if (relativeHIndex < curHStartIndex || relativeHIndex > curHEndIndex || relativeWIndex < curWStartIndex ||
        relativeWIndex > curWEndIndex) {
        return INVALID_INDEX_VALUE;
    }
    int32_t ubIndex = (relativeHIndex - curHStartIndex) * wOutputActual_ + (relativeWIndex - curWStartIndex) +
                      (argmaxLoopIndex * tilingData_.argmaxNcInner + innerNcIndex) * tilingData_.hOutputInner *
                          tilingData_.wOutputInner;
    return ubIndex;
}
} // namespace MaxPoolGradWithArgmaxV3NCHWScalarNameSpace
#endif