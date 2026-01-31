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
 * \file adaptive_pool3d_big_kernel.h
 * \brief
 */

#ifndef ADAPTIVE_POOL3D_BIG_KERNEL_H_
#define ADAPTIVE_POOL3D_BIG_KERNEL_H_

#include <type_traits>
#include "kernel_operator.h"
#include "adaptive_pool3d_tiling_struct.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace AdaptivePool3d{
using namespace AscendC;

constexpr int32_t USE_BUFFER_NUM = 2;
constexpr int32_t BATCH_COPYOUT_COUNT = 1024;
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIM3 = 3;
constexpr int32_t DIGHT0 = 0;
constexpr int32_t DIGHT1 = 1;
constexpr int32_t DIGHT2 = 2;
constexpr int32_t DIGHT4 = 4;
constexpr int32_t NO_SPLIT = 0;
constexpr int32_t SPLIT_D = 1;
constexpr int32_t SPLIT_H = 2;
constexpr int32_t SPLIT_W = 3;

constexpr uint32_t MIN_FLOAT32 = 0xFF800000; // float32's -inf
constexpr uint16_t MIN_FLOAT16 = 0xFC00;     // float16(half)'s -inf
constexpr uint16_t MIN_BFLOAT16 = 0xFF80;    // bfloat16's -inf
constexpr int32_t HALF_OVERFLOW_MODE_CTRL = 48;

constexpr MicroAPI::CastTrait CASTB4TOB2 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait CASTB2TOB4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait CASTB4TOB8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T>
class AdaptivePool3dBigKernel
{
public:
    __aicore__ inline AdaptivePool3dBigKernel(const AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData &tilingData, TPipe &pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockLen, int64_t blockCount, int64_t loopCount);
    __aicore__ inline void CalcWindowSize(int64_t curIdx);
    __aicore__ inline void CopyOut(int64_t copyCount, int64_t offset);
    __aicore__ inline void CalcBatchWindowSize(int64_t startOutIdx, int64_t endOutIdx);
    template <typename U>
    __aicore__ inline U GetDtypeMinValue();

private:
    __aicore__ inline int64_t CalStartIdx(int64_t idx, int64_t inLen, int64_t outLen)
    {
        return ops::FloorDiv(idx * inLen, outLen);
    }

    __aicore__ inline int64_t CalEndIdx(int64_t idx, int64_t inLen, int64_t outLen)
    {
        return ops::CeilDiv((idx + 1) * inLen, outLen);
    }

protected:
    TPipe pipe_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> inputQue_;
    TBuf<> maxUB_;
    GlobalTensor<T> xGm_, yGm_;
    const AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData tilingData_;

    int64_t inHW_ = 1;
    int64_t inDHW_ = 1;
    int64_t outHW_ = 1;
    int64_t outDHW_ = 1;
    int64_t curNc_ = 0;
    int64_t curkD_ = 1;
    int64_t curkH_ = 1;
    int64_t curkW_ = 1;
    int64_t curkHW_ = 1;
    int64_t curkDHW_ = 1;
    int64_t curInOffset_ = 0;

    T minValue_ = 0;
};

template <typename T>
template <typename U>
__aicore__ inline U AdaptivePool3dBigKernel<T>::GetDtypeMinValue()
{
    U minValue = 0;
    if constexpr (IsSameType<U, half>::value) {
        uint16_t minFloat16 = MIN_FLOAT16;
        minValue = *reinterpret_cast<U*>(&minFloat16);
    } else if constexpr (IsSameType<U, bfloat16_t>::value) {
        uint16_t minBfloat16 = MIN_BFLOAT16;
        minValue = *reinterpret_cast<U*>(&minBfloat16);
    } else {
        uint32_t minFloat32 = MIN_FLOAT32;
        minValue = *reinterpret_cast<U*>(&minFloat32);
    }
    return minValue;
}

template <typename T>
__aicore__ inline void AdaptivePool3dBigKernel<T>::CopyIn(
    int64_t offset, int64_t blockLen, int64_t blockCount, int64_t loopCount)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    // NDDMA loopInfo init
    MultiCopyLoopInfo<DIM3> loopInfo;
    loopInfo.loopSize[DIM0] = blockLen;
    loopInfo.loopSize[DIM1] = blockCount;
    loopInfo.loopSize[DIM2] = loopCount;

    loopInfo.loopSrcStride[DIM0] = DIGHT1;
    loopInfo.loopSrcStride[DIM1] = tilingData_.wInDim;
    loopInfo.loopSrcStride[DIM2] = inHW_;

    loopInfo.loopDstStride[DIM0] = DIGHT1;
    loopInfo.loopDstStride[DIM1] = blockLen;
    loopInfo.loopDstStride[DIM2] = blockLen * blockCount;

    static constexpr MultiCopyConfig mulConfig = { false };
    MultiCopyParams<T, DIM3> paramsMain = { loopInfo };
    DataCopy<T, DIM3, mulConfig>(xLocal, xGm_[offset], paramsMain);
    inputQue_.EnQue(xLocal);
}


template <typename T>
__aicore__ inline void AdaptivePool3dBigKernel<T>::CopyOut(int64_t copyCount, int64_t offset)
{
    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    LocalTensor<T> maxLocal = maxUB_.Get<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = DIGHT1;
    extParams.blockLen = copyCount * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(yGm_[offset], maxLocal, extParams);
}

template <typename T>
__aicore__ inline void AdaptivePool3dBigKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    inHW_ = tilingData_.hInDim * tilingData_.wInDim;
    inDHW_ = tilingData_.dInDim * inHW_;
    outHW_ = tilingData_.hOutDim * tilingData_.wOutDim;
    outDHW_ = tilingData_.dOutDim * outHW_;

    minValue_ = GetDtypeMinValue<T>();

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_.InitBuffer(inputQue_, USE_BUFFER_NUM, tilingData_.maxCount * sizeof(T));
    pipe_.InitBuffer(maxUB_, BATCH_COPYOUT_COUNT * sizeof(T));
}

template <typename T>
__aicore__ inline void AdaptivePool3dBigKernel<T>::CalcWindowSize(int64_t curOutIdx)
{
    if (tilingData_.dOutDim == 1 && tilingData_.hOutDim == 1 && tilingData_.wOutDim == 1) {
        curNc_ = curOutIdx;
        curkD_ = tilingData_.dInDim;
        curkH_ = tilingData_.hInDim;
        curkW_ = tilingData_.wInDim;
        curkHW_ = curkH_ * curkW_;
        curkDHW_ = curkD_ * curkHW_;
        curInOffset_ = curNc_ * inDHW_;
        return;
    }
    curNc_ = curOutIdx / outDHW_;
    int64_t cur3D = curOutIdx - curNc_ * outDHW_;
    int64_t curDo = cur3D / outHW_;
    int64_t cur2D = cur3D - curDo * outHW_;
    int64_t curHo = cur2D / tilingData_.wOutDim;
    int64_t curWo = cur2D - curHo * tilingData_.wOutDim;

    // calc output idx startIdx and endIdx
    int64_t curOriginD = CalStartIdx(curDo, tilingData_.dInDim, tilingData_.dOutDim);
    int64_t curOriginH = CalStartIdx(curHo, tilingData_.hInDim, tilingData_.hOutDim);
    int64_t curOriginW = CalStartIdx(curWo, tilingData_.wInDim, tilingData_.wOutDim);

    curkD_ = CalEndIdx(curDo, tilingData_.dInDim, tilingData_.dOutDim) - curOriginD;
    curkH_ = CalEndIdx(curHo, tilingData_.hInDim, tilingData_.hOutDim) - curOriginH;
    curkW_ = CalEndIdx(curWo, tilingData_.wInDim, tilingData_.wOutDim) - curOriginW;
    curkHW_ = curkH_ * curkW_;
    curkDHW_ = curkD_ * curkHW_;

    // calc output idx offset on current nc
    int64_t curOriginIndex = curOriginD * inHW_ + curOriginH * tilingData_.wInDim + curOriginW;
    // calc output idx offset on whole input data
    curInOffset_ = curNc_ * inDHW_ + curOriginIndex;
}
} // namespace AdaptivePool3d
#endif // ADAPTIVE_POOL3D_BIG_KERNEL_H