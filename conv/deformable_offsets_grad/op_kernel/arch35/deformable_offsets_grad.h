/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file deformable_offsets_grad.h
 * \brief deformable_offsets_grad kernel info
 */
#ifndef DEFORMABLE_OFFSET_H
#define DEFORMABLE_OFFSET_H
#include "kernel_operator.h"
using namespace AscendC;
namespace DeformableOffsetsGrad {
const uint32_t MAX_THREAD_NUM = 512;
const uint32_t OFFSET_DIM_VALUE = 3;

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#ifndef BOUND_THREAD_NUM
#define BOUND_THREAD_NUM (512)
#endif

template <typename T, typename T1>
class DeformableOffsetGrad {
public:
    __aicore__ inline DeformableOffsetGrad()
    {}
    __aicore__ inline void Init(
        GM_ADDR grad, GM_ADDR x, GM_ADDR offsets, GM_ADDR grad_x, GM_ADDR grad_offsets,
        const DeformableOffsetsGradTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> offsetsGm_;
    GlobalTensor<T> yGradXGm_;
    GlobalTensor<T> yGradOffsetsGm_;
    const DeformableOffsetsGradTilingData* tilingData_;
    T1 blockId_ = GetBlockIdx();
};

template <typename T, typename T1>
__aicore__ inline void DeformableOffsetGrad<T, T1>::Init(
    GM_ADDR grad, GM_ADDR x, GM_ADDR offsets, GM_ADDR grad_x, GM_ADDR grad_offsets,
    const DeformableOffsetsGradTilingData* __restrict tilingData)
{
    gradGm_.SetGlobalBuffer((__gm__ T*)(grad));
    inputXGm_.SetGlobalBuffer((__gm__ T*)(x));
    offsetsGm_.SetGlobalBuffer((__gm__ T*)(offsets));
    yGradXGm_.SetGlobalBuffer((__gm__ T*)(grad_x));
    yGradOffsetsGm_.SetGlobalBuffer((__gm__ T*)(grad_offsets));
    tilingData_ = tilingData;
}

template <typename T1>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline T1 GetFloorValue(float x)
{
    float negativeValue = static_cast<float>(0.0);
    float floorFactor = static_cast<float>(-1);
    return (x >= negativeValue ? static_cast<T1>(x) : static_cast<T1>(floorFactor + x));
}

template <typename T, typename T1>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void ComputeForGetFloorValueDeformableGrad(
    __gm__ T* gradGmAddr, T1 c, T1 gradPrevIdx, float mask, float& gradOffsetsValueH, float vRow1,
    float Row1_2, float& gradOffsetsValueW, float vRow2, float Row2_2, float& gradOffsetsValueMask, float w,
    float vRow3)
{
    gradOffsetsValueH += (vRow1 * Row1_2) * static_cast<float>(gradGmAddr[gradPrevIdx + c]) * mask;
    gradOffsetsValueW += (vRow2 * Row2_2) * static_cast<float>(gradGmAddr[gradPrevIdx + c]) * mask;
    gradOffsetsValueMask += (w * vRow3) * static_cast<float>(gradGmAddr[gradPrevIdx + c]);
}

template <typename T, typename T1>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void BilinearInterpolate(
    __gm__ T* gradGmAddr, __gm__ T* inputXGmAddr, __gm__ T* offsetsGmAddr, __gm__ T* yGradXGmAddr,
    __gm__ T* yGradOffsetsGmAddr, T1 gradPrevIdx, float mask, T1 inputXIdx, T1 yGradXIdx,
    T1 gradOffsetsHIdx, T1 gradOffsetsWIdx, T1 gradOffsetsMaskIdx, float posH, float posW,
    T1 channelPerGroup, T1 inW_, T1 inChannel_, T1 inH_)
{
    T1 lowH = GetFloorValue<T1>(posH);
    T1 lowW = GetFloorValue<T1>(posW);
    T1 HighH = lowH + 1;
    T1 HighW = lowW + 1;

    float lh = static_cast<float>(posH - lowH);
    float lw = static_cast<float>(posW - lowW);
    float hh = static_cast<float>(static_cast<float>(1) - lh);
    float hw = static_cast<float>(static_cast<float>(1) - lw);

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float gradXW1 = mask * w1, gradXW2 = mask * w2, gradXW3 = mask * w3, gradXW4 = mask * w4;

    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;

    float gradOffsetsValueH = 0.0f, gradOffsetsValueW = 0.0f, gradOffsetsValueMask = 0.0f;

    // four positions cumulative calculations atomicadd separately
    if (lowH >= 0 && lowW >= 0) {
        T1 posOffset = lowH * inW_ * inChannel_ + lowW * inChannel_;
        for (T1 c = 0; c < channelPerGroup; c++) {
            T gradXValue1 = static_cast<T>(static_cast<float>(gradGmAddr[gradPrevIdx + c]) * gradXW1);
            Simt::AtomicAdd(yGradXGmAddr + (yGradXIdx + posOffset + c), gradXValue1);
            v1 = static_cast<float>(inputXGmAddr[inputXIdx + posOffset + c]);
            ComputeForGetFloorValueDeformableGrad<T, T1>(
                gradGmAddr, c, gradPrevIdx, mask, gradOffsetsValueH, -v1, hw, gradOffsetsValueW, -v1, hh,
                gradOffsetsValueMask, w1, v1);
        }
    }

    if (lowH >= 0 && HighW < inW_) {
        T1 posOffset = lowH * inW_ * inChannel_ + HighW * inChannel_;
        for (T1 c = 0; c < channelPerGroup; c++) {
            T gradXValue2 = static_cast<T>(static_cast<float>(gradGmAddr[gradPrevIdx + c]) * gradXW2);
            Simt::AtomicAdd(yGradXGmAddr + (yGradXIdx + posOffset + c), gradXValue2);
            v2 = static_cast<float>(inputXGmAddr[inputXIdx + posOffset + c]);
            ComputeForGetFloorValueDeformableGrad<T, T1>(
                gradGmAddr, c, gradPrevIdx, mask, gradOffsetsValueH, -v2, lw, gradOffsetsValueW, v2, hh,
                gradOffsetsValueMask, w2, v2);
        }
    }

    if (HighH < inH_ && lowW >= 0) {
        T1 posOffset = HighH * inW_ * inChannel_ + lowW * inChannel_;
        for (T1 c = 0; c < channelPerGroup; c++) {
            T gradXValue3 = static_cast<T>(static_cast<float>(gradGmAddr[gradPrevIdx + c]) * gradXW3);
            Simt::AtomicAdd(yGradXGmAddr + (yGradXIdx + posOffset + c), gradXValue3);
            v3 = static_cast<float>(inputXGmAddr[inputXIdx + posOffset + c]);
            ComputeForGetFloorValueDeformableGrad<T, T1>(
                gradGmAddr, c, gradPrevIdx, mask, gradOffsetsValueH, v3, hw, gradOffsetsValueW, -v3, lh,
                gradOffsetsValueMask, w3, v3);
        }
    }

    if (HighH < inH_ && HighW < inW_) {
        T1 posOffset = HighH * inW_ * inChannel_ + HighW * inChannel_;
        for (T1 c = 0; c < channelPerGroup; c++) {
            T gradXValue4 = static_cast<T>(static_cast<float>(gradGmAddr[gradPrevIdx + c]) * gradXW4);
            Simt::AtomicAdd(yGradXGmAddr + (yGradXIdx + posOffset + c), gradXValue4);
            v4 = static_cast<float>(inputXGmAddr[inputXIdx + posOffset + c]);
            ComputeForGetFloorValueDeformableGrad<T, T1>(
                gradGmAddr, c, gradPrevIdx, mask, gradOffsetsValueH, v4, lw, gradOffsetsValueW, v4, lh,
                gradOffsetsValueMask, w4, v4);
        }
    }
    // assign gradoffsets value to GM
    yGradOffsetsGmAddr[gradOffsetsHIdx] = gradOffsetsValueH;
    yGradOffsetsGmAddr[gradOffsetsWIdx] = gradOffsetsValueW;
    yGradOffsetsGmAddr[gradOffsetsMaskIdx] = gradOffsetsValueMask;
}

template <typename T, typename T1>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void DoComputePerGroup(
    __gm__ T* gradGmAddr, __gm__ T* inputXGmAddr, __gm__ T* offsetsGmAddr, __gm__ T* yGradXGmAddr,
    __gm__ T* yGradOffsetsGmAddr, T1 gradIdx, T1 inputXIdx, T1 offsetsIdx, T1 yGradXIdx,
    T1 yGradOffsetsIdx, T1 channelPerGroup, T1 iHo, T1 iWo, T1 iKh, T1 iKw,
    T1 outW_, T1 inW_, T1 inChannel_, T1 strideH_, T1 strideW_, T1 dilationH_,
    T1 dilationW_, T1 padsH_, T1 padsW_, T1 kSizeH_, T1 kSizeW_, T1 batchSize_,
    T1 deformableGroup_, T1 offsetsChannel_, T1 inH_)
{
    T zero = static_cast<T>(0);
    // calculate input coordinates based on output
    float hInPos = static_cast<float>(iHo * strideH_ + iKh * dilationH_ - padsH_);
    float wInPos = static_cast<float>(iWo * strideW_ + iKw * dilationW_ - padsW_);
    T1 posOffset = iHo * outW_ * offsetsChannel_ + iWo * offsetsChannel_ + iKh * kSizeW_ + iKw;
    T1 offsetsWIdx = offsetsIdx + posOffset;
    T1 offsetsHIdx = offsetsWIdx + deformableGroup_ * kSizeH_ * kSizeW_;
    T1 offsetsMaskIdx = offsetsWIdx + 2 * deformableGroup_ * kSizeH_ * kSizeW_;
    float mask = static_cast<float>(offsetsGmAddr[offsetsMaskIdx]);

    T1 gradOffsetsWIdx = yGradOffsetsIdx + posOffset;
    T1 gradOffsetsHIdx = gradOffsetsWIdx + deformableGroup_ * kSizeH_ * kSizeW_;
    T1 gradOffsetsMaskIdx = gradOffsetsWIdx + 2 * deformableGroup_ * kSizeH_ * kSizeW_;

    float originOffsetH = static_cast<float>(hInPos + static_cast<float>(offsetsGmAddr[offsetsHIdx]));
    float originOffsetW = static_cast<float>(wInPos + static_cast<float>(offsetsGmAddr[offsetsWIdx]));

    // bilinear interpolate within the range
    if (originOffsetH > -1 && originOffsetH < inH_ && originOffsetW > -1 && originOffsetW < inW_) {
        T1 gradYPixelIdx =
            gradIdx + inChannel_ * (iHo * kSizeH_ * outW_ * kSizeW_ + iKh * outW_ * kSizeW_ + iWo * kSizeW_ + iKw);
        BilinearInterpolate<T, T1>(
            (__gm__ T*)gradGmAddr, (__gm__ T*)inputXGmAddr, (__gm__ T*)offsetsGmAddr, (__gm__ T*)yGradXGmAddr,
            (__gm__ T*)yGradOffsetsGmAddr, gradYPixelIdx, mask, inputXIdx, yGradXIdx, gradOffsetsHIdx, gradOffsetsWIdx,
            gradOffsetsMaskIdx, originOffsetH, originOffsetW, channelPerGroup, inW_, inChannel_, inH_);
    } else {
        yGradOffsetsGmAddr[gradOffsetsWIdx] = zero;
        yGradOffsetsGmAddr[gradOffsetsHIdx] = zero;
        yGradOffsetsGmAddr[gradOffsetsMaskIdx] = zero;
    }
}

template <typename T, typename T1>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__ void ComputeSetValueGradX(
    __gm__ T* gradGmAddr, __gm__ T* inputXGmAddr, __gm__ T* offsetsGmAddr, __gm__ T* yGradXGmAddr,
    __gm__ T* yGradOffsetsGmAddr, T1 blockClearProcessNum, T1 blockClearStartOffset)
{
    for (T1 idx = static_cast<T1>(Simt::GetThreadIdx()); idx < blockClearProcessNum;
         idx += static_cast<T1>(Simt::GetThreadNum())) {
        T1 curIdx = idx + blockClearStartOffset;
        yGradXGmAddr[curIdx] = static_cast<T>(0);
    }
}

template <typename T, typename T1>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__ void ComputeDeformableOffsetGrad(
    __gm__ T* gradGmAddr, __gm__ T* inputXGmAddr, __gm__ T* offsetsGmAddr, __gm__ T* yGradXGmAddr,
    __gm__ T* yGradOffsetsGmAddr, T1 blockProcessNum, T1 blockStartOffset, T1 kSizeH_,
    T1 kSizeW_, T1 deformableGroup_, T1 offsetsChannel_, T1 outH_, T1 outW_,
    T1 inH_, T1 inW_, T1 inChannel_, T1 strideH_, T1 strideW_, T1 dilationH_,
    T1 dilationW_, T1 padsH_, T1 padsW_, T1 batchSize_)
{
    T1 channelPerGroup = inChannel_ / deformableGroup_;
    for (T1 idx = static_cast<T1>(Simt::GetThreadIdx()); idx < blockProcessNum;
         idx += static_cast<T1>(Simt::GetThreadNum())) {
        T1 curIdx = idx + blockStartOffset;

        // split multi-core and multi-thread according to the N * Ho * Wo * DeformableGroup * Kh *Kw
        T1 iKw = curIdx % kSizeW_;
        T1 iKh = (curIdx / kSizeW_) % kSizeH_;
        T1 iDg = (curIdx / kSizeW_ / kSizeH_) % deformableGroup_;
        T1 iWo = (curIdx / kSizeW_ / kSizeH_ / deformableGroup_) % outW_;
        T1 iHo = (curIdx / kSizeW_ / kSizeH_ / deformableGroup_ / outW_) % outH_;
        T1 iBatch = curIdx / kSizeW_ / kSizeH_ / deformableGroup_ / outW_ / outH_;

        T1 gradIdx = iBatch * inChannel_ * kSizeH_ * outH_ * kSizeW_ * outW_ + iDg * channelPerGroup;
        T1 inputXIdx = iBatch * inChannel_ * inH_ * inW_ + iDg * channelPerGroup;
        T1 offsetsIdx = iBatch * outH_ * outW_ * offsetsChannel_ + iDg * kSizeH_ * kSizeW_;
        T1 yGradXIdx = iBatch * inChannel_ * inH_ * inW_ + iDg * channelPerGroup;
        T1 yGradOffsetsIdx = iBatch * outH_ * outW_ * offsetsChannel_ + iDg * kSizeH_ * kSizeW_;

        DoComputePerGroup<T, T1>(
            (__gm__ T*)gradGmAddr, (__gm__ T*)inputXGmAddr, (__gm__ T*)offsetsGmAddr, (__gm__ T*)yGradXGmAddr,
            (__gm__ T*)yGradOffsetsGmAddr, gradIdx, inputXIdx, offsetsIdx, yGradXIdx, yGradOffsetsIdx, channelPerGroup,
            iHo, iWo, iKh, iKw, outW_, inW_, inChannel_, strideH_, strideW_, dilationH_, dilationW_, padsH_, padsW_,
            kSizeH_, kSizeW_, batchSize_, deformableGroup_, offsetsChannel_, inH_);
    }
}

template <typename T, typename T1>
__aicore__ inline void DeformableOffsetGrad<T, T1>::Process()
{
    T1 blockClearProcessNum = tilingData_->gradXFactor;
    T1 blockClearStartOffset = tilingData_->gradXFactor * blockId_;
    T1 offsetsChannel_ =
        tilingData_->deformableGroups * OFFSET_DIM_VALUE * tilingData_->dimKHeight * tilingData_->dimKWidth;
    if (blockId_ < tilingData_->gradXFactorTail) {
        blockClearProcessNum += 1;
        blockClearStartOffset += blockId_;
    } else {
        blockClearStartOffset += tilingData_->gradXFactorTail;
    }

    T1 blockProcessNum = tilingData_->blockFactor;
    T1 blockStartOffset = tilingData_->blockFactor * blockId_;
    if (blockId_ < tilingData_->blockFactorTail) {
        blockProcessNum += 1;
        blockStartOffset += blockId_;
    } else {
        blockStartOffset += tilingData_->blockFactorTail;
    }

    if (blockId_ < tilingData_->clearGradXCoreNum) {
        Simt::VF_CALL<ComputeSetValueGradX<T, T1>>(
            Simt::Dim3{MAX_THREAD_NUM, 1, 1}, (__gm__ T*)gradGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
            (__gm__ T*)offsetsGm_.GetPhyAddr(), (__gm__ T*)yGradXGm_.GetPhyAddr(),
            (__gm__ T*)yGradOffsetsGm_.GetPhyAddr(), blockClearProcessNum, blockClearStartOffset);
    }
    SyncAll();
    if (blockId_ < tilingData_->realCoreNum) {
        Simt::VF_CALL<ComputeDeformableOffsetGrad<T, T1>>(
            Simt::Dim3{MAX_THREAD_NUM, 1, 1}, (__gm__ T*)gradGm_.GetPhyAddr(), (__gm__ T*)inputXGm_.GetPhyAddr(),
            (__gm__ T*)offsetsGm_.GetPhyAddr(), (__gm__ T*)yGradXGm_.GetPhyAddr(),
            (__gm__ T*)yGradOffsetsGm_.GetPhyAddr(), blockProcessNum, blockStartOffset, tilingData_->dimKHeight,
            tilingData_->dimKWidth, tilingData_->deformableGroups, offsetsChannel_, tilingData_->imgOutHeight,
            tilingData_->imgOutWidth, tilingData_->imgHeight, tilingData_->imgWidth, tilingData_->imgChannel,
            tilingData_->strideHeight, tilingData_->strideWidth, tilingData_->dilationHeight,
            tilingData_->dilationWidth, tilingData_->padsHeight, tilingData_->padsWidth, tilingData_->imgBatchNum);
    }
}

} // namespace DeformableOffsetsGrad
#endif