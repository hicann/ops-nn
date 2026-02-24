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
 * \file deformable_offsets.h
 * \brief deformable_offsets kernel info
 */
#ifndef DEFORMABLE_OFFSET_H
#define DEFORMABLE_OFFSET_H
#include "kernel_operator.h"
namespace DeformableOffsets {
using namespace AscendC;
const int32_t WIDTH_OFFSET_INDEX = 0;
const int32_t HEIGHT_OFFSET_INDEX = 1;
const int32_t POINT_WEIGHT_OFFSET_INDEX = 2;
const int32_t VF_MAX_THREAD_NUM = 512;
const int32_t OFFSET_DIM_VALUE = 3;
template <typename T, typename T1, typename T2>
class DeformableOffset {
public:
    __aicore__ inline DeformableOffset()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR offsets, GM_ADDR y, GM_ADDR workspace,
        const DeformableOffsetsTilingDataSimt* __restrict tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputImgGm_;
    GlobalTensor<T> offsetsGm_;
    GlobalTensor<T> yGm_;
    T1 blockId_ = GetBlockIdx();
    const DeformableOffsetsTilingDataSimt* tiling_;
};

template <typename T, typename T1, typename T2>
__aicore__ inline void DeformableOffset<T, T1, T2>::Init(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR y, GM_ADDR workspace,
    const DeformableOffsetsTilingDataSimt* __restrict tilingData)
{
    inputImgGm_.SetGlobalBuffer((__gm__ T*)(x));
    offsetsGm_.SetGlobalBuffer((__gm__ T*)(offsets));
    yGm_.SetGlobalBuffer((__gm__ T*)(y));

    tiling_ = tilingData;
}

__aicore__ __attribute__((always_inline)) inline float GetFloorValue(float x)
{
    return __floorf(x);
}

template <typename T, typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline T GetInputPointValue(
    __gm__ T* inputImgGmAddr, T1 inputHeight, T1 inputWidth, T1 channelIndex,
    T1 inputDataBatchOffset, T1 imgHeight, T1 imgWidth, T1 imgWidthStride, T1 imgChannel)
{
    if (inputHeight >= 0 && inputWidth >= 0 && inputHeight < imgHeight && inputWidth < imgWidth) {
        return inputImgGmAddr
            [inputDataBatchOffset + inputHeight * imgWidthStride + inputWidth * imgChannel + channelIndex];
    }
    return static_cast<T>(0.0);
}

template <typename T, typename T1, typename T2>
__aicore__ __attribute__((always_inline)) inline T DeformableOffsetBilinear(
    __gm__ T* inputImgGmAddr, float pointHeight, float pointWidth, T1 channelIndex, T offsetPointWeight,
    T1 inputDataBatchOffset, T1 imgHeight, T1 imgWidth, T1 imgWidthStride, T1 imgChannel)
{
    float heightFloor = GetFloorValue(pointHeight);
    float widthFloor = GetFloorValue(pointWidth);

    float heightFloorDelta = pointHeight - heightFloor;
    float widthFloorDelta = pointWidth - widthFloor;
    // pointLeftUp
    float inputValue = static_cast<float>(GetInputPointValue<T, T1, T2>(
        (__gm__ T*)inputImgGmAddr, heightFloor, widthFloor, channelIndex, inputDataBatchOffset, imgHeight, imgWidth,
        imgWidthStride, imgChannel));
    float inputWeight = (1.0f - heightFloorDelta) * (1.0f - widthFloorDelta);
    float bilinearValue = (inputValue * inputWeight);

    // pointRightUp
    inputValue = static_cast<float>(GetInputPointValue<T, T1, T2>(
        (__gm__ T*)inputImgGmAddr, heightFloor, (widthFloor + 1), channelIndex, inputDataBatchOffset, imgHeight,
        imgWidth, imgWidthStride, imgChannel));
    inputWeight = (1.0f - heightFloorDelta) * widthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    // pointLeftBottom
    inputValue = static_cast<float>(GetInputPointValue<T, T1, T2>(
        (__gm__ T*)inputImgGmAddr, (heightFloor + 1), widthFloor, channelIndex, inputDataBatchOffset, imgHeight,
        imgWidth, imgWidthStride, imgChannel));
    inputWeight = heightFloorDelta * (1.0f - widthFloorDelta);
    bilinearValue += (inputValue * inputWeight);

    // pointRightBottom
    inputValue = static_cast<float>(GetInputPointValue<T, T1, T2>(
        (__gm__ T*)inputImgGmAddr, (heightFloor + 1), (widthFloor + 1), channelIndex, inputDataBatchOffset, imgHeight,
        imgWidth, imgWidthStride, imgChannel));
    inputWeight = heightFloorDelta * widthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    return static_cast<T>(bilinearValue * static_cast<float>(offsetPointWeight));
}

// LAUNCH_BOUND
template <typename T, typename T1, typename T2>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void ComputeDeformableOffset(
    __gm__ T* inputImgGmAddr, __gm__ T* offsetsGmAddr, __gm__ T* yGmAddr, T1 blockNumber, T1 numKernels,
    T1 imgOutWidth, T1 imgChannel, T1 imgHeight, T1 imgWidth, T1 strideH,
    T1 strideW, T1 dilationH, T1 dilationW, T1 padsH, T1 padsW, T1 dimKh,
    T1 dimKw, T1 outputPointWidthStride, T1 outputWidthStride, T1 outputKernelWidthStride,
    T1 outputBatchStride, T1 offsetBatchStride, T1 offsetKernelElementStride,
    T1 offsetPointStride, T1 offsetWidthStride, T1 imgBatchStride, T1 imgWidthStride,
    T1 groups, T1 outImgSize, T2 shiftB_, T2 mB_, T2 shiftH_, T2 mH_,
    T2 shiftW_, T2 mW_, T2 shiftC_, T2 mC_, T1 blockId_)
{
    T1 offsetGroupKernelStride = dimKh * dimKw;
    T1 heightOffset = HEIGHT_OFFSET_INDEX * offsetKernelElementStride;
    T1 widthOffset = WIDTH_OFFSET_INDEX * offsetKernelElementStride;
    T1 weightOffset = POINT_WEIGHT_OFFSET_INDEX * offsetKernelElementStride;

    for (T1 index = blockId_ * VF_MAX_THREAD_NUM + Simt::GetThreadIdx(); index < numKernels;
         index += (blockNumber * VF_MAX_THREAD_NUM)) {
        // output info (N H K_h W K_w, groups, groupC)
        T1 batchNum, heightCol, widthCol, channelIndex, groupsIndex;
        // fast division, addr/factor
        batchNum = Simt::UintDiv(static_cast<T2>(index), mB_, shiftB_);
        T1 remain = index - batchNum * outImgSize;

        heightCol = Simt::UintDiv(static_cast<T2>(remain), mH_, shiftH_);
        remain = remain - heightCol * (imgOutWidth * imgChannel);

        widthCol = Simt::UintDiv(static_cast<T2>(remain), mW_, shiftW_);
        channelIndex = remain - widthCol * imgChannel;

        groupsIndex = Simt::UintDiv(static_cast<T2>(channelIndex), mC_, shiftC_);

        T1 newIndex = batchNum * outputBatchStride;
        T1 heightInput = heightCol * strideH - padsH;
        T1 widthInput = widthCol * strideW - padsW;

        T1 outputOffset = newIndex + heightCol * outputKernelWidthStride + widthCol * outputPointWidthStride;
        T1 newOffsetIndex = batchNum * offsetBatchStride;
        T1 newInputIndex = batchNum * imgBatchStride;

        T1 offsetBaseAdrr = newOffsetIndex + heightCol * offsetWidthStride + widthCol * offsetPointStride +
                                  groupsIndex * offsetGroupKernelStride;
        for (T1 i = 0; i < dimKh; i++) {
            for (T1 j = 0; j < dimKw; j++) {
                T1 offsetAdrr = offsetBaseAdrr + (i * dimKw + j);
                // offset height info
                T1 offsetValueIndex = offsetAdrr + heightOffset;
                float pointHeight = static_cast<float>(heightInput) + static_cast<float>(i * dilationH) +
                                    static_cast<float>(offsetsGmAddr[offsetValueIndex]);
                // offset width info
                offsetValueIndex = offsetAdrr + widthOffset;
                float pointWidth = static_cast<float>(widthInput) + static_cast<float>(j * dilationW) +
                                   static_cast<float>(offsetsGmAddr[offsetValueIndex]);
                // offset weight info
                offsetValueIndex = offsetAdrr + weightOffset;
                T bilinearValue = DeformableOffsetBilinear<T, T1, T2>(
                    (__gm__ T*)(inputImgGmAddr), pointHeight, pointWidth, channelIndex, offsetsGmAddr[offsetValueIndex],
                    newInputIndex, imgHeight, imgWidth, imgWidthStride, imgChannel);
                // data layout (n, h, k_h, w, k_w, c)
                yGmAddr[outputOffset + i * outputWidthStride + j * imgChannel + channelIndex] = bilinearValue;
            }
        }
    }
}

template <typename T, typename T1, typename T2>
__aicore__ inline void DeformableOffset<T, T1, T2>::Process()
{
    T1 outImgSize = tiling_->imgOutWidth * tiling_->imgOutHeight * tiling_->imgChannel;
    T2 shiftB_, mB_, shiftH_, mH_, shiftW_, mW_, shiftC_, mC_;
    GetUintDivMagicAndShift(mB_, shiftB_, static_cast<T2>(outImgSize));
    GetUintDivMagicAndShift(mH_, shiftH_, static_cast<T2>(tiling_->imgOutWidth * tiling_->imgChannel));
    GetUintDivMagicAndShift(mW_, shiftW_, static_cast<T2>(tiling_->imgChannel));
    GetUintDivMagicAndShift(mC_, shiftC_, static_cast<T2>(tiling_->imgChannel / tiling_->deformableGroups));
    Simt::VF_CALL<ComputeDeformableOffset<T, T1, T2>>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ T*)(inputImgGm_.GetPhyAddr()),
        (__gm__ T*)(offsetsGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), tiling_->blockNum, tiling_->numKernels,
        tiling_->imgOutWidth, tiling_->imgChannel, tiling_->imgHeight, tiling_->imgWidth, tiling_->strideHeight,
        tiling_->strideWidth, tiling_->dilationHeight, tiling_->dilationWidth, tiling_->padsHeight, tiling_->padsWidth,
        tiling_->dimKHeight, tiling_->dimKWidth, tiling_->outputPointWidthStride, tiling_->outputWidthStride,
        tiling_->outputKernelWidthStride, tiling_->outputBatchStride, tiling_->offsetBatchStride,
        tiling_->offsetKernelElementStride, tiling_->offsetPointStride, tiling_->offsetWidthStride,
        tiling_->imgBatchStride, tiling_->imgWidthStride, tiling_->deformableGroups, outImgSize, shiftB_, mB_, shiftH_,
        mH_, shiftW_, mW_, shiftC_, mC_, blockId_);
}

} // namespace DeformableOffsets
#endif // DEFORMABLE_OFFSETS_H