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
const uint32_t WIDTH_OFFSET_INDEX = 0;
const uint32_t HEIGHT_OFFSET_INDEX = 1;
const uint32_t POINT_WEIGHT_OFFSET_INDEX = 2;
const uint32_t VF_MAX_THREAD_NUM = 512;
const uint32_t OFFSET_DIM_VALUE = 3;
template <typename T>
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
    uint32_t blockId_ = GetBlockIdx();
    const DeformableOffsetsTilingDataSimt* tiling_;
};

template <typename T>
__aicore__ inline void DeformableOffset<T>::Init(
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

template <typename T>
__aicore__ __attribute__((always_inline)) inline T GetInputPointValue(
    __gm__ T* inputImgGmAddr, int32_t inputHeight, int32_t inputWidth, uint32_t channelIndex,
    uint32_t inputDataBatchOffset, uint32_t imgHeight, uint32_t imgWidth, uint32_t imgWidthStride, uint32_t imgChannel)
{
    if (inputHeight >= 0 && inputWidth >= 0 && inputHeight < imgHeight && inputWidth < imgWidth) {
        return inputImgGmAddr
            [inputDataBatchOffset + inputHeight * imgWidthStride + inputWidth * imgChannel + channelIndex];
    }
    return static_cast<T>(0.0);
}

template <typename T>
__aicore__ __attribute__((always_inline)) inline T DeformableOffsetBilinear(
    __gm__ T* inputImgGmAddr, float pointHeight, float pointWidth, uint32_t channelIndex, T offsetPointWeight,
    uint32_t inputDataBatchOffset, uint32_t imgHeight, uint32_t imgWidth, uint32_t imgWidthStride, uint32_t imgChannel)
{
    float heightFloor = GetFloorValue(pointHeight);
    float widthFloor = GetFloorValue(pointWidth);

    float heightFloorDelta = pointHeight - heightFloor;
    float widthFloorDelta = pointWidth - widthFloor;
    // pointLeftUp
    float inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, heightFloor, widthFloor, channelIndex, inputDataBatchOffset, imgHeight, imgWidth,
        imgWidthStride, imgChannel));
    float inputWeight = (1.0f - heightFloorDelta) * (1.0f - widthFloorDelta);
    float bilinearValue = (inputValue * inputWeight);

    // pointRightUp
    inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, heightFloor, (widthFloor + 1), channelIndex, inputDataBatchOffset, imgHeight,
        imgWidth, imgWidthStride, imgChannel));
    inputWeight = (1.0f - heightFloorDelta) * widthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    // pointLeftBottom
    inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, (heightFloor + 1), widthFloor, channelIndex, inputDataBatchOffset, imgHeight,
        imgWidth, imgWidthStride, imgChannel));
    inputWeight = heightFloorDelta * (1.0f - widthFloorDelta);
    bilinearValue += (inputValue * inputWeight);

    // pointRightBottom
    inputValue = static_cast<float>(GetInputPointValue(
        (__gm__ T*)inputImgGmAddr, (heightFloor + 1), (widthFloor + 1), channelIndex, inputDataBatchOffset, imgHeight,
        imgWidth, imgWidthStride, imgChannel));
    inputWeight = heightFloorDelta * widthFloorDelta;
    bilinearValue += (inputValue * inputWeight);

    return static_cast<T>(bilinearValue * static_cast<float>(offsetPointWeight));
}

// LAUNCH_BOUND
template <typename T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) __aicore__ void ComputeDeformableOffset(
    __gm__ T* inputImgGmAddr, __gm__ T* offsetsGmAddr, __gm__ T* yGmAddr, uint32_t blockNumber, uint32_t numKernels,
    uint32_t imgOutWidth, uint32_t imgChannel, uint32_t imgHeight, uint32_t imgWidth, uint32_t strideH,
    uint32_t strideW, uint32_t dilationH, uint32_t dilationW, uint32_t padsH, uint32_t padsW, uint32_t dimKh,
    uint32_t dimKw, uint32_t outputPointWidthStride, uint32_t outputWidthStride, uint32_t outputKernelWidthStride,
    uint32_t outputBatchStride, uint32_t offsetBatchStride, uint32_t offsetKernelElementStride,
    uint32_t offsetPointStride, uint32_t offsetWidthStride, uint32_t imgBatchStride, uint32_t imgWidthStride,
    uint32_t groups, uint32_t outImgSize, uint32_t shiftB_, uint32_t mB_, uint32_t shiftH_, uint32_t mH_,
    uint32_t shiftW_, uint32_t mW_, uint32_t shiftC_, uint32_t mC_, uint32_t blockId_)
{
    uint32_t offsetGroupKernelStride = dimKh * dimKw;
    uint32_t heightOffset = HEIGHT_OFFSET_INDEX * offsetKernelElementStride;
    uint32_t widthOffset = WIDTH_OFFSET_INDEX * offsetKernelElementStride;
    uint32_t weightOffset = POINT_WEIGHT_OFFSET_INDEX * offsetKernelElementStride;

    for (uint32_t index = blockId_ * VF_MAX_THREAD_NUM + Simt::GetThreadIdx(); index < numKernels;
         index += (blockNumber * VF_MAX_THREAD_NUM)) {
        // output info (N H K_h W K_w, groups, groupC)
        uint32_t batchNum, heightCol, widthCol, channelIndex, groupsIndex;
        // fast division, addr/factor
        batchNum = Simt::UintDiv(index, mB_, shiftB_);
        uint32_t remain = index - batchNum * outImgSize;

        heightCol = Simt::UintDiv(remain, mH_, shiftH_);
        remain = remain - heightCol * (imgOutWidth * imgChannel);

        widthCol = Simt::UintDiv(remain, mW_, shiftW_);
        channelIndex = remain - widthCol * imgChannel;

        groupsIndex = Simt::UintDiv(channelIndex, mC_, shiftC_);

        uint32_t newIndex = batchNum * outputBatchStride;
        int32_t heightInput = heightCol * strideH - padsH;
        int32_t widthInput = widthCol * strideW - padsW;

        uint32_t outputOffset = newIndex + heightCol * outputKernelWidthStride + widthCol * outputPointWidthStride;
        uint32_t newOffsetIndex = batchNum * offsetBatchStride;
        uint32_t newInputIndex = batchNum * imgBatchStride;

        uint32_t offsetBaseAdrr = newOffsetIndex + heightCol * offsetWidthStride + widthCol * offsetPointStride +
                                  groupsIndex * offsetGroupKernelStride;
        for (int32_t i = 0; i < dimKh; i++) {
            for (int32_t j = 0; j < dimKw; j++) {
                uint32_t offsetAdrr = offsetBaseAdrr + (i * dimKw + j);
                // offset height info
                uint32_t offsetValueIndex = offsetAdrr + heightOffset;
                float pointHeight = static_cast<float>(heightInput) + static_cast<float>(i * dilationH) +
                                    static_cast<float>(offsetsGmAddr[offsetValueIndex]);
                // offset width info
                offsetValueIndex = offsetAdrr + widthOffset;
                float pointWidth = static_cast<float>(widthInput) + static_cast<float>(j * dilationW) +
                                   static_cast<float>(offsetsGmAddr[offsetValueIndex]);
                // offset weight info
                offsetValueIndex = offsetAdrr + weightOffset;
                T bilinearValue = DeformableOffsetBilinear(
                    (__gm__ T*)(inputImgGmAddr), pointHeight, pointWidth, channelIndex, offsetsGmAddr[offsetValueIndex],
                    newInputIndex, imgHeight, imgWidth, imgWidthStride, imgChannel);
                // data layout (n, h, k_h, w, k_w, c)
                yGmAddr[outputOffset + i * outputWidthStride + j * imgChannel + channelIndex] = bilinearValue;
            }
        }
    }
}

template <typename T>
__aicore__ inline void DeformableOffset<T>::Process()
{
    uint32_t outImgSize = tiling_->imgOutWidth * tiling_->imgOutHeight * tiling_->imgChannel;
    uint32_t shiftB_, mB_, shiftH_, mH_, shiftW_, mW_, shiftC_, mC_;
    GetUintDivMagicAndShift(mB_, shiftB_, outImgSize);
    GetUintDivMagicAndShift(mH_, shiftH_, tiling_->imgOutWidth * tiling_->imgChannel);
    GetUintDivMagicAndShift(mW_, shiftW_, tiling_->imgChannel);
    GetUintDivMagicAndShift(mC_, shiftC_, tiling_->imgChannel / tiling_->deformableGroups);
    Simt::VF_CALL<ComputeDeformableOffset<T>>(
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