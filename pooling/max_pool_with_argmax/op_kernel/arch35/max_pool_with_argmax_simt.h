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
 * \file max_pool_with_argmax_simt.h
 * \brief
 */

#ifndef CANN_MAX_POOL_WITH_ARGMAX_SIMT_H
#define CANN_MAX_POOL_WITH_ARGMAX_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "max_pool_with_argmax_struct_common.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

using namespace AscendC;

namespace SimtProc {
constexpr static uint32_t THREAD_DIM = 256;

template <typename VALUE_T, typename TYPE_T, const uint32_t NANPROP_T>
__aicore__ inline static void CycleUpdate(VALUE_T val, TYPE_T idxOffset, VALUE_T* maxVal, TYPE_T* maxIdx)
{
    if (NANPROP_T == 1) {
        if (!(static_cast<VALUE_T>(val) <= *maxVal)) {
            *maxIdx = idxOffset;
            *maxVal = val;
        }
    } else {
        if (static_cast<VALUE_T>(val) > *maxVal) {
            *maxIdx = idxOffset;
            *maxVal = val;
        }
    }
}

} // namespace SimtProc

namespace MaxPoolWithArgmaxSimtNamespace {

template <typename VALUE_T, typename INDICES_T, const uint32_t FORMAT_T, const bool ISINT64INDEX, const uint32_t NANPROP_T>
class MaxPoolWithArgmaxSimt {
public:
    __aicore__ inline MaxPoolWithArgmaxSimt(
        const MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData* __restrict tilingData)
        : tilingData_(tilingData), blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum())
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax);
    __aicore__ inline void Process();
    __aicore__ inline void Compute() const;

private:
    __aicore__ inline static INDICES_T min(INDICES_T left, INDICES_T right)
    {
        if (left <= right) {
            return left;
        }
        return right;
    }

private:
    AscendC::GlobalTensor<VALUE_T> x_;
    AscendC::GlobalTensor<VALUE_T> y_;
    AscendC::GlobalTensor<INDICES_T> argmax_;
    const MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData* tilingData_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 1;
};

template <typename VALUE_T, typename INDICES_T, const uint32_t FORMAT_T, const bool ISINT64INDEX, const uint32_t NANPROP_T>
__aicore__ inline void MaxPoolWithArgmaxSimt<VALUE_T, INDICES_T, FORMAT_T, ISINT64INDEX, NANPROP_T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR argmax)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    argmax_.SetGlobalBuffer((__gm__ INDICES_T*)(argmax));
}

template <typename VALUE_T, typename INDICES_T, const uint32_t FORMAT_T, const bool ISINT64INDEX, const uint32_t NANPROP_T>
__aicore__ inline void MaxPoolWithArgmaxSimt<VALUE_T, INDICES_T, FORMAT_T, ISINT64INDEX, NANPROP_T>::Process()
{
    Compute();
}

template <typename VALUE_T, typename INDICES_T, typename TYPE_T, const uint32_t NANPROP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolForwardNchw(
    const int64_t count, const __gm__ VALUE_T* bottomData, const int64_t channels, const int64_t height,
    const int64_t width, const int64_t outputHeight, const int64_t outputWidth, const int64_t kernelH, const int64_t kernelW,
    const int64_t strideH, const int64_t strideW, const int64_t padH, const int64_t padW, __gm__ VALUE_T* topData,
    __gm__ INDICES_T* topMask, uint32_t blockIdx, uint32_t blockNum, const int64_t includeBatchInIndex)
{
    for (TYPE_T index = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + blockNum * Simt::GetThreadNum()) {
        TYPE_T pw = index % outputWidth;
        TYPE_T ph = (index / outputWidth) % outputHeight;
        TYPE_T c = (index / outputWidth / outputHeight) % channels;
        TYPE_T n = index / outputWidth / outputHeight / channels;
        TYPE_T hstart = ph * strideH - padH;
        TYPE_T wstart = pw * strideW - padW;
        TYPE_T hend = hstart + kernelH < height ? (hstart + kernelH) : height;
        TYPE_T wend = wstart + kernelW < width ? (wstart + kernelW) : width;
        hstart = hstart > static_cast<TYPE_T>(0) ? hstart : static_cast<TYPE_T>(0);
        wstart = wstart > static_cast<TYPE_T>(0) ? wstart : static_cast<TYPE_T>(0);
        VALUE_T maxVal = AscendC::NumericLimits<VALUE_T>::Lowest();
        TYPE_T maxIdx = -1;
        TYPE_T offset = n * channels * height * width;
        auto btmData = bottomData + offset;
        for (TYPE_T h = hstart; h < hend; ++h) {
            for (TYPE_T w = wstart; w < wend; ++w) {
                TYPE_T idx = (c * height + h) * width + w;
                TYPE_T idxOffset = includeBatchInIndex ? idx + offset : idx;
                VALUE_T val = static_cast<VALUE_T>(btmData[idx]);
                SimtProc::CycleUpdate<VALUE_T, TYPE_T, NANPROP_T>(val, idxOffset, &maxVal, &maxIdx);
            }
        }
        topData[index] = static_cast<VALUE_T>(maxVal);
        topMask[index] = static_cast<INDICES_T>(maxIdx);
    }
}

template <typename VALUE_T, typename INDICES_T, typename TYPE_T, const uint32_t NANPROP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolForwardNhwc(
    const int64_t count, const __gm__ VALUE_T* bottomData, const int64_t channels, const int64_t height,
    const int64_t width, const int64_t outputHeight, const int64_t outputWidth, const int64_t kernelH, const int64_t kernelW,
    const int64_t strideH, const int64_t strideW, const int64_t padH, const int64_t padW, __gm__ VALUE_T* topData,
    __gm__ INDICES_T* topMask, uint32_t blockIdx, uint32_t blockNum, const int64_t includeBatchInIndex)
{
    for (TYPE_T index = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + blockNum * Simt::GetThreadNum()) {
        TYPE_T c = index % channels;
        TYPE_T pw = (index / channels) % outputWidth;
        TYPE_T ph = (index / channels / outputWidth) % outputHeight;
        TYPE_T n = index / channels / outputWidth / outputHeight;
        TYPE_T hstart = ph * strideH - padH;
        TYPE_T wstart = pw * strideW - padW;
        TYPE_T hend = hstart + kernelH < height ? (hstart + kernelH) : height;
        TYPE_T wend = wstart + kernelW < width ? (wstart + kernelW) : width;
        hstart = hstart > static_cast<TYPE_T>(0) ? hstart : static_cast<TYPE_T>(0);
        wstart = wstart > static_cast<TYPE_T>(0) ? wstart : static_cast<TYPE_T>(0);
        VALUE_T maxVal = AscendC::NumericLimits<VALUE_T>::Lowest();
        TYPE_T maxIdx = -1;
        TYPE_T offset = n * height * width * channels;
        auto btmData = bottomData + offset;
        for (TYPE_T h = hstart; h < hend; ++h) {
            for (TYPE_T w = wstart; w < wend; ++w) {
                TYPE_T idx = (h * width + w) * channels + c;
                TYPE_T idxOffset = includeBatchInIndex ? idx + offset : idx;
                VALUE_T val = static_cast<VALUE_T>(btmData[idx]);
                SimtProc::CycleUpdate<VALUE_T, TYPE_T, NANPROP_T>(val, idxOffset, &maxVal, &maxIdx);
            }
        }
        topData[index] = static_cast<VALUE_T>(maxVal);
        topMask[index] = static_cast<INDICES_T>(maxIdx);
    }
}

template <typename VALUE_T, typename INDICES_T, const uint32_t FORMAT_T, const bool ISINT64INDEX, const uint32_t NANPROP_T>
__aicore__ inline void MaxPoolWithArgmaxSimt<VALUE_T, INDICES_T, FORMAT_T, ISINT64INDEX, NANPROP_T>::Compute() const
{
    const int64_t kSizeH = tilingData_->kSizeH;
    const int64_t kSizeW = tilingData_->kSizeW;

    const int64_t stridesH = tilingData_->stridesH;
    const int64_t stridesW = tilingData_->stridesW;

    const int64_t padH = tilingData_->padH;
    const int64_t padW = tilingData_->padW;

    const int64_t nbatch = tilingData_->nDim;
    const int64_t channels = tilingData_->cDim;
    const int64_t height = tilingData_->hInDim;
    const int64_t width = tilingData_->wInDim;

    const int64_t outputHeight = tilingData_->hOutDim;
    const int64_t outputWidth = tilingData_->wOutDim;

    const int64_t includeBatchInIndex = tilingData_->includeBatchInIndex;

    auto inputData = (__gm__ VALUE_T*)x_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)argmax_.GetPhyAddr();
    int64_t count = nbatch * channels * outputHeight * outputWidth;
    if constexpr (FORMAT_T == 0 && !ISINT64INDEX) {
        Simt::VF_CALL<MaxPoolForwardNchw<VALUE_T, INDICES_T, int32_t, NANPROP_T>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, channels, height, width, outputHeight,
            outputWidth, kSizeH, kSizeW, stridesH, stridesW, padH, padW, outputData, indicesData, blockIdx_, blockNum_,
            includeBatchInIndex);
    } else if constexpr (FORMAT_T == 1 && !ISINT64INDEX) {
        Simt::VF_CALL<MaxPoolForwardNhwc<VALUE_T, INDICES_T, int32_t, NANPROP_T>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, channels, height, width, outputHeight,
            outputWidth, kSizeH, kSizeW, stridesH, stridesW, padH, padW, outputData, indicesData, blockIdx_, blockNum_,
            includeBatchInIndex);
    } else if constexpr (FORMAT_T == 0 && ISINT64INDEX) {
        Simt::VF_CALL<MaxPoolForwardNchw<VALUE_T, INDICES_T, int64_t, NANPROP_T>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, channels, height, width, outputHeight,
            outputWidth, kSizeH, kSizeW, stridesH, stridesW, padH, padW, outputData, indicesData, blockIdx_, blockNum_,
            includeBatchInIndex);
    } else if constexpr (FORMAT_T == 1 && ISINT64INDEX) {
        Simt::VF_CALL<MaxPoolForwardNhwc<VALUE_T, INDICES_T, int64_t, NANPROP_T>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, channels, height, width, outputHeight,
            outputWidth, kSizeH, kSizeW, stridesH, stridesW, padH, padW, outputData, indicesData, blockIdx_, blockNum_,
            includeBatchInIndex);
    }
}

} // namespace MaxPoolWithArgmaxSimtNamespace
#endif // CANN_MAX_POOL_WITH_ARGMAX_SIMT_H