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
 * \file max_pool_with_argmax_v3_simt.h
 * \brief max_pool_with_argmax_v3 implied by simt
 */

#ifndef CANN_MAX_POOL_WITH_ARGMAX_V3_SIMT_H
#define CANN_MAX_POOL_WITH_ARGMAX_V3_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

using namespace AscendC;

namespace SimtProc {
constexpr static uint32_t THREAD_DIM = 256;

template <typename idx_accscalar_t>
__aicore__ inline static void CycleUpdate(float val, idx_accscalar_t idxOffset, float* maxval, idx_accscalar_t* maxidx)
{
    if ((static_cast<float>(val) > *maxval) || Simt::IsNan(val)) {
        *maxidx = idxOffset;
        *maxval = val;
    }
}
} // namespace SimtProc

template <typename VALUE_T, typename INDICES_T, int Format_T, bool useINT64Index>
class MaxPoolWithArgmaxV3 {
public:
    __aicore__ inline MaxPoolWithArgmaxV3(const MaxPoolWithArgmaxV3SimtTilingData* __restrict tilingData)
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
    const MaxPoolWithArgmaxV3SimtTilingData* tilingData_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 1;
    const uint32_t F32_NEG_INF = 0xff800000;
};

template <typename VALUE_T, typename INDICES_T, int Format_T, bool useINT64Index>
__aicore__ inline void MaxPoolWithArgmaxV3<VALUE_T, INDICES_T, Format_T, useINT64Index>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR argmax)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    argmax_.SetGlobalBuffer((__gm__ INDICES_T*)(argmax));
}

template <typename VALUE_T, typename INDICES_T, int Format_T, bool useINT64Index>
__aicore__ inline void MaxPoolWithArgmaxV3<VALUE_T, INDICES_T, Format_T, useINT64Index>::Process()
{
    Compute();
}

template <typename scalar_t, typename idx_scalar_t, typename idx_accscalar_t>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolForwardNchw(
    const int64_t count, const __gm__ scalar_t* bottomData, const int64_t height, const int64_t width,
    const int outputHeight, const int outputWidth, const int kernelH, const int kernelW, const int strideH,
    const int strideW, const int padH, const int padW, const int dilationH, const int dilationW,
    __gm__ scalar_t* topData, __gm__ idx_scalar_t* topMask, int blockIdx, int blockNum)
{
    for (idx_accscalar_t index = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + blockNum * Simt::GetThreadNum()) {
        idx_accscalar_t pw = index % outputWidth;
        idx_accscalar_t ph = (index / outputWidth) % outputHeight;
        idx_accscalar_t nxc = index / outputWidth / outputHeight;
        idx_accscalar_t hstart = ph * strideH - padH;
        idx_accscalar_t wstart = pw * strideW - padW;
        idx_accscalar_t hend =
            (hstart + (kernelH - 1) * dilationH + 1) < height ? (hstart + (kernelH - 1) * dilationH + 1) : height;
        idx_accscalar_t wend =
            (wstart + (kernelW - 1) * dilationW + 1) < width ? (wstart + (kernelW - 1) * dilationW + 1) : width;
        while (hstart < 0)
            hstart += dilationH;
        while (wstart < 0)
            wstart += dilationW;
        float maxval = *reinterpret_cast<const float*>(&F32_NEG_INF); // -Infinity
        idx_accscalar_t maxidx = hstart * width + wstart;
        auto btmData = bottomData + nxc * height * width;
        for (idx_accscalar_t h = hstart; h < hend; h += dilationH) {
            for (idx_accscalar_t w = wstart; w < wend; w += dilationW) {
                idx_accscalar_t idxOffset = h * width + w;
                float val = static_cast<float>(btmData[idxOffset]);
                SimtProc::CycleUpdate<idx_accscalar_t>(val, idxOffset, &maxval, &maxidx);
            }
        }
        topData[index] = static_cast<scalar_t>(maxval);
        topMask[index] = static_cast<idx_scalar_t>(maxidx);
    }
}

template <typename scalar_t, typename idx_scalar_t, typename idx_accscalar_t>
__simt_vf__ __aicore__ LAUNCH_BOUND(SimtProc::THREAD_DIM) inline void MaxPoolForwardNhwc(
    const int64_t count, const __gm__ scalar_t* bottomData, const int64_t channels, const int64_t height,
    const int64_t width, const int outputHeight, const int outputWidth, const int kernelH, const int kernelW,
    const int strideH, const int strideW, const int padH, const int padW, const int dilationH, const int dilationW,
    __gm__ scalar_t* topData, __gm__ idx_scalar_t* topMask, int blockIdx, int blockNum)
{
    for (idx_accscalar_t index = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index = index + blockNum * Simt::GetThreadNum()) {
        idx_accscalar_t c = index % channels;
        idx_accscalar_t pw = (index / channels) % outputWidth;
        idx_accscalar_t ph = (index / channels / outputWidth) % outputHeight;
        idx_accscalar_t n = index / channels / outputWidth / outputHeight;
        idx_accscalar_t hstart = ph * strideH - padH;
        idx_accscalar_t wstart = pw * strideW - padW;
        idx_accscalar_t hend =
            (hstart + (kernelH - 1) * dilationH + 1) < height ? (hstart + (kernelH - 1) * dilationH + 1) : height;
        idx_accscalar_t wend =
            (wstart + (kernelW - 1) * dilationW + 1) < width ? (wstart + (kernelW - 1) * dilationW + 1) : width;
        while (hstart < 0)
            hstart += dilationH;
        while (wstart < 0)
            wstart += dilationW;
        float maxval = *reinterpret_cast<const float*>(&F32_NEG_INF);
        idx_accscalar_t maxidx = hstart * width + wstart;
        auto btmData = bottomData + (n * height * width * channels);
        for (idx_accscalar_t h = hstart; h < hend; h += dilationH) {
            for (idx_accscalar_t w = wstart; w < wend; w += dilationW) {
                idx_accscalar_t idxOffset = h * width + w;
                scalar_t val = static_cast<float>(btmData[idxOffset * channels + c]);
                SimtProc::CycleUpdate<idx_accscalar_t>(val, idxOffset, &maxval, &maxidx);
            }
        }
        topData[index] = static_cast<scalar_t>(maxval);
        topMask[index] = static_cast<idx_scalar_t>(maxidx);
    }
}

template <typename VALUE_T, typename INDICES_T, int Format_T, bool useINT64Index>
__aicore__ inline void MaxPoolWithArgmaxV3<VALUE_T, INDICES_T, Format_T, useINT64Index>::Compute() const
{
    const int kH = tilingData_->kSizeH;
    const int kW = tilingData_->kSizeW;

    const int dH = tilingData_->stridesH;
    const int dW = tilingData_->stridesW;

    const int padH = tilingData_->padH;
    const int padW = tilingData_->padW;

    const int dilationH = tilingData_->dilationH;
    const int dilationW = tilingData_->dilationW;
    const int64_t nbatch = tilingData_->nDim;
    const int64_t inputChannel = tilingData_->cDim;
    const int64_t inputHeight = tilingData_->hInDim;
    const int64_t inputWidth = tilingData_->wInDim;

    const int64_t outputHeight = tilingData_->hOutDim;
    const int64_t outputWidth = tilingData_->wOutDim;

    auto inputData = (__gm__ VALUE_T*)x_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)argmax_.GetPhyAddr();
    int64_t count = nbatch * inputChannel * outputHeight * outputWidth;
    if constexpr (Format_T == 0 && !useINT64Index) {
        Simt::VF_CALL<MaxPoolForwardNchw<VALUE_T, INDICES_T, int32_t>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, inputHeight, inputWidth, outputHeight, outputWidth, kH,
            kW, dH, dW, padH, padW, dilationH, dilationW, outputData, indicesData, blockIdx_, blockNum_);
    } else if constexpr (Format_T == 1 && !useINT64Index) {
        Simt::VF_CALL<MaxPoolForwardNhwc<VALUE_T, INDICES_T, int32_t>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, inputChannel, inputHeight, inputWidth, outputHeight,
            outputWidth, kH, kW, dH, dW, padH, padW, dilationH, dilationW, outputData, indicesData, blockIdx_,
            blockNum_);
    } else if constexpr (Format_T == 0 && useINT64Index) {
        Simt::VF_CALL<MaxPoolForwardNchw<VALUE_T, INDICES_T, int64_t>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, inputHeight, inputWidth, outputHeight, outputWidth, kH,
            kW, dH, dW, padH, padW, dilationH, dilationW, outputData, indicesData, blockIdx_, blockNum_);
    } else if constexpr (Format_T == 1 && useINT64Index) {
        Simt::VF_CALL<MaxPoolForwardNhwc<VALUE_T, INDICES_T, int64_t>>(
            Simt::Dim3(SimtProc::THREAD_DIM), count, inputData, inputChannel, inputHeight, inputWidth, outputHeight,
            outputWidth, kH, kW, dH, dW, padH, padW, dilationH, dilationW, outputData, indicesData, blockIdx_,
            blockNum_);
    }
}

#endif // CANN_MAX_POOL_WITH_ARGMAX_V3_SIMT_H
