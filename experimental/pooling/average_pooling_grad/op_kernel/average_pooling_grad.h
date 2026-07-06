/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file average_pooling_grad.h
 * \brief Generic AveragePoolingGrad kernel for avg_pool2d backward.
 */
#ifndef AVERAGE_POOLING_GRAD_H
#define AVERAGE_POOLING_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "average_pooling_grad_tiling_data.h"
#include "average_pooling_grad_utils.h"

namespace NsAveragePoolingGrad {

using namespace AscendC;

template<typename T>
class AveragePoolingGrad {
public:
    __aicore__ inline AveragePoolingGrad() {}
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR gradInput, const AveragePoolingGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int32_t PStart(int32_t inputIndex, uint32_t pad, uint32_t kernel, uint32_t stride) const;
    __aicore__ inline int32_t PEnd(int32_t inputIndex, uint32_t pad, uint32_t stride) const;
    __aicore__ inline int32_t CalcDivisor(int32_t oh, int32_t ow) const;
    __aicore__ inline float CalcGradInput(uint32_t nIdx, uint32_t cIdx, uint32_t ih, uint32_t iw) const;

    GlobalTensor<T> gradOutputGm;
    GlobalTensor<T> gradInputGm;

    uint32_t n;
    uint32_t c;
    uint32_t hIn;
    uint32_t wIn;
    uint32_t hOut;
    uint32_t wOut;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t padTop;
    uint32_t padLeft;
    uint32_t countIncludePad;
    int32_t divisorOverride;
    uint32_t totalElements;
    uint32_t normalCoreProcessNum;
    uint32_t tailCoreProcessNum;
    uint32_t usedCoreNum;
};

template<typename T>
__aicore__ inline void AveragePoolingGrad<T>::Init(
    GM_ADDR gradOutput, GM_ADDR gradInput, const AveragePoolingGradTilingData* tilingData)
{
    n = tilingData->n;
    c = tilingData->c;
    hIn = tilingData->hIn;
    wIn = tilingData->wIn;
    hOut = tilingData->hOut;
    wOut = tilingData->wOut;
    kernelH = tilingData->kernelH;
    kernelW = tilingData->kernelW;
    strideH = tilingData->strideH;
    strideW = tilingData->strideW;
    padTop = tilingData->padTop;
    padLeft = tilingData->padLeft;
    countIncludePad = tilingData->countIncludePad;
    divisorOverride = tilingData->divisorOverride;
    totalElements = tilingData->totalElements;
    normalCoreProcessNum = tilingData->normalCoreProcessNum;
    tailCoreProcessNum = tilingData->tailCoreProcessNum;
    usedCoreNum = tilingData->usedCoreNum;

    gradOutputGm.SetGlobalBuffer((__gm__ T*)gradOutput, n * c * hOut * wOut);
    gradInputGm.SetGlobalBuffer((__gm__ T*)gradInput, totalElements);
}



template<typename T>
__aicore__ inline int32_t AveragePoolingGrad<T>::PStart(
    int32_t inputIndex, uint32_t pad, uint32_t kernel, uint32_t stride) const
{
    return FloorDiv(inputIndex + static_cast<int32_t>(pad) - static_cast<int32_t>(kernel), static_cast<int32_t>(stride)) + 1;
}

template<typename T>
__aicore__ inline int32_t AveragePoolingGrad<T>::PEnd(int32_t inputIndex, uint32_t pad, uint32_t stride) const
{
    return FloorDiv(inputIndex + static_cast<int32_t>(pad), static_cast<int32_t>(stride)) + 1;
}

template<typename T>
__aicore__ inline int32_t AveragePoolingGrad<T>::CalcDivisor(int32_t oh, int32_t ow) const
{
    if (divisorOverride != 0) {
        return divisorOverride;
    }

    const int32_t hStart = oh * static_cast<int32_t>(strideH) - static_cast<int32_t>(padTop);
    const int32_t wStart = ow * static_cast<int32_t>(strideW) - static_cast<int32_t>(padLeft);
    const int32_t validH = ScalarMin(hStart + static_cast<int32_t>(kernelH), static_cast<int32_t>(hIn)) - ScalarMax(hStart, 0);
    const int32_t validW = ScalarMin(wStart + static_cast<int32_t>(kernelW), static_cast<int32_t>(wIn)) - ScalarMax(wStart, 0);

    if (countIncludePad != 0) {
        return static_cast<int32_t>(kernelH * kernelW);
    }
    return validH * validW;
}

template<typename T>
__aicore__ inline float AveragePoolingGrad<T>::CalcGradInput(
    uint32_t nIdx, uint32_t cIdx, uint32_t ih, uint32_t iw) const
{
    int32_t ohStart = PStart(static_cast<int32_t>(ih), padTop, kernelH, strideH);
    int32_t ohEnd = PEnd(static_cast<int32_t>(ih), padTop, strideH);
    int32_t owStart = PStart(static_cast<int32_t>(iw), padLeft, kernelW, strideW);
    int32_t owEnd = PEnd(static_cast<int32_t>(iw), padLeft, strideW);

    ohStart = ScalarMax(ohStart, 0);
    owStart = ScalarMax(owStart, 0);
    ohEnd = ScalarMin(ohEnd, static_cast<int32_t>(hOut));
    owEnd = ScalarMin(owEnd, static_cast<int32_t>(wOut));

    float acc = 0.0f;
    for (int32_t oh = ohStart; oh < ohEnd; ++oh) {
        for (int32_t ow = owStart; ow < owEnd; ++ow) {
            const int32_t divisor = CalcDivisor(oh, ow);
            if (divisor <= 0) {
                continue;
            }
            const uint32_t gradOffset = ((nIdx * c + cIdx) * hOut + static_cast<uint32_t>(oh)) * wOut + static_cast<uint32_t>(ow);
            acc += static_cast<float>(gradOutputGm.GetValue(gradOffset)) / static_cast<float>(divisor);
        }
    }
    return acc;
}

template<typename T>
__aicore__ inline void AveragePoolingGrad<T>::Process()
{
    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    if (blockIdx >= usedCoreNum) {
        return;
    }

    const uint32_t start = blockIdx * normalCoreProcessNum;
    const uint32_t count = (blockIdx == usedCoreNum - 1) ? tailCoreProcessNum : normalCoreProcessNum;
    const uint32_t hw = hIn * wIn;
    const uint32_t chw = c * hw;

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t index = start + i;
        if (index >= totalElements) {
            return;
        }
        const uint32_t nIdx = index / chw;
        const uint32_t cIdx = (index - nIdx * chw) / hw;
        const uint32_t rem = index - nIdx * chw - cIdx * hw;
        const uint32_t ih = rem / wIn;
        const uint32_t iw = rem - ih * wIn;
        gradInputGm.SetValue(index, static_cast<T>(CalcGradInput(nIdx, cIdx, ih, iw)));
    }
}

} // namespace NsAveragePoolingGrad
#endif // AVERAGE_POOLING_GRAD_H
