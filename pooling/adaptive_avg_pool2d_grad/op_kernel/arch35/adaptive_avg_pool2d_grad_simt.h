/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool2d_grad_simt.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H
#define ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_avg_pool2d_grad_struct.h"
#include "simt_api/asc_simt.h"
using namespace AscendC;

namespace AdaptiveAvgPool2dGradOp {
constexpr static uint32_t THREAD_DIM = 1024;
constexpr static uint32_t SIMT_PARAMS_NUM = 64;
constexpr static uint32_t MAGIC_C_IDX = 0;
constexpr static uint32_t MAGIC_IN_H_IDX = 2;
constexpr static uint32_t MAGIC_IN_W_IDX = 4;
constexpr static uint32_t MAGIC_OSIZE_H_IDX = 6;
constexpr static uint32_t MAGIC_OSIZE_W_IDX = 8;
constexpr static uint32_t MAGIC_SEG_IDX = 10;
constexpr static uint32_t SEG_INFO_IDX = 12;
constexpr static uint32_t SEG_INFO_STRIDE = 4;

template <typename OFFSET_T>
using DivForOffset = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
class AdaptiveAvgPool2dGradSimt {
public:
    __aicore__ inline AdaptiveAvgPool2dGradSimt(TPipe* pipe,
                                                const AdaptiveAvgPool2dGradSimtTiling* __restrict__ tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}
    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR xGrad);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    AscendC::GlobalTensor<VALUE_T> yGrad_;
    AscendC::GlobalTensor<VALUE_T> xGrad_;
    const AdaptiveAvgPool2dGradSimtTiling* tilingData_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T FloorDivMul(OFFSET_T numerator, OFFSET_T mulFactor,
                                                              DIV_T divisorMagic, DIV_T divisorShift)
{
    return static_cast<OFFSET_T>(Simt::UintDiv<DIV_T>(static_cast<DIV_T>(numerator) * static_cast<DIV_T>(mulFactor),
                                                      divisorMagic, divisorShift));
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T CeilDivMul(OFFSET_T numerator, OFFSET_T mulFactor,
                                                             OFFSET_T ceilAddend, DIV_T divisorMagic,
                                                             DIV_T divisorShift)
{
    return static_cast<OFFSET_T>(Simt::UintDiv<DIV_T>(
        static_cast<DIV_T>(numerator) * static_cast<DIV_T>(mulFactor) + static_cast<DIV_T>(ceilAddend), divisorMagic,
        divisorShift));
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T StartIndexIn2Out(OFFSET_T inIdx, OFFSET_T osize, DIV_T magicIsize,
                                                                   DIV_T shiftIsize)
{
    return FloorDivMul<OFFSET_T, DIV_T>(inIdx, osize, magicIsize, shiftIsize);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T EndIndexIn2Out(OFFSET_T inIdx, OFFSET_T isize, OFFSET_T osize,
                                                                 DIV_T magicIsize, DIV_T shiftIsize)
{
    return CeilDivMul<OFFSET_T, DIV_T>(inIdx + 1, osize, isize - 1, magicIsize, shiftIsize);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T StartIndexOut2In(OFFSET_T outIdx, OFFSET_T isize, DIV_T magicOsize,
                                                                   DIV_T shiftOsize)
{
    return FloorDivMul<OFFSET_T, DIV_T>(outIdx, isize, magicOsize, shiftOsize);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static OFFSET_T EndIndexOut2In(OFFSET_T outIdx, OFFSET_T osize, OFFSET_T isize,
                                                                 DIV_T magicOsize, DIV_T shiftOsize)
{
    return CeilDivMul<OFFSET_T, DIV_T>(outIdx + 1, isize, osize - 1, magicOsize, shiftOsize);
}

template <typename OFFSET_T, typename DIV_T>
__simt_callee__ __aicore__ inline static void LoadSimtParams(__ubuf__ OFFSET_T* p, DIV_T& magicInH, DIV_T& shiftInH,
                                                             DIV_T& magicInW, DIV_T& shiftInW, DIV_T& magicOsizeH,
                                                             DIV_T& shiftOsizeH, DIV_T& magicOsizeW, DIV_T& shiftOsizeW)
{
    magicInH = static_cast<DIV_T>(p[MAGIC_IN_H_IDX]);
    shiftInH = static_cast<DIV_T>(p[MAGIC_IN_H_IDX + 1]);
    magicInW = static_cast<DIV_T>(p[MAGIC_IN_W_IDX]);
    shiftInW = static_cast<DIV_T>(p[MAGIC_IN_W_IDX + 1]);
    magicOsizeH = static_cast<DIV_T>(p[MAGIC_OSIZE_H_IDX]);
    shiftOsizeH = static_cast<DIV_T>(p[MAGIC_OSIZE_H_IDX + 1]);
    magicOsizeW = static_cast<DIV_T>(p[MAGIC_OSIZE_W_IDX]);
    shiftOsizeW = static_cast<DIV_T>(p[MAGIC_OSIZE_W_IDX + 1]);
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradHInOne(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inW);
    DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T nc = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW), w = index - nc * static_cast<DIV_T>(inW),
              base = nc * outHW;
        OFFSET_T owStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), outW, magicInW, shiftInW);
        OFFSET_T owEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), inW, outW, magicInW, shiftInW);
        float gradient = 0.0f;
        for (OFFSET_T ow = owStart; ow < owEnd; ++ow) {
            OFFSET_T iw0 = StartIndexOut2In<OFFSET_T, DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
            OFFSET_T iw1 = EndIndexOut2In<OFFSET_T, DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
            float invKW = 1.0f / static_cast<float>(iw1 - iw0);
            for (OFFSET_T oh = 0; oh < outH; ++oh) {
                gradient += static_cast<float>(gradY[base + static_cast<DIV_T>(oh) * static_cast<DIV_T>(outW) +
                                                     static_cast<DIV_T>(ow)]) *
                            invKW;
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T, uint32_t OUT_W>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradSmallOutWRow(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    OFFSET_T iw0[OUT_W], iw1[OUT_W];
    float invKW[OUT_W];
    for (uint32_t ow = 0; ow < OUT_W; ++ow) {
        iw0[ow] = StartIndexOut2In<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(ow), inW, magicOsizeW, shiftOsizeW);
        iw1[ow] = EndIndexOut2In<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(ow), static_cast<OFFSET_T>(OUT_W), inW,
                                                  magicOsizeW, shiftOsizeW);
        invKW[ow] = 1.0f / static_cast<float>(iw1[ow] - iw0[ow]);
    }
    DIV_T rowCount = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inH);
    DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(OUT_W);
    for (DIV_T row = blockIdx.x * blockDim.x + threadIdx.x; row < rowCount; row += gridDim.x * blockDim.x) {
        DIV_T nc = Simt::UintDiv<DIV_T>(row, magicInH, shiftInH), h = row - nc * static_cast<DIV_T>(inH),
              base = nc * outHW;
        DIV_T xBase = (nc * static_cast<DIV_T>(inH) + h) * static_cast<DIV_T>(inW);
        OFFSET_T ohStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), outH, magicInH, shiftInH);
        OFFSET_T ohEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), inH, outH, magicInH, shiftInH);
        float owSum[OUT_W];
        for (uint32_t ow = 0; ow < OUT_W; ++ow) {
            owSum[ow] = 0.0f;
        }
        for (OFFSET_T oh = ohStart; oh < ohEnd; ++oh) {
            OFFSET_T ih0 = StartIndexOut2In<OFFSET_T, DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
            OFFSET_T ih1 = EndIndexOut2In<OFFSET_T, DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
            DIV_T yBase = base + static_cast<DIV_T>(oh) * static_cast<DIV_T>(OUT_W);
            float invKH = 1.0f / static_cast<float>(ih1 - ih0);
            for (uint32_t ow = 0; ow < OUT_W; ++ow) {
                owSum[ow] += static_cast<float>(gradY[yBase + static_cast<DIV_T>(ow)]) * invKH * invKW[ow];
            }
        }
        for (OFFSET_T w = 0; w < inW; ++w) {
            float gradient = 0.0f;
            for (uint32_t ow = 0; ow < OUT_W; ++ow) {
                if (w >= iw0[ow] && w < iw1[ow]) {
                    gradient += owSum[ow];
                }
            }
            gradX[xBase + static_cast<DIV_T>(w)] = static_cast<VALUE_T>(gradient);
        }
    }
}

template <typename VALUE_T, typename OFFSET_T, uint32_t OUT_W>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradSmallOutWSegFast(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    constexpr uint32_t SEG_NUM_U32 = OUT_W * 2 - 1;
    constexpr DIV_T SEG_NUM = static_cast<DIV_T>(SEG_NUM_U32);
    DIV_T magicSeg = static_cast<DIV_T>(p[MAGIC_SEG_IDX]), shiftSeg = static_cast<DIV_T>(p[MAGIC_SEG_IDX + 1]);
    DIV_T taskCount = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inH) * SEG_NUM;
    DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(OUT_W);
    for (DIV_T task = blockIdx.x * blockDim.x + threadIdx.x; task < taskCount; task += gridDim.x * blockDim.x) {
        DIV_T row = Simt::UintDiv<DIV_T>(task, magicSeg, shiftSeg), seg = task - row * SEG_NUM;
        DIV_T nc = Simt::UintDiv<DIV_T>(row, magicInH, shiftInH), h = row - nc * static_cast<DIV_T>(inH);
        uint32_t meta = SEG_INFO_IDX + static_cast<uint32_t>(seg) * SEG_INFO_STRIDE;
        OFFSET_T startW = static_cast<OFFSET_T>(p[meta]), endW = static_cast<OFFSET_T>(p[meta + 1]);
        OFFSET_T kW0 = static_cast<OFFSET_T>(p[meta + 2]), kW1 = static_cast<OFFSET_T>(p[meta + 3]);
        if (startW >= endW) {
            continue;
        }
        if (kW0 <= static_cast<OFFSET_T>(0)) {
            continue;
        }
        uint32_t ow0 = static_cast<uint32_t>(seg >> 1);
        bool boundary = (seg & static_cast<DIV_T>(1)) != 0;
        if (boundary && kW1 <= static_cast<OFFSET_T>(0)) {
            continue;
        }
        OFFSET_T ohStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), outH, magicInH, shiftInH);
        OFFSET_T ohEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), inH, outH, magicInH, shiftInH);
        DIV_T hLeft = h * static_cast<DIV_T>(outH), hRight = (h + static_cast<DIV_T>(1)) * static_cast<DIV_T>(outH);
        bool leftCross = static_cast<DIV_T>(ohStart) * static_cast<DIV_T>(inH) < hLeft;
        bool rightCross = static_cast<DIV_T>(ohEnd) * static_cast<DIV_T>(inH) > hRight;
        bool singleOut = ohEnd <= ohStart + static_cast<OFFSET_T>(1);
        float firstWeight = (leftCross || (singleOut && rightCross)) ? 0.5f : 1.0f;
        float lastWeight = rightCross ? 0.5f : 1.0f;
        float invKW0 = 1.0f / static_cast<float>(kW0);
        float invKW1 = boundary ? ((kW0 == kW1) ? invKW0 : 1.0f / static_cast<float>(kW1)) : 0.0f;
        DIV_T base = nc * outHW;
        DIV_T firstBase = base + static_cast<DIV_T>(ohStart) * static_cast<DIV_T>(OUT_W);
        float gradient = static_cast<float>(gradY[firstBase + static_cast<DIV_T>(ow0)]) * firstWeight * invKW0;
        if (boundary) {
            gradient += static_cast<float>(gradY[firstBase + static_cast<DIV_T>(ow0 + 1)]) * firstWeight * invKW1;
        }
        for (OFFSET_T oh = ohStart + static_cast<OFFSET_T>(1); oh < ohEnd - static_cast<OFFSET_T>(1); ++oh) {
            DIV_T yBase = base + static_cast<DIV_T>(oh) * static_cast<DIV_T>(OUT_W);
            gradient += static_cast<float>(gradY[yBase + static_cast<DIV_T>(ow0)]) * invKW0;
            if (boundary) {
                gradient += static_cast<float>(gradY[yBase + static_cast<DIV_T>(ow0 + 1)]) * invKW1;
            }
        }
        if (ohEnd > ohStart + static_cast<OFFSET_T>(1)) {
            DIV_T lastBase = base + static_cast<DIV_T>(ohEnd - static_cast<OFFSET_T>(1)) * static_cast<DIV_T>(OUT_W);
            gradient += static_cast<float>(gradY[lastBase + static_cast<DIV_T>(ow0)]) * lastWeight * invKW0;
            if (boundary) {
                gradient += static_cast<float>(gradY[lastBase + static_cast<DIV_T>(ow0 + 1)]) * lastWeight * invKW1;
            }
        }
        VALUE_T outVal = static_cast<VALUE_T>(gradient);
        DIV_T xBase = (nc * static_cast<DIV_T>(inH) + h) * static_cast<DIV_T>(inW);
        for (OFFSET_T w = startW; w < endW; ++w) {
            gradX[xBase + static_cast<DIV_T>(w)] = outVal;
        }
    }
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradOutWSmall(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inH) *
                  static_cast<DIV_T>(inW);
    DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T tmp = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
        DIV_T w = index - tmp * static_cast<DIV_T>(inW), nc = Simt::UintDiv<DIV_T>(tmp, magicInH, shiftInH),
              h = tmp - nc * static_cast<DIV_T>(inH);
        DIV_T base = nc * outHW;
        OFFSET_T ohStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), outH, magicInH, shiftInH);
        OFFSET_T ohEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), inH, outH, magicInH, shiftInH);
        OFFSET_T owStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), outW, magicInW, shiftInW);
        OFFSET_T owEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), inW, outW, magicInW, shiftInW);
        float gradient = 0.0f;
        for (OFFSET_T ow = owStart; ow < owEnd; ++ow) {
            OFFSET_T iw0 = StartIndexOut2In<OFFSET_T, DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
            OFFSET_T iw1 = EndIndexOut2In<OFFSET_T, DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
            float invKW = 1.0f / static_cast<float>(iw1 - iw0);
            for (OFFSET_T oh = ohStart; oh < ohEnd; ++oh) {
                OFFSET_T ih0 = StartIndexOut2In<OFFSET_T, DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
                OFFSET_T ih1 = EndIndexOut2In<OFFSET_T, DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
                gradient += static_cast<float>(gradY[base + static_cast<DIV_T>(oh) * static_cast<DIV_T>(outW) +
                                                     static_cast<DIV_T>(ow)]) *
                            invKW / static_cast<float>(ih1 - ih0);
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T, uint32_t H_SCALE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradHExpandExactOutWSmallFast(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inH) *
                  static_cast<DIV_T>(inW);
    DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);
    constexpr DIV_T hScale = static_cast<DIV_T>(H_SCALE);
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T tmp = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
        DIV_T w = index - tmp * static_cast<DIV_T>(inW);
        DIV_T nc = Simt::UintDiv<DIV_T>(tmp, magicInH, shiftInH);
        DIV_T h = tmp - nc * static_cast<DIV_T>(inH);
        DIV_T base = nc * outHW + h * hScale * static_cast<DIV_T>(outW);
        OFFSET_T owStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), outW, magicInW, shiftInW);
        OFFSET_T owEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), inW, outW, magicInW, shiftInW);
        float gradient = 0.0f;
        for (OFFSET_T ow = owStart; ow < owEnd; ++ow) {
            OFFSET_T iw0 = StartIndexOut2In<OFFSET_T, DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
            OFFSET_T iw1 = EndIndexOut2In<OFFSET_T, DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
            DIV_T yBase = base + static_cast<DIV_T>(ow);
            float invKW = 1.0f / static_cast<float>(iw1 - iw0);
#pragma unroll
            for (uint32_t i = 0; i < H_SCALE; ++i) {
                gradient += static_cast<float>(gradY[yBase + static_cast<DIV_T>(i) * static_cast<DIV_T>(outW)]) * invKW;
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradHReduceWExpandFast(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inH) *
                  static_cast<DIV_T>(inW);
    DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T tmp = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
        DIV_T w = index - tmp * static_cast<DIV_T>(inW), nc = Simt::UintDiv<DIV_T>(tmp, magicInH, shiftInH),
              h = tmp - nc * static_cast<DIV_T>(inH);
        DIV_T base = nc * outHW;
        OFFSET_T ohStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), outH, magicInH, shiftInH);
        OFFSET_T ohEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), inH, outH, magicInH, shiftInH);
        OFFSET_T owStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), outW, magicInW, shiftInW);
        OFFSET_T owEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), inW, outW, magicInW, shiftInW);
        float gradient = 0.0f;
        for (OFFSET_T oh = ohStart; oh < ohEnd; ++oh) {
            OFFSET_T ih0 = StartIndexOut2In<OFFSET_T, DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
            OFFSET_T ih1 = EndIndexOut2In<OFFSET_T, DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
            float invKH = 1.0f / static_cast<float>(ih1 - ih0);
            DIV_T yBase = base + static_cast<DIV_T>(oh) * static_cast<DIV_T>(outW);
            if (owEnd <= owStart) {
                continue;
            }
            if (owEnd == owStart + static_cast<OFFSET_T>(1)) {
                OFFSET_T iw0 = StartIndexOut2In<OFFSET_T, DIV_T>(owStart, inW, magicOsizeW, shiftOsizeW);
                OFFSET_T iw1 = EndIndexOut2In<OFFSET_T, DIV_T>(owStart, outW, inW, magicOsizeW, shiftOsizeW);
                gradient += static_cast<float>(gradY[yBase + static_cast<DIV_T>(owStart)]) * invKH /
                            static_cast<float>(iw1 - iw0);
                continue;
            }
            OFFSET_T firstIw0 = StartIndexOut2In<OFFSET_T, DIV_T>(owStart, inW, magicOsizeW, shiftOsizeW);
            OFFSET_T firstIw1 = EndIndexOut2In<OFFSET_T, DIV_T>(owStart, outW, inW, magicOsizeW, shiftOsizeW);
            gradient += static_cast<float>(gradY[yBase + static_cast<DIV_T>(owStart)]) * invKH /
                        static_cast<float>(firstIw1 - firstIw0);
            OFFSET_T middleEnd = owEnd - static_cast<OFFSET_T>(1);
            for (OFFSET_T ow = owStart + static_cast<OFFSET_T>(1); ow < middleEnd; ++ow) {
                gradient += static_cast<float>(gradY[yBase + static_cast<DIV_T>(ow)]) * invKH;
            }
            OFFSET_T lastOw = owEnd - static_cast<OFFSET_T>(1);
            OFFSET_T lastIw0 = StartIndexOut2In<OFFSET_T, DIV_T>(lastOw, inW, magicOsizeW, shiftOsizeW);
            OFFSET_T lastIw1 = EndIndexOut2In<OFFSET_T, DIV_T>(lastOw, outW, inW, magicOsizeW, shiftOsizeW);
            gradient += static_cast<float>(gradY[yBase + static_cast<DIV_T>(lastOw)]) * invKH /
                        static_cast<float>(lastIw1 - lastIw0);
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradHExpandW2SmallFast(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    static_cast<void>(outW);

    constexpr DIV_T OUT_W = static_cast<DIV_T>(2);
    DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inH) *
                  static_cast<DIV_T>(inW);
    DIV_T outHW = static_cast<DIV_T>(outH) * OUT_W;
    DIV_T kW = (static_cast<DIV_T>(inW) + static_cast<DIV_T>(1)) >> static_cast<DIV_T>(1);
    DIV_T rightStart = static_cast<DIV_T>(inW) >> static_cast<DIV_T>(1);
    float invKW = 1.0f / static_cast<float>(kW);

    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T tmp = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
        DIV_T w = index - tmp * static_cast<DIV_T>(inW);
        DIV_T nc = Simt::UintDiv<DIV_T>(tmp, magicInH, shiftInH);
        DIV_T h = tmp - nc * static_cast<DIV_T>(inH);

        OFFSET_T ohStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), outH, magicInH, shiftInH);
        OFFSET_T ohEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), inH, outH, magicInH, shiftInH);
        DIV_T hLeft = h * static_cast<DIV_T>(outH);
        DIV_T hRight = (h + static_cast<DIV_T>(1)) * static_cast<DIV_T>(outH);
        bool leftCross = (h != static_cast<DIV_T>(0)) &&
                         (static_cast<DIV_T>(ohStart) * static_cast<DIV_T>(inH) < hLeft);
        bool rightCross = (h + static_cast<DIV_T>(1) < static_cast<DIV_T>(inH)) &&
                          (static_cast<DIV_T>(ohEnd) * static_cast<DIV_T>(inH) > hRight);
        OFFSET_T midBegin = leftCross ? (ohStart + static_cast<OFFSET_T>(1)) : ohStart;
        OFFSET_T midEnd = rightCross ? (ohEnd - static_cast<OFFSET_T>(1)) : ohEnd;

        DIV_T base = nc * outHW;
        bool useOw0 = w < kW;
        bool useOw1 = w >= rightStart;

        if (useOw0 && !useOw1) {
            float sum0 = 0.0f;
            if (leftCross) {
                sum0 += static_cast<float>(gradY[base + static_cast<DIV_T>(ohStart) * OUT_W]) * 0.5f * invKW;
            }
            OFFSET_T oh = midBegin;
            for (; oh + static_cast<OFFSET_T>(3) < midEnd; oh += static_cast<OFFSET_T>(4)) {
                DIV_T yBase = base + static_cast<DIV_T>(oh) * OUT_W;
                sum0 += static_cast<float>(gradY[yBase]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(2)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(4)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(6)]) * invKW;
            }
            for (; oh < midEnd; ++oh) {
                sum0 += static_cast<float>(gradY[base + static_cast<DIV_T>(oh) * OUT_W]) * invKW;
            }
            if (rightCross) {
                sum0 += static_cast<float>(gradY[base + static_cast<DIV_T>(ohEnd - static_cast<OFFSET_T>(1)) * OUT_W]) *
                        0.5f * invKW;
            }
            gradX[index] = static_cast<VALUE_T>(sum0);
        } else if (!useOw0 && useOw1) {
            float sum1 = 0.0f;
            if (leftCross) {
                sum1 += static_cast<float>(gradY[base + static_cast<DIV_T>(ohStart) * OUT_W + static_cast<DIV_T>(1)]) *
                        0.5f * invKW;
            }
            OFFSET_T oh = midBegin;
            for (; oh + static_cast<OFFSET_T>(3) < midEnd; oh += static_cast<OFFSET_T>(4)) {
                DIV_T yBase = base + static_cast<DIV_T>(oh) * OUT_W + static_cast<DIV_T>(1);
                sum1 += static_cast<float>(gradY[yBase]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(2)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(4)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(6)]) * invKW;
            }
            for (; oh < midEnd; ++oh) {
                sum1 += static_cast<float>(gradY[base + static_cast<DIV_T>(oh) * OUT_W + static_cast<DIV_T>(1)]) *
                        invKW;
            }
            if (rightCross) {
                sum1 += static_cast<float>(gradY[base + static_cast<DIV_T>(ohEnd - static_cast<OFFSET_T>(1)) * OUT_W +
                                                 static_cast<DIV_T>(1)]) *
                        0.5f * invKW;
            }
            gradX[index] = static_cast<VALUE_T>(sum1);
        } else {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            if (leftCross) {
                DIV_T yBase = base + static_cast<DIV_T>(ohStart) * OUT_W;
                sum0 += static_cast<float>(gradY[yBase]) * 0.5f * invKW;
                sum1 += static_cast<float>(gradY[yBase + static_cast<DIV_T>(1)]) * 0.5f * invKW;
            }
            OFFSET_T oh = midBegin;
            for (; oh + static_cast<OFFSET_T>(3) < midEnd; oh += static_cast<OFFSET_T>(4)) {
                DIV_T yBase = base + static_cast<DIV_T>(oh) * OUT_W;
                sum0 += static_cast<float>(gradY[yBase]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(2)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(4)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(6)]) * invKW;
                sum1 += static_cast<float>(gradY[yBase + static_cast<DIV_T>(1)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(3)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(5)]) * invKW +
                        static_cast<float>(gradY[yBase + static_cast<DIV_T>(7)]) * invKW;
            }
            for (; oh < midEnd; ++oh) {
                DIV_T yBase = base + static_cast<DIV_T>(oh) * OUT_W;
                sum0 += static_cast<float>(gradY[yBase]) * invKW;
                sum1 += static_cast<float>(gradY[yBase + static_cast<DIV_T>(1)]) * invKW;
            }
            if (rightCross) {
                DIV_T yBase = base + static_cast<DIV_T>(ohEnd - static_cast<OFFSET_T>(1)) * OUT_W;
                sum0 += static_cast<float>(gradY[yBase]) * 0.5f * invKW;
                sum1 += static_cast<float>(gradY[yBase + static_cast<DIV_T>(1)]) * 0.5f * invKW;
            }
            gradX[index] = static_cast<VALUE_T>(sum0 + sum1);
        }
    }
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool2dGradNchw(
    __ubuf__ OFFSET_T* p, const __gm__ VALUE_T* gradY, OFFSET_T nDims, OFFSET_T cDims, OFFSET_T inH, OFFSET_T inW,
    OFFSET_T outH, OFFSET_T outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = DivForOffset<OFFSET_T>;
    DIV_T magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0, shiftOsizeH = 0, magicOsizeW = 0,
          shiftOsizeW = 0;
    LoadSimtParams<OFFSET_T, DIV_T>(p, magicInH, shiftInH, magicInW, shiftInW, magicOsizeH, shiftOsizeH, magicOsizeW,
                                    shiftOsizeW);
    DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inH) *
                  static_cast<DIV_T>(inW);
    DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);
    for (DIV_T index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        DIV_T tmp = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
        DIV_T w = index - tmp * static_cast<DIV_T>(inW), nc = Simt::UintDiv<DIV_T>(tmp, magicInH, shiftInH),
              h = tmp - nc * static_cast<DIV_T>(inH);
        DIV_T base = nc * outHW;
        OFFSET_T ohStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), outH, magicInH, shiftInH);
        OFFSET_T ohEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(h), inH, outH, magicInH, shiftInH);
        OFFSET_T owStart = StartIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), outW, magicInW, shiftInW);
        OFFSET_T owEnd = EndIndexIn2Out<OFFSET_T, DIV_T>(static_cast<OFFSET_T>(w), inW, outW, magicInW, shiftInW);
        float gradient = 0.0f;
        for (OFFSET_T oh = ohStart; oh < ohEnd; ++oh) {
            OFFSET_T ih0 = StartIndexOut2In<OFFSET_T, DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
            OFFSET_T ih1 = EndIndexOut2In<OFFSET_T, DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
            float invKH = 1.0f / static_cast<float>(ih1 - ih0);
            for (OFFSET_T ow = owStart; ow < owEnd; ++ow) {
                OFFSET_T iw0 = StartIndexOut2In<OFFSET_T, DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
                OFFSET_T iw1 = EndIndexOut2In<OFFSET_T, DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
                gradient += static_cast<float>(gradY[base + static_cast<DIV_T>(oh) * static_cast<DIV_T>(outW) +
                                                     static_cast<DIV_T>(ow)]) *
                            invKH / static_cast<float>(iw1 - iw0);
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void AdaptiveAvgPool2dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST>::Init(GM_ADDR yGrad, GM_ADDR xGrad)
{
    yGrad_.SetGlobalBuffer((__gm__ VALUE_T*)(yGrad));
    xGrad_.SetGlobalBuffer((__gm__ VALUE_T*)(xGrad));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(OFFSET_T));
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void AdaptiveAvgPool2dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST>::Process()
{
    using DIV_T = DivForOffset<OFFSET_T>;
    OFFSET_T n = static_cast<OFFSET_T>(tilingData_->nDim), c = static_cast<OFFSET_T>(tilingData_->cDim),
             hIn = static_cast<OFFSET_T>(tilingData_->hInDim);
    OFFSET_T wIn = static_cast<OFFSET_T>(tilingData_->wInDim), hOut = static_cast<OFFSET_T>(tilingData_->hOutDim),
             wOut = static_cast<OFFSET_T>(tilingData_->wOutDim);
    LocalTensor<DIV_T> simtParam = paramBuf_.Get<DIV_T>();
    DIV_T magicC = 0, shiftC = 0, magicInH = 0, shiftInH = 0, magicInW = 0, shiftInW = 0, magicOsizeH = 0,
          shiftOsizeH = 0, magicOsizeW = 0, shiftOsizeW = 0, magicSeg = 0, shiftSeg = 0;
    DIV_T segNum = (wOut >= static_cast<OFFSET_T>(2) && wOut <= static_cast<OFFSET_T>(4)) ?
                       static_cast<DIV_T>(wOut) * static_cast<DIV_T>(2) - static_cast<DIV_T>(1) :
                       static_cast<DIV_T>(7);
    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, static_cast<DIV_T>(c));
    GetUintDivMagicAndShift<DIV_T>(magicInH, shiftInH, static_cast<DIV_T>(hIn));
    GetUintDivMagicAndShift<DIV_T>(magicInW, shiftInW, static_cast<DIV_T>(wIn));
    GetUintDivMagicAndShift<DIV_T>(magicOsizeH, shiftOsizeH, static_cast<DIV_T>(hOut));
    GetUintDivMagicAndShift<DIV_T>(magicOsizeW, shiftOsizeW, static_cast<DIV_T>(wOut));
    GetUintDivMagicAndShift<DIV_T>(magicSeg, shiftSeg, segNum);
    simtParam.SetValue(MAGIC_C_IDX, magicC);
    simtParam.SetValue(MAGIC_C_IDX + 1, shiftC);
    simtParam.SetValue(MAGIC_IN_H_IDX, magicInH);
    simtParam.SetValue(MAGIC_IN_H_IDX + 1, shiftInH);
    simtParam.SetValue(MAGIC_IN_W_IDX, magicInW);
    simtParam.SetValue(MAGIC_IN_W_IDX + 1, shiftInW);
    simtParam.SetValue(MAGIC_OSIZE_H_IDX, magicOsizeH);
    simtParam.SetValue(MAGIC_OSIZE_H_IDX + 1, shiftOsizeH);
    simtParam.SetValue(MAGIC_OSIZE_W_IDX, magicOsizeW);
    simtParam.SetValue(MAGIC_OSIZE_W_IDX + 1, shiftOsizeW);
    simtParam.SetValue(MAGIC_SEG_IDX, magicSeg);
    simtParam.SetValue(MAGIC_SEG_IDX + 1, shiftSeg);
    if (wOut >= static_cast<OFFSET_T>(2) && wOut <= static_cast<OFFSET_T>(4)) {
        for (uint32_t seg = 0; seg < 7; ++seg) {
            DIV_T startW = 0, endW = 0, kW0 = 1, kW1 = 1;
            if (static_cast<DIV_T>(seg) < segNum) {
                DIV_T ow = static_cast<DIV_T>(seg >> 1), wInDiv = static_cast<DIV_T>(wIn),
                      wOutDiv = static_cast<DIV_T>(wOut);
                DIV_T left0 = ow * wInDiv, right0 = (ow + static_cast<DIV_T>(1)) * wInDiv;
                DIV_T iw0 = left0 / wOutDiv, iw1 = (right0 + wOutDiv - static_cast<DIV_T>(1)) / wOutDiv;
                kW0 = iw1 - iw0;
                if ((seg & 1) != 0) {
                    DIV_T boundaryPos = (ow + static_cast<DIV_T>(1)) * wInDiv;
                    startW = boundaryPos / wOutDiv;
                    endW = (boundaryPos + wOutDiv - static_cast<DIV_T>(1)) / wOutDiv;
                    DIV_T ow1 = ow + static_cast<DIV_T>(1), left1 = ow1 * wInDiv,
                          right1 = (ow1 + static_cast<DIV_T>(1)) * wInDiv;
                    kW1 = (right1 + wOutDiv - static_cast<DIV_T>(1)) / wOutDiv - left1 / wOutDiv;
                } else {
                    startW = (ow == static_cast<DIV_T>(0)) ? static_cast<DIV_T>(0) :
                                                             (left0 + wOutDiv - static_cast<DIV_T>(1)) / wOutDiv;
                    endW = (ow == wOutDiv - static_cast<DIV_T>(1)) ? wInDiv : right0 / wOutDiv;
                }
            }
            uint32_t meta = SEG_INFO_IDX + seg * SEG_INFO_STRIDE;
            simtParam.SetValue(meta, startW);
            simtParam.SetValue(meta + 1, endW);
            simtParam.SetValue(meta + 2, kW0);
            simtParam.SetValue(meta + 3, kW1);
        }
    }
    DataSyncBarrier<MemDsbT::UB>();
    auto gradData = (__gm__ VALUE_T*)yGrad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)xGrad_.GetPhyAddr();
    auto params = (__ubuf__ OFFSET_T*)simtParam.GetPhyAddr();
    if (hIn == static_cast<OFFSET_T>(1)) {
        asc_vf_call<AdaptiveAvgPool2dGradHInOne<VALUE_T, OFFSET_T>>(dim3(THREAD_DIM), params, gradData, n, c, hIn, wIn,
                                                                    hOut, wOut, outputData);
    } else if (hOut > hIn && wOut == static_cast<OFFSET_T>(4) && wOut <= wIn && wIn >= static_cast<OFFSET_T>(32)) {
        asc_vf_call<AdaptiveAvgPool2dGradSmallOutWSegFast<VALUE_T, OFFSET_T, 4>>(dim3(THREAD_DIM), params, gradData, n,
                                                                                 c, hIn, wIn, hOut, wOut, outputData);
    } else if (hOut > hIn && wOut == static_cast<OFFSET_T>(3) && wOut <= wIn && wIn >= static_cast<OFFSET_T>(24)) {
        asc_vf_call<AdaptiveAvgPool2dGradSmallOutWSegFast<VALUE_T, OFFSET_T, 3>>(dim3(THREAD_DIM), params, gradData, n,
                                                                                 c, hIn, wIn, hOut, wOut, outputData);
    } else if (hOut > hIn && wOut == static_cast<OFFSET_T>(2) && wOut <= wIn && wIn >= static_cast<OFFSET_T>(32)) {
        asc_vf_call<AdaptiveAvgPool2dGradSmallOutWSegFast<VALUE_T, OFFSET_T, 2>>(dim3(THREAD_DIM), params, gradData, n,
                                                                                 c, hIn, wIn, hOut, wOut, outputData);
    } else if (hOut > hIn && wOut == static_cast<OFFSET_T>(2) && wOut <= wIn && wIn <= static_cast<OFFSET_T>(16)) {
        if (hOut >= hIn * static_cast<OFFSET_T>(1024)) {
            asc_vf_call<AdaptiveAvgPool2dGradHExpandW2SmallFast<VALUE_T, OFFSET_T>>(
                dim3(THREAD_DIM), params, gradData, n, c, hIn, wIn, hOut, wOut, outputData);
        } else {
            asc_vf_call<AdaptiveAvgPool2dGradSmallOutWRow<VALUE_T, OFFSET_T, 2>>(dim3(THREAD_DIM), params, gradData, n,
                                                                                 c, hIn, wIn, hOut, wOut, outputData);
        }
    } else if (hOut > hIn && wOut <= wIn && hIn == static_cast<OFFSET_T>(2) && hOut == static_cast<OFFSET_T>(24)) {
        asc_vf_call<AdaptiveAvgPool2dGradHExpandExactOutWSmallFast<VALUE_T, OFFSET_T, 12>>(
            dim3(THREAD_DIM), params, gradData, n, c, hIn, wIn, hOut, wOut, outputData);
    } else if (hOut > hIn && wOut <= wIn) {
        asc_vf_call<AdaptiveAvgPool2dGradOutWSmall<VALUE_T, OFFSET_T>>(dim3(THREAD_DIM), params, gradData, n, c, hIn,
                                                                       wIn, hOut, wOut, outputData);
    } else if (hIn > hOut && wOut >= wIn * static_cast<OFFSET_T>(4)) {
        asc_vf_call<AdaptiveAvgPool2dGradHReduceWExpandFast<VALUE_T, OFFSET_T>>(dim3(THREAD_DIM), params, gradData, n,
                                                                                c, hIn, wIn, hOut, wOut, outputData);
    } else {
        asc_vf_call<AdaptiveAvgPool2dGradNchw<VALUE_T, OFFSET_T>>(dim3(THREAD_DIM), params, gradData, n, c, hIn, wIn,
                                                                  hOut, wOut, outputData);
    }
}

} // namespace AdaptiveAvgPool2dGradOp
#endif // ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H