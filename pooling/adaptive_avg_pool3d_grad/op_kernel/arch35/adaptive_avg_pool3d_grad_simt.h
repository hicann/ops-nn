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
 * \file adaptive_avg_pool3d_grad_simt.h
 * \brief adaptive_avg_pool3d_grad implied by simt
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_SIMT_H
#define ADAPTIVE_AVG_POOL3D_GRAD_SIMT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_avg_pool3d_grad_struct.h"

using namespace AscendC;

namespace AdaptiveAvgPool3dGradOp {
constexpr static uint32_t SIMT_PARAMS_NUM = 16;
constexpr static uint32_t MAGIC_C_IDX = 0;
constexpr static uint32_t MAGIC_IN_D_IDX = 2;
constexpr static uint32_t MAGIC_IN_H_IDX = 4;
constexpr static uint32_t MAGIC_IN_W_IDX = 6;
constexpr static uint32_t MAGIC_OSIZE_D_IDX = 8;
constexpr static uint32_t MAGIC_OSIZE_H_IDX = 10;
constexpr static uint32_t MAGIC_OSIZE_W_IDX = 12;

constexpr static uint32_t AXIS_D = 0;
constexpr static uint32_t AXIS_H = 1;
constexpr static uint32_t AXIS_W = 2;

constexpr static int64_t CHANNEL_LAST_YES = 1;

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST, uint32_t THREADS>
class AdaptiveAvgPool3dGradSimt {
public:
    __aicore__ inline AdaptiveAvgPool3dGradSimt(TPipe* pipe,
                                                const AdaptiveAvgPool3dGradTilingDataV35* __restrict__ tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR xGrad);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    AscendC::GlobalTensor<VALUE_T> yGrad_;
    AscendC::GlobalTensor<VALUE_T> xGrad_;
    const AdaptiveAvgPool3dGradTilingDataV35* tilingData_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename OFFSET_T>
using SimtDivT = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T FloorDivMul(DIV_T numerator, DIV_T mulFactor, DIV_T divisorMagic,
                                                           DIV_T divisorShift)
{
    DIV_T wideNumerator = numerator * mulFactor;
    DIV_T quotient = Simt::UintDiv<DIV_T>(wideNumerator, divisorMagic, divisorShift);
    return quotient;
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T CeilDivMul(DIV_T numerator, DIV_T mulFactor, DIV_T ceilAddend,
                                                          DIV_T divisorMagic, DIV_T divisorShift)
{
    DIV_T wideNumerator = numerator * mulFactor + ceilAddend;
    DIV_T quotient = Simt::UintDiv<DIV_T>(wideNumerator, divisorMagic, divisorShift);
    return quotient;
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T StartIndexIn2Out(DIV_T inIdx, DIV_T osize, DIV_T magicIsize,
                                                                DIV_T shiftIsize)
{
    return FloorDivMul<DIV_T>(inIdx, osize, magicIsize, shiftIsize);
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T EndIndexIn2Out(DIV_T inIdx, DIV_T isize, DIV_T osize, DIV_T magicIsize,
                                                              DIV_T shiftIsize)
{
    return CeilDivMul<DIV_T>(inIdx + 1, osize, isize - 1, magicIsize, shiftIsize);
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T StartIndexOut2In(DIV_T outIdx, DIV_T isize, DIV_T magicOsize,
                                                                DIV_T shiftOsize)
{
    return FloorDivMul<DIV_T>(outIdx, isize, magicOsize, shiftOsize);
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T EndIndexOut2In(DIV_T outIdx, DIV_T osize, DIV_T isize, DIV_T magicOsize,
                                                              DIV_T shiftOsize)
{
    return CeilDivMul<DIV_T>(outIdx + 1, isize, osize - 1, magicOsize, shiftOsize);
}

template <typename VALUE_T, typename OFFSET_T, uint32_t THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void AdaptiveAvgPool3dGradNcdhw(
    __ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY, const SimtDivT<OFFSET_T> nDims,
    const SimtDivT<OFFSET_T> cDims, const SimtDivT<OFFSET_T> inD, const SimtDivT<OFFSET_T> inH,
    const SimtDivT<OFFSET_T> inW, const SimtDivT<OFFSET_T> outD, const SimtDivT<OFFSET_T> outH,
    const SimtDivT<OFFSET_T> outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = SimtDivT<OFFSET_T>;
    using INDEX_T = uint64_t;

    DIV_T magicInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX]);
    DIV_T shiftInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX + 1]);
    DIV_T magicInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX]);
    DIV_T shiftInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX + 1]);
    DIV_T magicInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX]);
    DIV_T shiftInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX + 1]);

    DIV_T magicOsizeD = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX]);
    DIV_T shiftOsizeD = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX + 1]);
    DIV_T magicOsizeH = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX]);
    DIV_T shiftOsizeH = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX + 1]);
    DIV_T magicOsizeW = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX]);
    DIV_T shiftOsizeW = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX + 1]);

    const DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inD) *
                        static_cast<DIV_T>(inH) * static_cast<DIV_T>(inW);
    const DIV_T outDHW = static_cast<DIV_T>(outD) * static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);
    const DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);

    const DIV_T threadStart = static_cast<DIV_T>(blockIdx.x) * static_cast<DIV_T>(blockDim.x) +
                              static_cast<DIV_T>(threadIdx.x);
    const DIV_T threadStride = static_cast<DIV_T>(gridDim.x) * static_cast<DIV_T>(blockDim.x);

    for (DIV_T index = threadStart; index < count; index += threadStride) {
        const DIV_T t1 = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
        const DIV_T w = index - t1 * static_cast<DIV_T>(inW);
        const DIV_T t2 = Simt::UintDiv<DIV_T>(t1, magicInH, shiftInH);
        const DIV_T h = t1 - t2 * static_cast<DIV_T>(inH);
        const DIV_T nc = Simt::UintDiv<DIV_T>(t2, magicInD, shiftInD);
        const DIV_T d = t2 - nc * static_cast<DIV_T>(inD);

        DIV_T odStarts = StartIndexIn2Out<DIV_T>(d, outD, magicInD, shiftInD);
        DIV_T odEnds = EndIndexIn2Out<DIV_T>(d, inD, outD, magicInD, shiftInD);
        DIV_T ohStarts = StartIndexIn2Out<DIV_T>(h, outH, magicInH, shiftInH);
        DIV_T ohEnds = EndIndexIn2Out<DIV_T>(h, inH, outH, magicInH, shiftInH);
        DIV_T owStarts = StartIndexIn2Out<DIV_T>(w, outW, magicInW, shiftInW);
        DIV_T owEnds = EndIndexIn2Out<DIV_T>(w, inW, outW, magicInW, shiftInW);

        float gradient = 0.0f;
        const INDEX_T ncBase = static_cast<INDEX_T>(nc) * static_cast<INDEX_T>(outDHW);
        for (DIV_T od = odStarts; od < odEnds; ++od) {
            DIV_T id0 = StartIndexOut2In<DIV_T>(od, inD, magicOsizeD, shiftOsizeD);
            DIV_T id1 = EndIndexOut2In<DIV_T>(od, outD, inD, magicOsizeD, shiftOsizeD);
            DIV_T kD = id1 - id0;
            for (DIV_T oh = ohStarts; oh < ohEnds; ++oh) {
                DIV_T ih0 = StartIndexOut2In<DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
                DIV_T ih1 = EndIndexOut2In<DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
                DIV_T kH = ih1 - ih0;
                const INDEX_T ohBase = ncBase + static_cast<INDEX_T>(od) * static_cast<INDEX_T>(outHW) +
                                       static_cast<INDEX_T>(oh) * static_cast<INDEX_T>(outW);
                for (DIV_T ow = owStarts; ow < owEnds; ++ow) {
                    DIV_T iw0 = StartIndexOut2In<DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
                    DIV_T iw1 = EndIndexOut2In<DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
                    DIV_T kW = iw1 - iw0;
                    DIV_T div = kD * kH * kW;

                    gradient += static_cast<float>(gradY[ohBase + static_cast<INDEX_T>(ow)]) / static_cast<float>(div);
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T, uint32_t THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void AdaptiveAvgPool3dGradNdhwc(
    __ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY, const SimtDivT<OFFSET_T> nDims,
    const SimtDivT<OFFSET_T> cDims, const SimtDivT<OFFSET_T> inD, const SimtDivT<OFFSET_T> inH,
    const SimtDivT<OFFSET_T> inW, const SimtDivT<OFFSET_T> outD, const SimtDivT<OFFSET_T> outH,
    const SimtDivT<OFFSET_T> outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = SimtDivT<OFFSET_T>;
    using INDEX_T = uint64_t;

    DIV_T magicC = static_cast<DIV_T>(simtParams[MAGIC_C_IDX]);
    DIV_T shiftC = static_cast<DIV_T>(simtParams[MAGIC_C_IDX + 1]);
    DIV_T magicInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX]);
    DIV_T shiftInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX + 1]);
    DIV_T magicInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX]);
    DIV_T shiftInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX + 1]);
    DIV_T magicInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX]);
    DIV_T shiftInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX + 1]);

    DIV_T magicOsizeD = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX]);
    DIV_T shiftOsizeD = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX + 1]);
    DIV_T magicOsizeH = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX]);
    DIV_T shiftOsizeH = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX + 1]);
    DIV_T magicOsizeW = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX]);
    DIV_T shiftOsizeW = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX + 1]);

    const DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(inD) * static_cast<DIV_T>(inH) *
                        static_cast<DIV_T>(inW) * static_cast<DIV_T>(cDims);
    const DIV_T outDHWc = static_cast<DIV_T>(outD) * static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW) *
                          static_cast<DIV_T>(cDims);
    const DIV_T outHWc = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW) * static_cast<DIV_T>(cDims);
    const DIV_T outWc = static_cast<DIV_T>(outW) * static_cast<DIV_T>(cDims);

    const DIV_T threadStart = static_cast<DIV_T>(blockIdx.x) * static_cast<DIV_T>(blockDim.x) +
                              static_cast<DIV_T>(threadIdx.x);
    const DIV_T threadStride = static_cast<DIV_T>(gridDim.x) * static_cast<DIV_T>(blockDim.x);

    for (DIV_T index = threadStart; index < count; index += threadStride) {
        const DIV_T t1 = Simt::UintDiv<DIV_T>(index, magicC, shiftC);
        const DIV_T c = index - t1 * static_cast<DIV_T>(cDims);
        const DIV_T t2 = Simt::UintDiv<DIV_T>(t1, magicInW, shiftInW);
        const DIV_T w = t1 - t2 * static_cast<DIV_T>(inW);
        const DIV_T t3 = Simt::UintDiv<DIV_T>(t2, magicInH, shiftInH);
        const DIV_T h = t2 - t3 * static_cast<DIV_T>(inH);
        const DIV_T n = Simt::UintDiv<DIV_T>(t3, magicInD, shiftInD);
        const DIV_T d = t3 - n * static_cast<DIV_T>(inD);

        DIV_T odStarts = StartIndexIn2Out<DIV_T>(d, outD, magicInD, shiftInD);
        DIV_T odEnds = EndIndexIn2Out<DIV_T>(d, inD, outD, magicInD, shiftInD);
        DIV_T ohStarts = StartIndexIn2Out<DIV_T>(h, outH, magicInH, shiftInH);
        DIV_T ohEnds = EndIndexIn2Out<DIV_T>(h, inH, outH, magicInH, shiftInH);
        DIV_T owStarts = StartIndexIn2Out<DIV_T>(w, outW, magicInW, shiftInW);
        DIV_T owEnds = EndIndexIn2Out<DIV_T>(w, inW, outW, magicInW, shiftInW);

        float gradient = 0.0f;
        const INDEX_T nBase = static_cast<INDEX_T>(n) * static_cast<INDEX_T>(outDHWc) + static_cast<INDEX_T>(c);
        for (DIV_T od = odStarts; od < odEnds; ++od) {
            DIV_T id0 = StartIndexOut2In<DIV_T>(od, inD, magicOsizeD, shiftOsizeD);
            DIV_T id1 = EndIndexOut2In<DIV_T>(od, outD, inD, magicOsizeD, shiftOsizeD);
            DIV_T kD = id1 - id0;
            for (DIV_T oh = ohStarts; oh < ohEnds; ++oh) {
                DIV_T ih0 = StartIndexOut2In<DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
                DIV_T ih1 = EndIndexOut2In<DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
                DIV_T kH = ih1 - ih0;
                const INDEX_T ohBase = nBase + static_cast<INDEX_T>(od) * static_cast<INDEX_T>(outHWc) +
                                       static_cast<INDEX_T>(oh) * static_cast<INDEX_T>(outWc);
                for (DIV_T ow = owStarts; ow < owEnds; ++ow) {
                    DIV_T iw0 = StartIndexOut2In<DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
                    DIV_T iw1 = EndIndexOut2In<DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
                    DIV_T kW = iw1 - iw0;
                    DIV_T div = kD * kH * kW;

                    gradient += static_cast<float>(
                                    gradY[ohBase + static_cast<INDEX_T>(ow) * static_cast<INDEX_T>(cDims)]) /
                                static_cast<float>(div);
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

// 仅一个输出轴 > 1 时的特化 kernel。
// AXIS：0 = D 变化，1 = H 变化，2 = W 变化。
// 另两个退化轴只有一个输出点，其感受野跨度可每线程算一次而非每次内层迭代重算。
template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST, uint32_t AXIS, uint32_t THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void AdaptiveAvgPool3dGradSingleAxis(
    __ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY, const SimtDivT<OFFSET_T> nDims,
    const SimtDivT<OFFSET_T> cDims, const SimtDivT<OFFSET_T> inD, const SimtDivT<OFFSET_T> inH,
    const SimtDivT<OFFSET_T> inW, const SimtDivT<OFFSET_T> outD, const SimtDivT<OFFSET_T> outH,
    const SimtDivT<OFFSET_T> outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = SimtDivT<OFFSET_T>;
    using INDEX_T = uint64_t;

    DIV_T magicC = static_cast<DIV_T>(simtParams[MAGIC_C_IDX]);
    DIV_T shiftC = static_cast<DIV_T>(simtParams[MAGIC_C_IDX + 1]);
    DIV_T magicInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX]);
    DIV_T shiftInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX + 1]);
    DIV_T magicInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX]);
    DIV_T shiftInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX + 1]);
    DIV_T magicInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX]);
    DIV_T shiftInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX + 1]);

    DIV_T magicOsize = 0;
    DIV_T shiftOsize = 0;
    if constexpr (AXIS == AXIS_D) {
        magicOsize = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX]);
        shiftOsize = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX + 1]);
    } else if constexpr (AXIS == AXIS_H) {
        magicOsize = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX]);
        shiftOsize = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX + 1]);
    } else if constexpr (AXIS == AXIS_W) {
        magicOsize = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX]);
        shiftOsize = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX + 1]);
    }

    const DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inD) *
                        static_cast<DIV_T>(inH) * static_cast<DIV_T>(inW);

    const DIV_T threadStart = static_cast<DIV_T>(blockIdx.x) * static_cast<DIV_T>(blockDim.x) +
                              static_cast<DIV_T>(threadIdx.x);
    const DIV_T threadStride = static_cast<DIV_T>(gridDim.x) * static_cast<DIV_T>(blockDim.x);

    // 两个退化轴各只有一个输出点、覆盖整条输入轴，故感受野大小即输入轴长。
    const DIV_T degSize = (AXIS == AXIS_D) ? (inH * inW) : ((AXIS == AXIS_H) ? (inD * inW) : (inD * inH));
    const DIV_T outAxis = (AXIS == AXIS_D) ? outD : ((AXIS == AXIS_H) ? outH : outW);
    const DIV_T outStride = (CHANNEL_LAST == CHANNEL_LAST_YES) ? static_cast<DIV_T>(cDims) : static_cast<DIV_T>(1);
    const DIV_T outBatch = outAxis * outStride;

    for (DIV_T index = threadStart; index < count; index += threadStride) {
        DIV_T d = 0;
        DIV_T h = 0;
        DIV_T w = 0;
        INDEX_T base = 0;
        if constexpr (CHANNEL_LAST == CHANNEL_LAST_YES) {
            const DIV_T t1 = Simt::UintDiv<DIV_T>(index, magicC, shiftC);
            const DIV_T c = index - t1 * static_cast<DIV_T>(cDims);
            const DIV_T t2 = Simt::UintDiv<DIV_T>(t1, magicInW, shiftInW);
            w = t1 - t2 * static_cast<DIV_T>(inW);
            const DIV_T t3 = Simt::UintDiv<DIV_T>(t2, magicInH, shiftInH);
            h = t2 - t3 * static_cast<DIV_T>(inH);
            const DIV_T n = Simt::UintDiv<DIV_T>(t3, magicInD, shiftInD);
            d = t3 - n * static_cast<DIV_T>(inD);
            base = static_cast<INDEX_T>(n) * static_cast<INDEX_T>(outBatch) + static_cast<INDEX_T>(c);
        } else {
            const DIV_T t1 = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
            w = index - t1 * static_cast<DIV_T>(inW);
            const DIV_T t2 = Simt::UintDiv<DIV_T>(t1, magicInH, shiftInH);
            h = t1 - t2 * static_cast<DIV_T>(inH);
            const DIV_T nc = Simt::UintDiv<DIV_T>(t2, magicInD, shiftInD);
            d = t2 - nc * static_cast<DIV_T>(inD);
            base = static_cast<INDEX_T>(nc) * static_cast<INDEX_T>(outBatch);
        }

        DIV_T idx0 = 0;
        DIV_T idx1 = 0;
        DIV_T inAxis = 0;
        if constexpr (AXIS == AXIS_D) {
            idx0 = StartIndexIn2Out<DIV_T>(d, outD, magicInD, shiftInD);
            idx1 = EndIndexIn2Out<DIV_T>(d, inD, outD, magicInD, shiftInD);
            inAxis = inD;
        } else if constexpr (AXIS == AXIS_H) {
            idx0 = StartIndexIn2Out<DIV_T>(h, outH, magicInH, shiftInH);
            idx1 = EndIndexIn2Out<DIV_T>(h, inH, outH, magicInH, shiftInH);
            inAxis = inH;
        } else if constexpr (AXIS == AXIS_W) {
            idx0 = StartIndexIn2Out<DIV_T>(w, outW, magicInW, shiftInW);
            idx1 = EndIndexIn2Out<DIV_T>(w, inW, outW, magicInW, shiftInW);
            inAxis = inW;
        }

        float gradient = 0.0f;
        for (DIV_T oi = idx0; oi < idx1; ++oi) {
            const DIV_T i0 = StartIndexOut2In<DIV_T>(oi, inAxis, magicOsize, shiftOsize);
            const DIV_T i1 = EndIndexOut2In<DIV_T>(oi, outAxis, inAxis, magicOsize, shiftOsize);
            const INDEX_T outIdx = base + static_cast<INDEX_T>(oi) * static_cast<INDEX_T>(outStride);
            gradient += static_cast<float>(gradY[outIdx]) / static_cast<float>(degSize * (i1 - i0));
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

// “一个归约轴 + 两个直通轴”的特化 kernel（仅 NCDHW）。
// AXIS：0 = D 归约，1 = H 归约，2 = W 归约。
// 两个直通轴的窗口大小折进除数 degK；归约轴保留逐项窗口，且除法留在循环内、
template <typename VALUE_T, typename OFFSET_T, uint32_t AXIS, uint32_t THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void AdaptiveAvgPool3dGradSingleReduce(
    __ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY, const SimtDivT<OFFSET_T> nDims,
    const SimtDivT<OFFSET_T> cDims, const SimtDivT<OFFSET_T> inD, const SimtDivT<OFFSET_T> inH,
    const SimtDivT<OFFSET_T> inW, const SimtDivT<OFFSET_T> outD, const SimtDivT<OFFSET_T> outH,
    const SimtDivT<OFFSET_T> outW, __gm__ VALUE_T* gradX)
{
    using DIV_T = SimtDivT<OFFSET_T>;
    using INDEX_T = uint64_t;

    DIV_T magicInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX]);
    DIV_T shiftInD = static_cast<DIV_T>(simtParams[MAGIC_IN_D_IDX + 1]);
    DIV_T magicInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX]);
    DIV_T shiftInH = static_cast<DIV_T>(simtParams[MAGIC_IN_H_IDX + 1]);
    DIV_T magicInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX]);
    DIV_T shiftInW = static_cast<DIV_T>(simtParams[MAGIC_IN_W_IDX + 1]);

    DIV_T magicOsizeD = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX]);
    DIV_T shiftOsizeD = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_D_IDX + 1]);
    DIV_T magicOsizeH = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX]);
    DIV_T shiftOsizeH = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_H_IDX + 1]);
    DIV_T magicOsizeW = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX]);
    DIV_T shiftOsizeW = static_cast<DIV_T>(simtParams[MAGIC_OSIZE_W_IDX + 1]);

    const DIV_T count = static_cast<DIV_T>(nDims) * static_cast<DIV_T>(cDims) * static_cast<DIV_T>(inD) *
                        static_cast<DIV_T>(inH) * static_cast<DIV_T>(inW);
    const DIV_T outDHW = static_cast<DIV_T>(outD) * static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);
    const DIV_T outHW = static_cast<DIV_T>(outH) * static_cast<DIV_T>(outW);

    // 归约轴的长度，以及其相邻输出点之间的间距。
    const DIV_T inAxis = (AXIS == AXIS_D) ? inD : ((AXIS == AXIS_H) ? inH : inW);
    const DIV_T outAxis = (AXIS == AXIS_D) ? outD : ((AXIS == AXIS_H) ? outH : outW);
    const DIV_T outStride = (AXIS == AXIS_D) ? outHW : ((AXIS == AXIS_H) ? outW : static_cast<DIV_T>(1));
    const DIV_T magicOsize = (AXIS == AXIS_D) ? magicOsizeD : ((AXIS == AXIS_H) ? magicOsizeH : magicOsizeW);
    const DIV_T shiftOsize = (AXIS == AXIS_D) ? shiftOsizeD : ((AXIS == AXIS_H) ? shiftOsizeH : shiftOsizeW);
    const DIV_T magicInAxis = (AXIS == AXIS_D) ? magicInD : ((AXIS == AXIS_H) ? magicInH : magicInW);
    const DIV_T shiftInAxis = (AXIS == AXIS_D) ? shiftInD : ((AXIS == AXIS_H) ? shiftInH : shiftInW);

    const DIV_T threadStart = static_cast<DIV_T>(blockIdx.x) * static_cast<DIV_T>(blockDim.x) +
                              static_cast<DIV_T>(threadIdx.x);
    const DIV_T threadStride = static_cast<DIV_T>(gridDim.x) * static_cast<DIV_T>(blockDim.x);

    for (DIV_T index = threadStart; index < count; index += threadStride) {
        const DIV_T t1 = Simt::UintDiv<DIV_T>(index, magicInW, shiftInW);
        const DIV_T w = index - t1 * static_cast<DIV_T>(inW);
        const DIV_T t2 = Simt::UintDiv<DIV_T>(t1, magicInH, shiftInH);
        const DIV_T h = t1 - t2 * static_cast<DIV_T>(inH);
        const DIV_T nc = Simt::UintDiv<DIV_T>(t2, magicInD, shiftInD);
        const DIV_T d = t2 - nc * static_cast<DIV_T>(inD);

        // 每个直通轴只对应一个输出点，其窗口大小成为除数中的常量因子。
        DIV_T odFix = 0;
        DIV_T ohFix = 0;
        DIV_T owFix = 0;
        DIV_T degK = 1;
        if constexpr (AXIS != AXIS_D) {
            odFix = StartIndexIn2Out<DIV_T>(d, outD, magicInD, shiftInD);
            const DIV_T id0 = StartIndexOut2In<DIV_T>(odFix, inD, magicOsizeD, shiftOsizeD);
            const DIV_T id1 = EndIndexOut2In<DIV_T>(odFix, outD, inD, magicOsizeD, shiftOsizeD);
            degK *= (id1 - id0);
        }
        if constexpr (AXIS != AXIS_H) {
            ohFix = StartIndexIn2Out<DIV_T>(h, outH, magicInH, shiftInH);
            const DIV_T ih0 = StartIndexOut2In<DIV_T>(ohFix, inH, magicOsizeH, shiftOsizeH);
            const DIV_T ih1 = EndIndexOut2In<DIV_T>(ohFix, outH, inH, magicOsizeH, shiftOsizeH);
            degK *= (ih1 - ih0);
        }
        if constexpr (AXIS != AXIS_W) {
            owFix = StartIndexIn2Out<DIV_T>(w, outW, magicInW, shiftInW);
            const DIV_T iw0 = StartIndexOut2In<DIV_T>(owFix, inW, magicOsizeW, shiftOsizeW);
            const DIV_T iw1 = EndIndexOut2In<DIV_T>(owFix, outW, inW, magicOsizeW, shiftOsizeW);
            degK *= (iw1 - iw0);
        }

        // 两个直通轴解析完后的输出基址。
        const INDEX_T base = static_cast<INDEX_T>(nc) * static_cast<INDEX_T>(outDHW) +
                             static_cast<INDEX_T>(odFix) * static_cast<INDEX_T>(outHW) +
                             static_cast<INDEX_T>(ohFix) * static_cast<INDEX_T>(outW) + static_cast<INDEX_T>(owFix);

        const DIV_T iAxis = (AXIS == AXIS_D) ? d : ((AXIS == AXIS_H) ? h : w);
        const DIV_T oStart = StartIndexIn2Out<DIV_T>(iAxis, outAxis, magicInAxis, shiftInAxis);
        const DIV_T oEnd = EndIndexIn2Out<DIV_T>(iAxis, inAxis, outAxis, magicInAxis, shiftInAxis);

        float gradient = 0.0f;
        for (DIV_T oi = oStart; oi < oEnd; ++oi) {
            const DIV_T i0 = StartIndexOut2In<DIV_T>(oi, inAxis, magicOsize, shiftOsize);
            const DIV_T i1 = EndIndexOut2In<DIV_T>(oi, outAxis, inAxis, magicOsize, shiftOsize);
            const INDEX_T outIdx = base + static_cast<INDEX_T>(oi) * static_cast<INDEX_T>(outStride);
            gradient += static_cast<float>(gradY[outIdx]) / static_cast<float>(degK * (i1 - i0));
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

// 通用 kernel 对每个输入点要三层循环遍历 D/H/W 上所有相关的输出点。
// 某些 shape 下部分轴只会命中一个输出点，对应的循环可以省掉，故做特化。
template <int64_t CHANNEL_LAST, typename VALUE_T, typename OFFSET_T, uint32_t THREADS>
__aicore__ inline void LaunchAdaptiveAvgPool3dGradSimtKernel(__ubuf__ OFFSET_T* simtParams, const __gm__ VALUE_T* gradY,
                                                             SimtDivT<OFFSET_T> nDims, SimtDivT<OFFSET_T> cDims,
                                                             SimtDivT<OFFSET_T> inD, SimtDivT<OFFSET_T> inH,
                                                             SimtDivT<OFFSET_T> inW, SimtDivT<OFFSET_T> outD,
                                                             SimtDivT<OFFSET_T> outH, SimtDivT<OFFSET_T> outW,
                                                             __gm__ VALUE_T* gradX)
{
    if (outH == 1 && outW == 1) {
        // 输出轴长度为 1 时该轴只有一个输出点，两层循环退化，只剩 D 轴要遍历。
        asc_vf_call<AdaptiveAvgPool3dGradSingleAxis<VALUE_T, OFFSET_T, CHANNEL_LAST, AXIS_D, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else if (outD == 1 && outW == 1) {
        asc_vf_call<AdaptiveAvgPool3dGradSingleAxis<VALUE_T, OFFSET_T, CHANNEL_LAST, AXIS_H, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else if (outD == 1 && outH == 1) {
        asc_vf_call<AdaptiveAvgPool3dGradSingleAxis<VALUE_T, OFFSET_T, CHANNEL_LAST, AXIS_W, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else if constexpr (CHANNEL_LAST == CHANNEL_LAST_YES) {
        asc_vf_call<AdaptiveAvgPool3dGradNdhwc<VALUE_T, OFFSET_T, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else if (outD > inD && outH <= inH && outW <= inW && inH % outH == 0 && inW % outW == 0) {
        // 只有 D 轴上采样需要归约；H、W 不放大且整除，每个输入点在这两轴上
        // 恰好落进一个输出窗口，故两层循环退化。整除不可放宽为 out <= in：
        // 如 inH=4、outH=3 时窗口为 [0,2)、[1,3)、[2,4)，输入点 1 跨了两个窗口。
        asc_vf_call<AdaptiveAvgPool3dGradSingleReduce<VALUE_T, OFFSET_T, AXIS_D, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else if (outH > inH && outD <= inD && outW <= inW && inD % outD == 0 && inW % outW == 0) {
        asc_vf_call<AdaptiveAvgPool3dGradSingleReduce<VALUE_T, OFFSET_T, AXIS_H, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else if (outW > inW && outD <= inD && outH <= inH && inD % outD == 0 && inH % outH == 0) {
        asc_vf_call<AdaptiveAvgPool3dGradSingleReduce<VALUE_T, OFFSET_T, AXIS_W, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else {
        asc_vf_call<AdaptiveAvgPool3dGradNcdhw<VALUE_T, OFFSET_T, THREADS>>(
            dim3(THREADS), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    }
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST, uint32_t THREADS>
__aicore__ inline void AdaptiveAvgPool3dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST, THREADS>::Init(GM_ADDR yGrad,
                                                                                                 GM_ADDR xGrad)
{
    yGrad_.SetGlobalBuffer((__gm__ VALUE_T*)(yGrad));
    xGrad_.SetGlobalBuffer((__gm__ VALUE_T*)(xGrad));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(OFFSET_T));
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST, uint32_t THREADS>
__aicore__ inline void AdaptiveAvgPool3dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST, THREADS>::Process()
{
    using DIV_T = SimtDivT<OFFSET_T>;

    LocalTensor<OFFSET_T> simtParam = paramBuf_.Get<OFFSET_T>();
    const int64_t* tilingPtr = reinterpret_cast<const int64_t*>(tilingData_);

    const DIV_T nDims = static_cast<DIV_T>(tilingPtr[0]);
    const DIV_T cDims = static_cast<DIV_T>(tilingPtr[1]);
    const DIV_T inD = static_cast<DIV_T>(tilingPtr[2]);
    const DIV_T inH = static_cast<DIV_T>(tilingPtr[3]);
    const DIV_T inW = static_cast<DIV_T>(tilingPtr[4]);
    const DIV_T outD = static_cast<DIV_T>(tilingPtr[5]);
    const DIV_T outH = static_cast<DIV_T>(tilingPtr[6]);
    const DIV_T outW = static_cast<DIV_T>(tilingPtr[7]);

    DIV_T magicC = 0;
    DIV_T shiftC = 0;
    DIV_T magicInD = 0;
    DIV_T shiftInD = 0;
    DIV_T magicInH = 0;
    DIV_T shiftInH = 0;
    DIV_T magicInW = 0;
    DIV_T shiftInW = 0;
    DIV_T magicOsizeD = 0;
    DIV_T shiftOsizeD = 0;
    DIV_T magicOsizeH = 0;
    DIV_T shiftOsizeH = 0;
    DIV_T magicOsizeW = 0;
    DIV_T shiftOsizeW = 0;

    GetUintDivMagicAndShift<DIV_T>(magicC, shiftC, cDims);
    GetUintDivMagicAndShift<DIV_T>(magicInD, shiftInD, inD);
    GetUintDivMagicAndShift<DIV_T>(magicInH, shiftInH, inH);
    GetUintDivMagicAndShift<DIV_T>(magicInW, shiftInW, inW);
    GetUintDivMagicAndShift<DIV_T>(magicOsizeD, shiftOsizeD, outD);
    GetUintDivMagicAndShift<DIV_T>(magicOsizeH, shiftOsizeH, outH);
    GetUintDivMagicAndShift<DIV_T>(magicOsizeW, shiftOsizeW, outW);

    simtParam.SetValue(MAGIC_C_IDX, static_cast<OFFSET_T>(magicC));
    simtParam.SetValue(MAGIC_C_IDX + 1, static_cast<OFFSET_T>(shiftC));

    simtParam.SetValue(MAGIC_IN_D_IDX, static_cast<OFFSET_T>(magicInD));
    simtParam.SetValue(MAGIC_IN_D_IDX + 1, static_cast<OFFSET_T>(shiftInD));
    simtParam.SetValue(MAGIC_IN_H_IDX, static_cast<OFFSET_T>(magicInH));
    simtParam.SetValue(MAGIC_IN_H_IDX + 1, static_cast<OFFSET_T>(shiftInH));
    simtParam.SetValue(MAGIC_IN_W_IDX, static_cast<OFFSET_T>(magicInW));
    simtParam.SetValue(MAGIC_IN_W_IDX + 1, static_cast<OFFSET_T>(shiftInW));

    simtParam.SetValue(MAGIC_OSIZE_D_IDX, static_cast<OFFSET_T>(magicOsizeD));
    simtParam.SetValue(MAGIC_OSIZE_D_IDX + 1, static_cast<OFFSET_T>(shiftOsizeD));
    simtParam.SetValue(MAGIC_OSIZE_H_IDX, static_cast<OFFSET_T>(magicOsizeH));
    simtParam.SetValue(MAGIC_OSIZE_H_IDX + 1, static_cast<OFFSET_T>(shiftOsizeH));
    simtParam.SetValue(MAGIC_OSIZE_W_IDX, static_cast<OFFSET_T>(magicOsizeW));
    simtParam.SetValue(MAGIC_OSIZE_W_IDX + 1, static_cast<OFFSET_T>(shiftOsizeW));

    DataSyncBarrier<MemDsbT::UB>();

    auto gradData = (__gm__ VALUE_T*)yGrad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)xGrad_.GetPhyAddr();

    LaunchAdaptiveAvgPool3dGradSimtKernel<CHANNEL_LAST, VALUE_T, OFFSET_T, THREADS>(
        (__ubuf__ OFFSET_T*)simtParam.GetPhyAddr(), gradData, nDims, cDims, inD, inH, inW, outD, outH, outW,
        outputData);
}

} // namespace AdaptiveAvgPool3dGradOp

#endif // ADAPTIVE_AVG_POOL3D_GRAD_SIMT_H
