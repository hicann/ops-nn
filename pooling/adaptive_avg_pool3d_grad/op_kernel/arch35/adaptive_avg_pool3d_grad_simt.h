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
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_avg_pool3d_grad_struct.h"

using namespace AscendC;

namespace AdaptiveAvgPool3dGradOp {
constexpr static uint32_t THREAD_DIM = 512;
constexpr static uint32_t SIMT_PARAMS_NUM = 20;
constexpr static uint32_t MAGIC_C_IDX = 0;
constexpr static uint32_t MAGIC_IN_D_IDX = 2;
constexpr static uint32_t MAGIC_IN_H_IDX = 4;
constexpr static uint32_t MAGIC_IN_W_IDX = 6;
constexpr static uint32_t MAGIC_OSIZE_D_IDX = 8;
constexpr static uint32_t MAGIC_OSIZE_H_IDX = 10;
constexpr static uint32_t MAGIC_OSIZE_W_IDX = 12;

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
class AdaptiveAvgPool3dGradSimt {
public:
    __aicore__ inline AdaptiveAvgPool3dGradSimt(
        TPipe *pipe, const AdaptiveAvgPool3dGradTilingDataV35 *__restrict__ tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR xGrad);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<VALUE_T> yGrad_;
    AscendC::GlobalTensor<VALUE_T> xGrad_;
    const AdaptiveAvgPool3dGradTilingDataV35 *tilingData_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename OFFSET_T>
using SimtDivT = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T FloorDivMul(DIV_T numerator, DIV_T mulFactor, DIV_T divisorMagic, DIV_T divisorShift)
{
    DIV_T wideNumerator = numerator * mulFactor;
    DIV_T quotient = Simt::UintDiv<DIV_T>(wideNumerator, divisorMagic, divisorShift);
    return quotient;
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T CeilDivMul(
    DIV_T numerator, DIV_T mulFactor, DIV_T ceilAddend, DIV_T divisorMagic, DIV_T divisorShift)
{
    DIV_T wideNumerator = numerator * mulFactor + ceilAddend;
    DIV_T quotient = Simt::UintDiv<DIV_T>(wideNumerator, divisorMagic, divisorShift);
    return quotient;
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T StartIndexIn2Out(DIV_T inIdx, DIV_T osize, DIV_T magicIsize, DIV_T shiftIsize)
{
    return FloorDivMul<DIV_T>(inIdx, osize, magicIsize, shiftIsize);
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T EndIndexIn2Out(
    DIV_T inIdx, DIV_T isize, DIV_T osize, DIV_T magicIsize, DIV_T shiftIsize)
{
    return CeilDivMul<DIV_T>(inIdx + 1, osize, isize - 1, magicIsize, shiftIsize);
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T StartIndexOut2In(DIV_T outIdx, DIV_T isize, DIV_T magicOsize, DIV_T shiftOsize)
{
    return FloorDivMul<DIV_T>(outIdx, isize, magicOsize, shiftOsize);
}

template <typename DIV_T>
__simt_callee__ __aicore__ inline static DIV_T EndIndexOut2In(
    DIV_T outIdx, DIV_T osize, DIV_T isize, DIV_T magicOsize, DIV_T shiftOsize)
{
    return CeilDivMul<DIV_T>(outIdx + 1, isize, osize - 1, magicOsize, shiftOsize);
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool3dGradNcdhw(
    __ubuf__ OFFSET_T* simtParams,
    const __gm__ VALUE_T* gradY,
    const SimtDivT<OFFSET_T> nDims,
    const SimtDivT<OFFSET_T> cDims,
    const SimtDivT<OFFSET_T> inD,
    const SimtDivT<OFFSET_T> inH,
    const SimtDivT<OFFSET_T> inW,
    const SimtDivT<OFFSET_T> outD,
    const SimtDivT<OFFSET_T> outH,
    const SimtDivT<OFFSET_T> outW,
    __gm__ VALUE_T* gradX)
{
    using DIV_T = SimtDivT<OFFSET_T>;

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

    DIV_T count = nDims * cDims * inD * inH * inW;
    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T temp1 = Simt::UintDiv(index, magicInW, shiftInW);
        DIV_T w = index - temp1 * inW;
        DIV_T temp2 = Simt::UintDiv(temp1, magicInH, shiftInH);
        DIV_T h = temp1 - temp2 * inH;
        DIV_T temp3 = Simt::UintDiv(temp2, magicInD, shiftInD);
        DIV_T d = temp2 - temp3 * inD;
        DIV_T n = Simt::UintDiv(temp3, magicC, shiftC);
        DIV_T c = temp3 - n * cDims;

        DIV_T odStarts = StartIndexIn2Out<DIV_T>(d, outD, magicInD, shiftInD);
        DIV_T odEnds = EndIndexIn2Out<DIV_T>(d, inD, outD, magicInD, shiftInD);
        DIV_T ohStarts = StartIndexIn2Out<DIV_T>(h, outH, magicInH, shiftInH);
        DIV_T ohEnds = EndIndexIn2Out<DIV_T>(h, inH, outH, magicInH, shiftInH);
        DIV_T owStarts = StartIndexIn2Out<DIV_T>(w, outW, magicInW, shiftInW);
        DIV_T owEnds = EndIndexIn2Out<DIV_T>(w, inW, outW, magicInW, shiftInW);

        float gradient = 0.0f;
        for (DIV_T od = odStarts; od < odEnds; ++od) {
            DIV_T id0 = StartIndexOut2In<DIV_T>(od, inD, magicOsizeD, shiftOsizeD);
            DIV_T id1 = EndIndexOut2In<DIV_T>(od, outD, inD, magicOsizeD, shiftOsizeD);
            DIV_T kD = id1 - id0;
            for (DIV_T oh = ohStarts; oh < ohEnds; ++oh) {
                DIV_T ih0 = StartIndexOut2In<DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
                DIV_T ih1 = EndIndexOut2In<DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
                DIV_T kH = ih1 - ih0;
                for (DIV_T ow = owStarts; ow < owEnds; ++ow) {
                    DIV_T iw0 = StartIndexOut2In<DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
                    DIV_T iw1 = EndIndexOut2In<DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
                    DIV_T kW = iw1 - iw0;
                    DIV_T div = kD * kH * kW;

                    DIV_T outputIdx = n * cDims * outD * outH * outW + c * outD * outH * outW +
                                      od * outH * outW + oh * outW + ow;

                    gradient += static_cast<float>(gradY[outputIdx]) / static_cast<float>(div);
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <typename VALUE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveAvgPool3dGradNdhwc(
    __ubuf__ OFFSET_T* simtParams,
    const __gm__ VALUE_T* gradY,
    const SimtDivT<OFFSET_T> nDims,
    const SimtDivT<OFFSET_T> cDims,
    const SimtDivT<OFFSET_T> inD,
    const SimtDivT<OFFSET_T> inH,
    const SimtDivT<OFFSET_T> inW,
    const SimtDivT<OFFSET_T> outD,
    const SimtDivT<OFFSET_T> outH,
    const SimtDivT<OFFSET_T> outW,
    __gm__ VALUE_T* gradX)
{
    using DIV_T = SimtDivT<OFFSET_T>;

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

    DIV_T count = nDims * inD * inH * inW * cDims;

    for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        DIV_T temp1 = Simt::UintDiv(index, magicC, shiftC);
        DIV_T c = index - temp1 * cDims;
        DIV_T temp2 = Simt::UintDiv(temp1, magicInW, shiftInW);
        DIV_T w = temp1 - temp2 * inW;
        DIV_T temp3 = Simt::UintDiv(temp2, magicInH, shiftInH);
        DIV_T h = temp2 - temp3 * inH;
        DIV_T n = Simt::UintDiv(temp3, magicInD, shiftInD);
        DIV_T d = temp3 - n * inD;

        DIV_T odStarts = StartIndexIn2Out<DIV_T>(d, outD, magicInD, shiftInD);
        DIV_T odEnds = EndIndexIn2Out<DIV_T>(d, inD, outD, magicInD, shiftInD);
        DIV_T ohStarts = StartIndexIn2Out<DIV_T>(h, outH, magicInH, shiftInH);
        DIV_T ohEnds = EndIndexIn2Out<DIV_T>(h, inH, outH, magicInH, shiftInH);
        DIV_T owStarts = StartIndexIn2Out<DIV_T>(w, outW, magicInW, shiftInW);
        DIV_T owEnds = EndIndexIn2Out<DIV_T>(w, inW, outW, magicInW, shiftInW);

        float gradient = 0.0f;
        for (DIV_T od = odStarts; od < odEnds; ++od) {
            DIV_T id0 = StartIndexOut2In<DIV_T>(od, inD, magicOsizeD, shiftOsizeD);
            DIV_T id1 = EndIndexOut2In<DIV_T>(od, outD, inD, magicOsizeD, shiftOsizeD);
            DIV_T kD = id1 - id0;
            for (DIV_T oh = ohStarts; oh < ohEnds; ++oh) {
                DIV_T ih0 = StartIndexOut2In<DIV_T>(oh, inH, magicOsizeH, shiftOsizeH);
                DIV_T ih1 = EndIndexOut2In<DIV_T>(oh, outH, inH, magicOsizeH, shiftOsizeH);
                DIV_T kH = ih1 - ih0;
                for (DIV_T ow = owStarts; ow < owEnds; ++ow) {
                    DIV_T iw0 = StartIndexOut2In<DIV_T>(ow, inW, magicOsizeW, shiftOsizeW);
                    DIV_T iw1 = EndIndexOut2In<DIV_T>(ow, outW, inW, magicOsizeW, shiftOsizeW);
                    DIV_T kW = iw1 - iw0;
                    DIV_T div = kD * kH * kW;

                    DIV_T outputIdx = n * outD * outH * outW * cDims + od * outH * outW * cDims +
                                      oh * outW * cDims + ow * cDims + c;
                    gradient += static_cast<float>(gradY[outputIdx]) / static_cast<float>(div);
                }
            }
        }
        gradX[index] = static_cast<VALUE_T>(gradient);
    }
}

template <int64_t CHANNEL_LAST, typename VALUE_T, typename OFFSET_T>
__aicore__ inline void LaunchAdaptiveAvgPool3dGradSimtKernel(
    __ubuf__ OFFSET_T* simtParams,
    const __gm__ VALUE_T* gradY,
    SimtDivT<OFFSET_T> nDims,
    SimtDivT<OFFSET_T> cDims,
    SimtDivT<OFFSET_T> inD,
    SimtDivT<OFFSET_T> inH,
    SimtDivT<OFFSET_T> inW,
    SimtDivT<OFFSET_T> outD,
    SimtDivT<OFFSET_T> outH,
    SimtDivT<OFFSET_T> outW,
    __gm__ VALUE_T* gradX)
{
    if constexpr (CHANNEL_LAST == 1) {
        Simt::VF_CALL<AdaptiveAvgPool3dGradNdhwc<VALUE_T, OFFSET_T>>(
            Simt::Dim3(THREAD_DIM), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    } else {
        Simt::VF_CALL<AdaptiveAvgPool3dGradNcdhw<VALUE_T, OFFSET_T>>(
            Simt::Dim3(THREAD_DIM), simtParams, gradY, nDims, cDims, inD, inH, inW, outD, outH, outW, gradX);
    }
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void AdaptiveAvgPool3dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST>::Init(
    GM_ADDR yGrad, GM_ADDR xGrad)
{
    yGrad_.SetGlobalBuffer((__gm__ VALUE_T *)(yGrad));
    xGrad_.SetGlobalBuffer((__gm__ VALUE_T *)(xGrad));
    pipe_->InitBuffer(paramBuf_, SIMT_PARAMS_NUM * sizeof(OFFSET_T));
}

template <typename VALUE_T, typename OFFSET_T, int64_t CHANNEL_LAST>
__aicore__ inline void AdaptiveAvgPool3dGradSimt<VALUE_T, OFFSET_T, CHANNEL_LAST>::Process()
{
    using DIV_T = SimtDivT<OFFSET_T>;

    LocalTensor<OFFSET_T> simtParam = paramBuf_.Get<OFFSET_T>();
    const int64_t *tilingPtr = reinterpret_cast<const int64_t *>(tilingData_);

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

    auto gradData = (__gm__ VALUE_T *)yGrad_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T *)xGrad_.GetPhyAddr();

    LaunchAdaptiveAvgPool3dGradSimtKernel<CHANNEL_LAST, VALUE_T, OFFSET_T>(
        (__ubuf__ OFFSET_T *)simtParam.GetPhyAddr(),
        gradData,
        nDims,
        cDims,
        inD,
        inH,
        inW,
        outD,
        outH,
        outW,
        outputData);
}

} // namespace AdaptiveAvgPool3dGradOp

#endif  // ADAPTIVE_AVG_POOL3D_GRAD_SIMT_H