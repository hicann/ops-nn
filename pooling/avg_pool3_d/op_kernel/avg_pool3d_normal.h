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
 * \file avg_pool3_d_normal.h
 * \brief
 */

 #ifndef AVG_POOL3D_NORMAL_H_
 #define AVG_POOL3D_NORMAL_H_

 #include "kernel_operator.h"
 #include "kernel_tiling/kernel_tiling.h"
 #include "avg_pool3d_common.h"
 #include "../pool_3d_common/arch32/pool_3d_memory_optimized_utils.h"

namespace AvgPool3d {
template <typename T>
class AvgPool3dNormal {
public:
    __aicore__ inline AvgPool3dNormal(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe);
    __aicore__ inline void InitTiling(const AvgPool3DTilingData* __restrict__ tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInX(int64_t curNcFactor, int64_t xGmOffset);
    __aicore__ inline void CalcIndex(int64_t index, int64_t baseW, int64_t baseH);
    __aicore__ inline void TransInput(
        int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor);
    __aicore__ inline void AvgPoolW(
        const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor, int64_t curWoFactor);
    __aicore__ inline void AvgPoolH(
        const uint8_t diFactor, const uint8_t hiFactor, const int64_t curWoFactor, int64_t curHoFactor);
    __aicore__ inline void AvgPoolD(
        const uint8_t diFactor, const uint8_t hiFactor, const int64_t curWoFactor, int64_t curHoFactor);
    __aicore__ inline void TransOut(int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor);
    __aicore__ inline void CopyOut(
        int64_t curNcFactor, int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor, int64_t yGmOffset);

    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TBuf<> inputTransBuffer;
    TBuf<> mulWBuffer;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    Pool3dMemCommon::PoolVars poolVars;
    int64_t kernelsize = 0;
    int64_t kW = 0;
    int64_t kH = 0;
    int64_t kD = 0;
    int64_t dW = 0;
    int64_t dH = 0;
    int64_t dD = 0;
    int64_t padD = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t divisorOverride = 0;
    bool countIncludePad = false;
    int64_t dStart = 0;
    int64_t hStart = 0;
    int64_t wStart = 0;
    int64_t dEnd = 0;
    int64_t hEnd = 0;
    int64_t wEnd = 0;
    uint8_t diFactor = 0;
    uint8_t hiFactor = 0;
    uint8_t wiFactor = 0;
    float mulsFactor = 0;
    bool isSamePoolSize = false;
};

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::InitTiling(const AvgPool3DTilingData* __restrict__ tiling) {
    poolVars.useCoreNum = tiling->useCoreNum;
    poolVars.N = tiling->inN;
    poolVars.C = tiling->inC;
    poolVars.Di = tiling->inD;
    poolVars.Hi = tiling->inH;
    poolVars.Wi = tiling->inW;
    poolVars.Do = tiling->outD;
    poolVars.Ho = tiling->outH;
    poolVars.Wo = tiling->outW;
    Pool3dMemCommon::InitPoolVars(poolVars, tiling);
    kernelsize = tiling->kD * tiling->kH * tiling->kW;
    kW = tiling->kW;
    kD = tiling->kD;
    kH = tiling->kH;
    dW = tiling->dW;
    dH = tiling->dH;
    dD = tiling->dD;
    padW = tiling->pW;
    padH = tiling->pH;
    padD = tiling->pD;
    divisorOverride = tiling->divisorOverride;
    countIncludePad = tiling->countIncludePad;
    isSamePoolSize = divisorOverride ||
                     ((countIncludePad || (padW ==0 && padH == 0 && padD == 0)) && !tiling->ceilMode);
}


template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe) {
    InitTiling(tiling);

    poolVars.cBlockIdx = GetBlockIdx();
    if (poolVars.cBlockIdx >= poolVars.useCoreNum) {
        return;
    }

    poolVars.DiHiWi = poolVars.Di * poolVars.Hi * poolVars.Wi;
    poolVars.HiWi = poolVars.Hi * poolVars.Wi;
    int64_t calBlockNum = (poolVars.cBlockIdx == poolVars.useCoreNum - 1) ? poolVars.blockTail : poolVars.blockFactor;
    poolVars.beginIdx = poolVars.cBlockIdx * poolVars.blockFactor;
    poolVars.endIdx = poolVars.cBlockIdx * poolVars.blockFactor + calBlockNum;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    // 初始化que
    pipe->InitBuffer(inputQue, 1, 32 * 1024);  // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(T)
    pipe->InitBuffer(yQue, 1, 8 * 1024);     // VL_NUM*doFactor*hoFactor*woFactorAlign*sizeof(T)
    pipe->InitBuffer(inputTransBuffer, 64 * 1024);  // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(float)
    pipe->InitBuffer(mulWBuffer, 64 * 1024);        // VL_NUM*diFactor*hiFactor*wiFactor16Align*sizeof(float)
}

/*
* 功能：input类型转换 <T> -> <fp32>类型, 并转置，把[VL, D, H, W] 转为[D, H, W, VL]
*/
template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::TransInput(
    int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor) {
    Pool3dMemCommon::TransposeInput<T, 1>(inputQue, inputTransBuffer, mulWBuffer, curNcFactor, diFactor, hiFactor, wiFactor, poolVars);
}

/*
* 功能：<fp32>类型out和<int32>类型index的转置，把[D, H, W, VL]转为[VL, D, H, W]
* 同时会cast out为<T>类型
*/
template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::TransOut(int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor) {
    auto curWoFactorAlign = Ceil(curWoFactor, 8) * 8;
    auto curWoFactorAlign16 = Ceil(curWoFactor, 32 / sizeof(T)) * 32 / sizeof(T);

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> yLocal = yQue.AllocTensor<T>();
    LocalTensor<float> mulDUb = mulWBuffer.Get<float>();
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        Pool3dMemCommon::OutTranspose<T>(yLocal, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, poolVars.VL_NUM);
    } else {
        if (curWoFactorAlign == curWoFactorAlign16) {
            Pool3dMemCommon::OutTranspose<T>(mulHUb, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, poolVars.VL_NUM);
        } else {
            UnaryRepeatParams repeatCastParams2{(uint16_t)(curWoFactorAlign16 / 8), (uint16_t)(curWoFactorAlign / 8),
                                                (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign16, 16) * 2),
                                                (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 2)};
            Pool3dMemCommon::OutTranspose<T>(mulHUb[4096], mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, poolVars.VL_NUM);
            AscendC::PipeBarrier<PIPE_V>();

            Adds(mulHUb, mulHUb[4096], (float)0.0, (uint8_t)(curWoFactorAlign * curDoFactor * curHoFactor), poolVars.VL_NUM,
                 repeatCastParams2);
        }
        AscendC::PipeBarrier<PIPE_V>();

        Cast(yLocal, mulHUb, RoundMode::CAST_ROUND, poolVars.VL_NUM * curWoFactorAlign16 * curDoFactor * curHoFactor);
    }
    yQue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::CopyOut(
    int64_t curNcFactor, int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor, int64_t yGmOffset) {
    auto curWoFactorAlign16 = Ceil(curWoFactor, 32 / sizeof(T)) * 32 / sizeof(T);
    LocalTensor<T> yLocal = yQue.DeQue<T>();

    DataCopyExtParams paramsOut2 = {
        static_cast<uint16_t>(curHoFactor), static_cast<uint32_t>(curWoFactor * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((poolVars.Wo - curWoFactor) * sizeof(T)), static_cast<uint32_t>(0)};
    for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
        for (int64_t dCopyi = 0; dCopyi < curDoFactor; dCopyi++) {
            auto dstAddr = yGmOffset + ncCopyi * poolVars.Do * poolVars.Ho * poolVars.Wo + dCopyi * poolVars.Ho * poolVars.Wo;
            auto srcAddr = ncCopyi * Ceil(curDoFactor * curHoFactor * curWoFactorAlign16, 16) * 16 +
                           dCopyi * curHoFactor * curWoFactorAlign16;
            DataCopyPad(yGm[dstAddr], yLocal[srcAddr], paramsOut2);
        }
    }

    yQue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::AvgPoolW(
    const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor, int64_t curWoFactor) {
    LocalTensor<float> xLocalTransVL = inputTransBuffer.Get<float>();

    const uint8_t wiFactorAlign = Ceil(wiFactor, 8) * 8;
    uint64_t mask = 256 / sizeof(float);
    auto repeat = hiFactor * diFactor;
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(poolVars.VL_NUM / 8 * wiFactorAlign)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, (uint8_t)(8 * wiFactorAlign)};
    LocalTensor<float> mulWUb = mulWBuffer.Get<float>();

    for (int kernelIdx = 0; kernelIdx < curWoFactor; kernelIdx++) {
        int32_t kerWEndIdx = Min(poolVars.Wi, wEnd + kernelIdx * dW);
        int32_t kerWStartIdx = Max(wStart + kernelIdx * dW, kerWEndIdx - kW);
        if (wStart == 0) {
            kerWStartIdx = Max(wStart, wEnd + kernelIdx * dW - kW);
        }
        if(kernelIdx == curWoFactor - 1) {
            if(wStart == 0) {
                kerWStartIdx = Max(wEnd + kernelIdx * dW - kW, 0);
            } else {
                kerWStartIdx = Min(wStart + kernelIdx * dW, poolVars.Wi);
            }
            if(curWoFactor == 1) {
                kerWEndIdx = wEnd;
                kerWStartIdx = wStart;
            }
        }
        auto mulWOffset = kernelIdx * diFactor * hiFactor * poolVars.VL_NUM;
        auto inputOffset = poolVars.VL_NUM * (kerWStartIdx - wStart);
        Adds(mulWUb[mulWOffset], xLocalTransVL[inputOffset], (float)0.0, poolVars.VL_NUM, repeat, repeatCopyParams);
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = kerWStartIdx + 1; i < kerWEndIdx; i++) {
            auto nexAddOffset = poolVars.VL_NUM * (i - wStart);
            Add(mulWUb[mulWOffset], mulWUb[mulWOffset], xLocalTransVL[nexAddOffset], mask, repeat, repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::AvgPoolH(
    const uint8_t diFactor, const uint8_t hiFactor, int64_t curWoFactor, int64_t curHoFactor) {
    auto woFactorAlign = Ceil(curWoFactor, 8) * 8;

    uint64_t mask = 256 / sizeof(float);
    auto repeat = woFactorAlign * diFactor;
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(poolVars.VL_NUM / 8 * hiFactor)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, (uint8_t)(8 * hiFactor)};
    LocalTensor<float> mulWUb = mulWBuffer.Get<float>();
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();

    for (int kernelIdx = 0; kernelIdx < curHoFactor; kernelIdx++) {
        int32_t kerHEndIdx = Min(poolVars.Hi, hEnd + kernelIdx * dH);
        int32_t kerHStartIdx = Max(hStart + kernelIdx * dH, kerHEndIdx - kH);
        if (hStart == 0) {
            kerHStartIdx = Max(hStart, hEnd + kernelIdx * dH -kH);
        }
        if(kernelIdx == curHoFactor - 1) {
            if(hStart == 0) {
                kerHStartIdx = Max(hEnd + kernelIdx * dH - kH, 0);
            } else {
                kerHStartIdx = Min(hStart + kernelIdx * dH, poolVars.Hi);
            }
            if(curHoFactor == 1) {
                kerHEndIdx = hEnd;
                kerHStartIdx = hStart;
            }
        }
        auto mulHOffset = kernelIdx * repeat * poolVars.VL_NUM;
        auto mulWOffset = poolVars.VL_NUM * (kerHStartIdx - hStart);
        Adds(mulHUb[mulHOffset], mulWUb[mulWOffset], (float)0.0, poolVars.VL_NUM, repeat, repeatCopyParams);
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = kerHStartIdx + 1; i < kerHEndIdx; i++) {
            auto nexAddOffset = poolVars.VL_NUM * (i - hStart);
            Add(mulHUb[mulHOffset], mulHUb[mulHOffset], mulWUb[nexAddOffset], mask, repeat, repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::AvgPoolD(
    const uint8_t diFactor, const uint8_t hiFactor, int64_t curWoFactor, int64_t curHoFactor) {
    auto woFactorAlign = Ceil(curWoFactor, 8) * 8;

    uint64_t mask = 256 / sizeof(float);
    auto repeat = curHoFactor * woFactorAlign;
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(poolVars.VL_NUM / 8 * diFactor)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, (uint8_t)(8 * diFactor)};
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
    LocalTensor<float> mulDUb = mulWBuffer.Get<float>();
    int32_t kerDStartIdx = dStart;
    int32_t kerDEndIdx = dEnd;
    auto mulDOffset = 0;
    auto mulHOffset = 0;
    Adds(mulDUb[mulDOffset], mulHUb[mulHOffset], (float)0.0, poolVars.VL_NUM, repeat, repeatCopyParams);
    AscendC::PipeBarrier<PIPE_V>();
    for (int i = 1; i < kerDEndIdx - kerDStartIdx; i++) {
        auto nexAddOffset = poolVars.VL_NUM * i;
        Add(mulDUb[mulDOffset], mulDUb[mulDOffset], mulHUb[nexAddOffset], mask, repeat, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if(isSamePoolSize){
        Muls(mulDUb, mulDUb, mulsFactor, 16384);
    }
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::CalcIndex(int64_t index,int64_t baseW, int64_t baseH) {
    auto indexD = (index / (poolVars.Ho * poolVars.Wo)) % poolVars.Do;
    auto indexH = (index / poolVars.Wo) % poolVars.Ho;
    auto indexW = index % poolVars.Wo;
    dStart = indexD * dD -padD;
    hStart = indexH * dH -padH;
    wStart = indexW * dW -padW;
    dEnd = Min(dStart + kD, poolVars.Di + padD);
    hEnd = Min(hStart + kH, poolVars.Hi + padH);
    wEnd = Min(wStart + kW, poolVars.Wi + padW);
    auto poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
    dStart = Max(dStart , 0);
    hStart = Max(hStart , 0);
    wStart = Max(wStart , 0);
    dEnd = Min(dEnd, poolVars.Di);
    hEnd = Min(hEnd, poolVars.Hi);
    wEnd = Min(wEnd, poolVars.Wi);
    kernelsize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
    diFactor = dEnd - dStart;
    hiFactor = Min(poolVars.Hi, ((hEnd - hStart) + (baseH - 1) * dH));
    wiFactor = Min(poolVars.Wi, ((wEnd - wStart) + (baseW - 1) * dW));
    if (divisorOverride) {
        mulsFactor = (float)1.0 / static_cast<float>(divisorOverride);
    } else if (countIncludePad) {
        mulsFactor = (float)1.0 / static_cast<float>(poolSize);
    } else {
        mulsFactor = (float)1.0 / static_cast<float>(kernelsize);
    }
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::CopyInX(int64_t curNcFactor,int64_t xGmOffset) {
    Pool3dMemCommon::CopyInputData<T, 1>(inputQue, xGm, curNcFactor, diFactor, hiFactor, wiFactor, xGmOffset, poolVars);
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::Process() {
    if (poolVars.cBlockIdx >= poolVars.useCoreNum) {
      return;
    }
    for (auto curIdx = poolVars.beginIdx; curIdx < poolVars.endIdx; curIdx++) {
        auto blockVar = Pool3dMemCommon::CalcBlockVar(curIdx, poolVars);
        auto kernelWMaxAlign = (kD * kH <= 8) ? 16 : 8;
        auto baseW = ((kernelWMaxAlign - kW) / dW) + 1;
        auto baseH = ((128 / kD / kernelWMaxAlign) - kH) / dH + 1;
        auto baseD = 1;
        auto ncCoreIdx = blockVar.curNcIdx * poolVars.ncFactor;
        auto doCoreIdx = blockVar.curDoIdx * poolVars.doFactor;
        auto hoCoreIdx = blockVar.curHoIdx * poolVars.hoFactor;
        auto woCoreIdx = blockVar.curWoIdx * poolVars.woFactor;
        auto incoreDCnt = CeilDiv(blockVar.curDoFactor, baseD);
        auto incoreHCnt = CeilDiv(blockVar.curHoFactor, baseH);
        auto incoreWCnt = CeilDiv(blockVar.curWoFactor, baseW);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
          for (auto doLoop = 0; doLoop < incoreDCnt; doLoop++) {
            auto doBlockIdx = doCoreIdx + doLoop;
            for (auto hoLoop = 0; hoLoop < incoreHCnt; hoLoop++) {
                auto nowH = baseH;
                auto hoBlockIdx = hoCoreIdx + hoLoop * nowH;
                nowH = (hoLoop == incoreHCnt - 1) ? (blockVar.curHoFactor - hoLoop * baseH) : baseH;
                auto nowW = baseW;
                for(auto woLoop = 0; woLoop < incoreWCnt; woLoop++) {
                    auto woBlockIdx = woCoreIdx + woLoop * nowW;
                    auto BlockIdx = doBlockIdx * poolVars.Ho * poolVars.Wo + hoBlockIdx * poolVars.Wo +woBlockIdx;
                    auto yGmOffset = ncCoreIdx * poolVars.Do * poolVars.Ho * poolVars.Wo + BlockIdx;
                    nowW = (woLoop == incoreWCnt - 1) ? (blockVar.curWoFactor - woLoop * baseW) : baseW;
                    CalcIndex(yGmOffset, nowW, nowH);
                    auto xGmOffset = ncCoreIdx * poolVars.DiHiWi + dStart * poolVars.HiWi + hStart * poolVars.Wi + wStart;
                    CopyInX(blockVar.curNcFactor, xGmOffset);
                    TransInput(blockVar.curNcFactor, diFactor, hiFactor, wiFactor);
                    AvgPoolW(diFactor, hiFactor, wiFactor, nowW);
                    AvgPoolH(diFactor, hiFactor, nowW, nowH);
                    AvgPoolD(diFactor, hiFactor, nowW, nowH);
                    TransOut(baseD, nowH, nowW);
                    CopyOut(blockVar.curNcFactor, baseD, nowH, nowW, yGmOffset);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                }
            }
        }
    }
}

} // namespace AvgPool3d
#endif  // AVG_POOL3D_NORMAL_H_