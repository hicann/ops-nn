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
    __aicore__ inline void OutTranspose(
        LocalTensor<float> xLocalTrans, LocalTensor<float> xLocal, int32_t rowNum, int32_t colNum);

    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TBuf<> inputTransBuffer;
    TBuf<> mulWBuffer;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t cBlockIdx = 0;

    int64_t N = 1;
    int64_t C = 1;
    int64_t Di = 1;
    int64_t Hi = 1;
    int64_t Wi = 1;
    int64_t Do = 1;
    int64_t Ho = 1;
    int64_t Wo = 1;
    int64_t DiHiWi = 1;
    int64_t HiWi = 1;
    const int32_t VL_NUM = 64;  // Vector calculate length / float size

    // 多核切分的整尾块
    int64_t ncFactor = 0;
    int64_t doFactor = 0;
    int64_t hoFactor = 0;
    int64_t woFactor = 0;
    int64_t ncTail = 0;
    int64_t doTail = 0;
    int64_t hoTail = 0;
    int64_t woTail = 0;

    // 多核切分的数量
    int64_t ncOuter = 0;
    int64_t doOuter = 0;
    int64_t hoOuter = 0;
    int64_t woOuter = 0;

    int64_t totalIdx = 0;     // 总UB计算块
    int64_t blockFactor = 0;  // 每个核最多计算的UB块
    int64_t useCoreNum = 0;   // 使用核数
    int64_t blockTail = 0;    // 多核尾块
    int64_t beginIdx = 0;  // 当前核计算块起始id
    int64_t endIdx = 0;    // 当前核计算块终止id
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
    useCoreNum = tiling->useCoreNum;
    N = tiling->inN;
    C = tiling->inC;
    Di = tiling->inD;
    Hi = tiling->inH;
    Wi = tiling->inW;
    Do = tiling->outD;
    Ho = tiling->outH;
    Wo = tiling->outW;
    totalIdx = tiling->totalIdx;
    blockFactor = tiling->blockFactor;
    blockTail = tiling->blockTail;
    ncFactor = tiling->ncFactor;
    woFactor = tiling->woFactor;
    hoFactor = tiling->hoFactor;
    doFactor = tiling->doFactor;
    doOuter = tiling->doOuter;
    doTail = tiling->doTail;
    hoOuter = tiling->hoOuter;
    hoTail = tiling->hoTail;
    woOuter = tiling->woOuter;
    woTail = tiling->woTail;
    ncOuter = tiling->ncOuter;
    ncTail = tiling->ncTail;
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

    cBlockIdx = GetBlockIdx();
    if (cBlockIdx >= useCoreNum) {
        return;
    }

    DiHiWi = Di * Hi * Wi;
    HiWi = Hi * Wi;
    int64_t calBlockNum = blockFactor;
    if (cBlockIdx == useCoreNum - 1) {
        calBlockNum = blockTail;
    }
    beginIdx = cBlockIdx * blockFactor;
    endIdx = cBlockIdx * blockFactor + calBlockNum;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    // 初始化que
    pipe->InitBuffer(inputQue, 1, 32 * 1024);  // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(T) 有问题？
    pipe->InitBuffer(yQue, 1, 8 * 1024);     // VL_NUM*doFactor*hoFactor*woFactorAlign*sizeof(T)
    pipe->InitBuffer(inputTransBuffer, 64 * 1024);  // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(float)
    pipe->InitBuffer(mulWBuffer, 64 * 1024);        // VL_NUM*diFactor*hiFactor*wiFactor16Align*sizeof(float)
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::OutTranspose(
    LocalTensor<float> xLocalTrans, LocalTensor<float> xLocal, int32_t rowNum, int32_t colNum) {
    LocalTensor<float> dstList[16];
    LocalTensor<float> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (colNum == 8) {
        transDataParams.repeatTimes = rowNum / 16;
        transDataParams.dstRepStride = 2;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = xLocal[i * 8];
        }

        for (int32_t i = 0; i < 8; i++) {
            dstList[i * 2] = xLocalTrans[i * rowNum];
            dstList[i * 2 + 1] = xLocalTrans[i * rowNum + 8];
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<float>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else {
        transDataParams.repeatTimes = colNum / 8;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < rowNum / 16; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = xLocal[i * colNum + j * 16 * colNum];
            }

            for (int32_t i = 0; i < 8; i++) {
                dstList[i * 2] = xLocalTrans[i * rowNum + j * 16];
                dstList[i * 2 + 1] = xLocalTrans[i * rowNum + 8 + j * 16];
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<float>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}
/*
* 功能：input类型转换 <T> -> <fp32>类型, 并转置，把[VL, D, H, W] 转为[D, H, W, VL]
*/
template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::TransInput(
    int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor) {
    const uint8_t wiFactor16Align = Ceil(wiFactor, 32 / sizeof(T)) * 32 / sizeof(T);
    const uint8_t wiFactorAlign = Ceil(wiFactor, 8) * 8;
    LocalTensor<T> xLocal = inputQue.DeQue<T>();
    LocalTensor<float> xLocalTransVL = inputTransBuffer.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        OutTranspose(xLocalTransVL, xLocal, VL_NUM, diFactor * hiFactor * wiFactorAlign);
    } else {
        LocalTensor<float> xLocalCast = mulWBuffer.Get<float>();
        UnaryRepeatParams repeatCastParams{(uint16_t)(wiFactorAlign / 8), (uint16_t)(wiFactor16Align / 8),
                                           (uint8_t)(wiFactorAlign / 8 * Ceil(diFactor * hiFactor, 2)),
                                           (uint8_t)(wiFactor16Align / 8 * Ceil(diFactor * hiFactor, 2))};

        Cast(xLocalCast, xLocal, RoundMode::CAST_NONE, wiFactor16Align * curNcFactor * diFactor * hiFactor);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(xLocalCast, xLocalCast, float(0.0), uint8_t(wiFactorAlign * Ceil(diFactor * hiFactor, 2)), curNcFactor * 2,
             repeatCastParams);

        AscendC::PipeBarrier<PIPE_V>();
        OutTranspose(xLocalTransVL, xLocalCast, VL_NUM, diFactor * hiFactor * wiFactorAlign);
    }
    AscendC::PipeBarrier<PIPE_V>();
    inputQue.FreeTensor(xLocal);
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
        OutTranspose(yLocal, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, VL_NUM);
    } else {
        if (curWoFactorAlign == curWoFactorAlign16) {
            OutTranspose(mulHUb, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, VL_NUM);
        } else {
            UnaryRepeatParams repeatCastParams2{(uint16_t)(curWoFactorAlign16 / 8), (uint16_t)(curWoFactorAlign / 8),
                                                (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign16, 16) * 2),
                                                (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 2)};
            OutTranspose(mulHUb[4096], mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, VL_NUM);
            AscendC::PipeBarrier<PIPE_V>();

            Adds(mulHUb, mulHUb[4096], (float)0.0, (uint8_t)(curWoFactorAlign * curDoFactor * curHoFactor), VL_NUM,
                 repeatCastParams2);
        }
        AscendC::PipeBarrier<PIPE_V>();

        Cast(yLocal, mulHUb, RoundMode::CAST_ROUND, VL_NUM * curWoFactorAlign16 * curDoFactor * curHoFactor);
    }
    yQue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::CopyOut(
    int64_t curNcFactor, int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor, int64_t yGmOffset) {
    auto curWoFactorAlign16 = Ceil(curWoFactor, 32 / sizeof(T)) * 32 / sizeof(T);

    LocalTensor<T> yLocal = yQue.DeQue<T>();

    DataCopyExtParams paramsOut2;
    paramsOut2.blockCount = curHoFactor;
    paramsOut2.blockLen = curWoFactor * sizeof(T);
    paramsOut2.srcStride = 0;
    paramsOut2.dstStride = (Wo - curWoFactor) * sizeof(T);
    for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
        for (int64_t dCopyi = 0; dCopyi < curDoFactor; dCopyi++) {
            auto dstAddr = yGmOffset + ncCopyi * Do * Ho * Wo + dCopyi * Ho * Wo;
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
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(VL_NUM / 8 * wiFactorAlign)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, (uint8_t)(8 * wiFactorAlign)};
    LocalTensor<float> mulWUb = mulWBuffer.Get<float>();

    for (int kernelIdx = 0; kernelIdx < curWoFactor; kernelIdx++) {
        int32_t kerWEndIdx = Min(Wi, wEnd + kernelIdx * dW);
        int32_t kerWStartIdx = Max(wStart + kernelIdx * dW, kerWEndIdx - kW);
        if (wStart == 0) {
            kerWStartIdx = Max(wStart, wEnd + kernelIdx * dW - kW);
        }
        if(kernelIdx == curWoFactor - 1) {
            if(wStart == 0) {
                kerWStartIdx = Max(wEnd + kernelIdx * dW - kW, 0);
            } else {
                kerWStartIdx = Min(wStart + kernelIdx * dW, Wi);
            }
            if(curWoFactor == 1) {
                kerWEndIdx = wEnd;
                kerWStartIdx = wStart;
            }
        }
        auto mulWOffset = kernelIdx * diFactor * hiFactor * VL_NUM;
        auto inputOffset = VL_NUM * (kerWStartIdx - wStart);
        Adds(mulWUb[mulWOffset], xLocalTransVL[inputOffset], (float)0.0, VL_NUM, repeat, repeatCopyParams);
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = kerWStartIdx + 1; i < kerWEndIdx; i++) {
            auto nexAddOffset = VL_NUM * (i - wStart);
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
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(VL_NUM / 8 * hiFactor)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, (uint8_t)(8 * hiFactor)};
    LocalTensor<float> mulWUb = mulWBuffer.Get<float>();
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();

    for (int kernelIdx = 0; kernelIdx < curHoFactor; kernelIdx++) {
        int32_t kerHEndIdx = Min(Hi, hEnd + kernelIdx * dH);
        int32_t kerHStartIdx = Max(hStart + kernelIdx * dH, kerHEndIdx - kH);
        if (hStart == 0) {
            kerHStartIdx = Max(hStart, hEnd + kernelIdx * dH -kH);
        }
        if(kernelIdx == curHoFactor - 1) {
            if(hStart == 0) {
                kerHStartIdx = Max(hEnd + kernelIdx * dH - kH, 0);
            } else {
                kerHStartIdx = Min(hStart + kernelIdx * dH, Hi);
            }
            if(curHoFactor == 1) {
                kerHEndIdx = hEnd;
                kerHStartIdx = hStart;
            }
        }
        auto mulHOffset = kernelIdx * repeat * VL_NUM;
        auto mulWOffset = VL_NUM * (kerHStartIdx - hStart);
        Adds(mulHUb[mulHOffset], mulWUb[mulWOffset], (float)0.0, VL_NUM, repeat, repeatCopyParams);
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = kerHStartIdx + 1; i < kerHEndIdx; i++) {
            auto nexAddOffset = VL_NUM * (i - hStart);
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
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(VL_NUM / 8 * diFactor)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, 8, (uint8_t)(8 * diFactor)};
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
    LocalTensor<float> mulDUb = mulWBuffer.Get<float>();
    int32_t kerDStartIdx = dStart;
    int32_t kerDEndIdx = dEnd;
    auto mulDOffset = 0;
    auto mulHOffset = 0;
    Adds(mulDUb[mulDOffset], mulHUb[mulHOffset], (float)0.0, VL_NUM, repeat, repeatCopyParams);
    AscendC::PipeBarrier<PIPE_V>();
    for (int i = 1; i < kerDEndIdx - kerDStartIdx; i++) {
        auto nexAddOffset = VL_NUM * i;
        Add(mulDUb[mulDOffset], mulDUb[mulDOffset], mulHUb[nexAddOffset], mask, repeat, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if(isSamePoolSize){
        Muls(mulDUb, mulDUb, mulsFactor, 16384);
    }
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::CalcIndex(int64_t index,int64_t baseW, int64_t baseH) {
    auto indexD = (index / (Ho * Wo)) % Do;
    auto indexH = (index / Wo) % Ho;
    auto indexW = index % Wo;
    dStart = indexD * dD -padD;
    hStart = indexH * dH -padH;
    wStart = indexW * dW -padW;
    dEnd = Min(dStart + kD, Di + padD);
    hEnd = Min(hStart + kH, Hi + padH);
    wEnd = Min(wStart + kW, Wi + padW);
    auto poolSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
    dStart = Max(dStart , 0);
    hStart = Max(hStart , 0);
    wStart = Max(wStart , 0);
    dEnd = Min(dEnd, Di);
    hEnd = Min(hEnd, Hi);
    wEnd = Min(wEnd, Wi);
    kernelsize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
    diFactor = dEnd - dStart;
    hiFactor = Min(Hi, ((hEnd - hStart) + (baseH - 1) * dH));
    wiFactor = Min(Wi, ((wEnd - wStart) + (baseW - 1) * dW));
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
    LocalTensor<T> xLocal = inputQue.AllocTensor<T>();
    const uint8_t wiFactor16Align = Ceil(wiFactor, 32 / sizeof(T)) * 32 / sizeof(T);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams paramsIn;
    paramsIn.blockCount = hiFactor;
    paramsIn.blockLen = wiFactor * sizeof(T);
    paramsIn.srcStride = (Wi - wiFactor) * sizeof(T);
    paramsIn.dstStride = 0;
    for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
        for (int64_t dCopyi = 0; dCopyi < diFactor; dCopyi++) {
            auto srcAddr = xGmOffset + ncCopyi * DiHiWi + dCopyi * HiWi;
            auto dstAddr = (ncCopyi * diFactor + dCopyi) * hiFactor * wiFactor16Align;
            DataCopyPad(xLocal[dstAddr], xGm[srcAddr], paramsIn, padParams);
        }
    }
    inputQue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void AvgPool3dNormal<T>::Process() {
    if (cBlockIdx >= useCoreNum) {
      return;
    }
    for (auto curIdx = beginIdx; curIdx < endIdx; curIdx++) {
        auto curNcIdx = curIdx / (doOuter * hoOuter * woOuter);
        auto curNcFactor = curNcIdx == (ncOuter - 1) ? ncTail : ncFactor;
        auto tmpIdx = curIdx % (doOuter * hoOuter * woOuter);
        auto curDoIdx = tmpIdx / (hoOuter * woOuter);
        auto curDoFactor = curDoIdx == (doOuter - 1) ? doTail : doFactor;
        tmpIdx = tmpIdx % (hoOuter * woOuter);
        auto curHoIdx = tmpIdx / woOuter;
        auto curHoFactor = curHoIdx == (hoOuter - 1) ? hoTail : hoFactor;
        auto curWoIdx = tmpIdx % woOuter;
        auto curWoFactor = curWoIdx == (woOuter - 1) ? woTail : woFactor;
        auto kernelWMaxAlign = (kD * kH <= 8) ? 16 : 8;
        auto baseW = ((kernelWMaxAlign - kW) / dW) + 1;
        auto baseH = ((128 / kD / kernelWMaxAlign) - kH) / dH + 1;
        auto baseD = 1;
        auto ncCoreIdx = curNcIdx * ncFactor;
        auto doCoreIdx = curDoIdx * doFactor;
        auto hoCoreIdx = curHoIdx * hoFactor;
        auto woCoreIdx = curWoIdx * woFactor;
        auto incoreDCnt = CeilDiv(curDoFactor, baseD);
        auto incoreHCnt = CeilDiv(curHoFactor, baseH);
        auto incoreWCnt = CeilDiv(curWoFactor, baseW);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
          for (auto doLoop = 0; doLoop < incoreDCnt; doLoop++) {
            auto doBlockIdx = doCoreIdx + doLoop;
            for (auto hoLoop = 0; hoLoop < incoreHCnt; hoLoop++) {
                auto nowH = baseH;
                auto hoBlockIdx = hoCoreIdx + hoLoop * nowH;
                nowH = (hoLoop == incoreHCnt - 1) ? (curHoFactor - hoLoop * baseH) : baseH;
                auto nowW = baseW;
                for(auto woLoop = 0; woLoop < incoreWCnt; woLoop++) {
                    auto woBlockIdx = woCoreIdx + woLoop * nowW;
                    auto BlockIdx = doBlockIdx * Ho * Wo + hoBlockIdx * Wo +woBlockIdx;
                    auto yGmOffset = ncCoreIdx * Do * Ho * Wo + BlockIdx;
                    nowW = (woLoop == incoreWCnt - 1) ? (curWoFactor - woLoop * baseW) : baseW;
                    CalcIndex(yGmOffset, nowW, nowH);
                    auto xGmOffset = ncCoreIdx * DiHiWi + dStart * HiWi + hStart * Wi + wStart;
                    CopyInX(curNcFactor, xGmOffset);
                    TransInput(curNcFactor, diFactor, hiFactor, wiFactor);
                    AvgPoolW(diFactor, hiFactor, wiFactor, nowW);
                    AvgPoolH(diFactor, hiFactor, nowW, nowH);
                    AvgPoolD(diFactor, hiFactor, nowW, nowH);
                    TransOut(baseD, nowH, nowW);
                    CopyOut(curNcFactor, baseD, nowH, nowW, yGmOffset);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                }
            }
        }
    }
}

} // namespace AvgPool3d
#endif  // AVG_POOL3D_NORMAL_H_