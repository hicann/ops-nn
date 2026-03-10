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
 * \file adaptive_max_pool3d_small_pool.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_SAMLL_POOL_H_
#define ADAPTIVE_MAX_POOL3D_SAMLL_POOL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../pool_3d_common/arch32/pool_3d_memory_optimized_utils.h"

using namespace AscendC;

template <typename T>
class AdaptiveMaxPool3dSmallPool
{
public:
    __aicore__ inline AdaptiveMaxPool3dSmallPool(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe,
        const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling);
    __aicore__ inline void InitTiling(const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitReset();
    __aicore__ inline void CalReset(const uint8_t diFactor, const uint8_t hiFactor);
    __aicore__ inline void CopyIn(int64_t curIdx);
    __aicore__ inline void CopyInput(
        int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor, int64_t xGmOffset);
    __aicore__ inline void TransInput(
        int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor);
    __aicore__ inline void MaxPoolW(
        const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor, int64_t curWoIdx, int64_t curWoFactor);
    __aicore__ inline void MaxPoolH(
        const uint8_t diFactor, const uint8_t hiFactor, const uint8_t curWoFactor, int64_t curHoIdx,
        int64_t curHoFactor);
    __aicore__ inline void MaxPoolD(
        const uint8_t diFactor, const uint8_t hiFactor, const uint8_t curWoFactor, int64_t curDoIdx,
        int64_t curDoFactor);
    __aicore__ inline void TransOutAndIdx(
        int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor, int64_t UbIdxOffset);
    __aicore__ inline void CopyOutAndIdx(
        int64_t curNcFactor, int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor, int64_t yGmOffset);

    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECOUT, 1> maxQue;
    TQue<QuePosition::VECOUT, 1> indexQue;
    TBuf<> inputTransBuffer;
    TBuf<> cmpMaskBuffer;
    TBuf<> cmpNanMaskBuffer;
    TBuf<> resetIndexBuf;
    TBuf<> nextCmpBuffer;
    TBuf<> mulWBuffer;
    TBuf<> mulWIdxBuffer;

    GlobalTensor<T> xGm, maxGm;
    GlobalTensor<int32_t> indicesGm;

    Pool3dMemCommon::PoolVars poolVars;

    SELMODE selMode = SELMODE::VSEL_TENSOR_TENSOR_MODE;
};

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::InitTiling(
    const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling)
{
    poolVars.useCoreNum = tiling->useCoreNum;
    poolVars.N = tiling->N;
    poolVars.C = tiling->C;
    poolVars.Di = tiling->Di;
    poolVars.Hi = tiling->Hi;
    poolVars.Wi = tiling->Wi;
    poolVars.Do = tiling->Do;
    poolVars.Ho = tiling->Ho;
    poolVars.Wo = tiling->Wo;
    Pool3dMemCommon::InitPoolVars(poolVars, tiling);
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, TPipe* pipe,
    const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tiling)
{
    InitTiling(tiling);

    poolVars.cBlockIdx = GetBlockIdx();
    if (poolVars.cBlockIdx >= poolVars.useCoreNum) {
        return;
    }

    poolVars.DiHiWi = poolVars.Di * poolVars.Hi * poolVars.Wi;
    poolVars.HiWi = poolVars.Hi * poolVars.Wi;
    int64_t calBlockNum = poolVars.blockFactor;
    if (poolVars.cBlockIdx == poolVars.useCoreNum - 1) {
        calBlockNum = poolVars.blockTail;
    }
    poolVars.beginIdx = poolVars.cBlockIdx * poolVars.blockFactor;
    poolVars.endIdx = poolVars.cBlockIdx * poolVars.blockFactor + calBlockNum;

    xGm.SetGlobalBuffer((__gm__ T*)x);
    maxGm.SetGlobalBuffer((__gm__ T*)y);
    indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);

    // 初始化que
    pipe->InitBuffer(inputQue, 1, 32 * 1024); // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(T) 有问题？
    pipe->InitBuffer(maxQue, 1, 8 * 1024);    // VL_NUM*doFactor*hoFactor*woFactorAlign*sizeof(T)
    pipe->InitBuffer(indexQue, 1, 8 * 1024);  // VL_NUM*doFactor*hoFactor*woFactorAlign*sizeof(int32)

    // 初始化Tbuf
    // VL_NUM/8 * ((kW>1)diFactor*hiFactor | (kH>1)woFactorAlign*diFactor | (kD>1)hoFactor * woFactorAlign)
    pipe->InitBuffer(cmpMaskBuffer, 512);
    pipe->InitBuffer(cmpNanMaskBuffer, 512);
    pipe->InitBuffer(inputTransBuffer, 32 * 1024); // VL_NUM*diFactor*hiFactor*wiFactorAlign*sizeof(float)
    pipe->InitBuffer(resetIndexBuf, 4 * 1024);     // VL_NUM*diFactor*hiFactor*sizeof(int32)
    pipe->InitBuffer(nextCmpBuffer, 4 * 1024);     // VL_NUM*diFactor*hiFactor*sizeof(int32)
    pipe->InitBuffer(mulWIdxBuffer, 32 * 1024);    // VL_NUM*diFactor*hiFactor*woFactorAlign*sizeof(int32)
    pipe->InitBuffer(mulWBuffer, 64 * 1024);       // VL_NUM*diFactor*hiFactor*wiFactor16Align*sizeof(float)
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::InitReset()
{
    int32_t inputVal(0);
    LocalTensor<int32_t> resetIdx = resetIndexBuf.Get<int32_t>();
    Duplicate<int32_t>(resetIdx, inputVal, 1024);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CalReset(const uint8_t diFactor, const uint8_t hiFactor)
{
    LocalTensor<int32_t> resetIdx = resetIndexBuf.Get<int32_t>();

    for (int i = 1; i < hiFactor; i++) {
        Adds(resetIdx[poolVars.VL_NUM * i], resetIdx, (int32_t)(poolVars.Wi * i), poolVars.VL_NUM);
    }
    PipeBarrier<PIPE_V>();
    for (int i = 1; i < diFactor; i++) {
        Adds(resetIdx[poolVars.VL_NUM * hiFactor * i], resetIdx, (int32_t)(poolVars.Wi * poolVars.Hi * i), poolVars.VL_NUM * hiFactor);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CopyIn(int64_t curIdx)
{
    auto blockVar = Pool3dMemCommon::CalcBlockVar(curIdx, poolVars);

    int32_t kerDStartIdxTotal = ((blockVar.curDoIdx * poolVars.doFactor) * poolVars.Di) / poolVars.Do;
    int32_t kerHStartIdxTotal = ((blockVar.curHoIdx * poolVars.hoFactor) * poolVars.Hi) / poolVars.Ho;
    int32_t kerWStartIdxTotal = ((blockVar.curWoIdx * poolVars.woFactor) * poolVars.Wi) / poolVars.Wo;
    int32_t kerDEndIdxTotal = Ceil((blockVar.curDoFactor + blockVar.curDoIdx * poolVars.doFactor) * poolVars.Di, poolVars.Do);
    int32_t kerHEndIdxTotal = Ceil((blockVar.curHoFactor + blockVar.curHoIdx * poolVars.hoFactor) * poolVars.Hi, poolVars.Ho);
    int32_t kerWEndIdxTotal = Ceil((blockVar.curWoFactor + blockVar.curWoIdx * poolVars.woFactor) * poolVars.Wi, poolVars.Wo);

    const uint8_t diFactor = kerDEndIdxTotal - kerDStartIdxTotal;
    const uint8_t hiFactor = kerHEndIdxTotal - kerHStartIdxTotal;
    const uint8_t wiFactor = kerWEndIdxTotal - kerWStartIdxTotal;

    auto xGmOffset = 
        blockVar.curNcIdx * poolVars.ncFactor * poolVars.DiHiWi + kerDStartIdxTotal * poolVars.HiWi + kerHStartIdxTotal * poolVars.Wi + kerWStartIdxTotal;

    CopyInput(blockVar.curNcFactor, diFactor, hiFactor, wiFactor, xGmOffset);
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CopyInput(
    int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor, int64_t xGmOffset)
{
    Pool3dMemCommon::CopyInputData<T, 1>(inputQue, xGm, curNcFactor, diFactor, hiFactor, wiFactor, xGmOffset, poolVars);
}

/*
 * 功能：input类型转换 <T> -> <fp32>类型, 并转置，把[VL, D, H, W] 转为[D, H, W, VL]
 */
template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::TransInput(
    int64_t curNcFactor, const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor)
{
    Pool3dMemCommon::TransposeInput<T, 1>(inputQue, inputTransBuffer, mulWBuffer, curNcFactor, diFactor, hiFactor, wiFactor, poolVars);
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::MaxPoolW(
    const uint8_t diFactor, const uint8_t hiFactor, const uint8_t wiFactor, int64_t curWoIdx, int64_t curWoFactor)
{
    LocalTensor<int32_t> resetIdx = resetIndexBuf.Get<int32_t>();

    LocalTensor<float> xLocalTransVL = inputTransBuffer.Get<float>();

    const uint8_t wiFactorAlign = Ceil(wiFactor, 8) * 8;

    int32_t kerWStartIdxTotal = ((curWoIdx * poolVars.woFactor) * poolVars.Wi) / poolVars.Wo;

    LocalTensor<int32_t> cmpIdx = nextCmpBuffer.Get<int32_t>();
    auto cmpIdxTmp = cmpIdx.ReinterpretCast<float>();

    LocalTensor<uint16_t> cmpMask = cmpMaskBuffer.Get<uint16_t>();
    LocalTensor<uint16_t> cmpMask2 = cmpNanMaskBuffer.Get<uint16_t>();
    uint64_t mask = 256 / sizeof(float);
    auto repeat = hiFactor * diFactor;
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(poolVars.VL_NUM / 8 * wiFactorAlign)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, (uint8_t)(8 * wiFactorAlign), 8};
    BinaryRepeatParams repeatParams2{1, 1, 1, 8, (uint8_t)(8 * wiFactorAlign), (uint8_t)(8 * wiFactorAlign)};

    LocalTensor<float> mulWUb = mulWBuffer.Get<float>();
    LocalTensor<int32_t> mulWIdxUb = mulWIdxBuffer.Get<int32_t>();
    auto mulWIdxCastUb = mulWIdxUb.ReinterpretCast<float>();

    for (int kernelIdx = 0; kernelIdx < curWoFactor; kernelIdx++) {
        int32_t kerWStartIdx = ((kernelIdx + curWoIdx * poolVars.woFactor) * poolVars.Wi) / poolVars.Wo;
        int32_t kerWEndIdx = Ceil((kernelIdx + curWoIdx * poolVars.woFactor + 1) * poolVars.Wi, poolVars.Wo);
        auto mulWOffset = kernelIdx * diFactor * hiFactor * poolVars.VL_NUM;
        auto inputOffset = poolVars.VL_NUM * (kerWStartIdx - kerWStartIdxTotal);

        Adds(mulWUb[mulWOffset], xLocalTransVL[inputOffset], (float)0.0, poolVars.VL_NUM, repeat, repeatCopyParams);
        Adds(mulWIdxUb[mulWOffset], resetIdx, (kerWStartIdx - kerWStartIdxTotal), poolVars.VL_NUM * repeat);
        PipeBarrier<PIPE_V>();
        for (int i = kerWStartIdx + 1; i < kerWEndIdx; i++) {
            Adds(cmpIdx, resetIdx, (i - kerWStartIdxTotal), poolVars.VL_NUM * repeat);
            auto nexCmpOffset = poolVars.VL_NUM * (i - kerWStartIdxTotal);
            Compare(cmpMask, xLocalTransVL[nexCmpOffset], mulWUb[mulWOffset], CMPMODE::GT, mask, repeat, repeatParams);
            Compare(
                cmpMask2, xLocalTransVL[nexCmpOffset], xLocalTransVL[nexCmpOffset], CMPMODE::EQ, mask, repeat,
                repeatParams2);
            PipeBarrier<PIPE_V>();
            Not(cmpMask2, cmpMask2, 128);
            PipeBarrier<PIPE_V>();
            Or(cmpMask, cmpMask, cmpMask2, 128);
            PipeBarrier<PIPE_V>();
            Select(
                mulWUb[mulWOffset], cmpMask, xLocalTransVL[nexCmpOffset], mulWUb[mulWOffset], selMode, mask, repeat,
                repeatParams);
            Select(mulWIdxCastUb[mulWOffset], cmpMask, cmpIdxTmp, mulWIdxCastUb[mulWOffset], selMode, poolVars.VL_NUM * repeat);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::MaxPoolH(
    const uint8_t diFactor, const uint8_t hiFactor, const uint8_t curWoFactor, int64_t curHoIdx, int64_t curHoFactor)
{
    auto woFactorAlign = Ceil(curWoFactor, 8) * 8;

    int32_t kerHStartIdxTotal = ((curHoIdx * poolVars.hoFactor) * poolVars.Hi) / poolVars.Ho;
    LocalTensor<int32_t> cmpIdx = nextCmpBuffer.Get<int32_t>();
    auto cmpIdxTmp = cmpIdx.ReinterpretCast<float>();
    LocalTensor<uint16_t> cmpMask = cmpMaskBuffer.Get<uint16_t>();
    LocalTensor<uint16_t> cmpMask2 = cmpNanMaskBuffer.Get<uint16_t>();
    uint64_t mask = 256 / sizeof(float);
    auto repeat = woFactorAlign * diFactor;
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(poolVars.VL_NUM / 8 * hiFactor)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, (uint8_t)(8 * hiFactor), 8};
    BinaryRepeatParams repeatParams2{1, 1, 1, 8, (uint8_t)(8 * hiFactor), (uint8_t)(8 * hiFactor)};
    LocalTensor<float> mulWUb = mulWBuffer.Get<float>();
    LocalTensor<int32_t> mulWIdxUb = mulWIdxBuffer.Get<int32_t>();
    auto mulWIdxCastUb = mulWIdxUb.ReinterpretCast<float>();
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
    LocalTensor<int32_t> mulWBufferInt = mulWBuffer.Get<int32_t>();
    LocalTensor<int32_t> mulHIdxUb = mulWBufferInt[8 * 1024]; // use mulWBuffer last 32K as index buff
    auto mulHIdxCastUb = mulHIdxUb.ReinterpretCast<float>();

    for (int kernelIdx = 0; kernelIdx < curHoFactor; kernelIdx++) {
        int32_t kerHStartIdx = ((kernelIdx + curHoIdx * poolVars.hoFactor) * poolVars.Hi) / poolVars.Ho;
        int32_t kerHEndIdx = Ceil((kernelIdx + curHoIdx * poolVars.hoFactor + 1) * poolVars.Hi, poolVars.Ho);
        auto mulHOffset = kernelIdx * repeat * poolVars.VL_NUM;
        auto mulWOffset = poolVars.VL_NUM * (kerHStartIdx - kerHStartIdxTotal);

        Adds(mulHUb[mulHOffset], mulWUb[mulWOffset], (float)0.0, poolVars.VL_NUM, repeat, repeatCopyParams);
        Adds(mulHIdxUb[mulHOffset], mulWIdxUb[mulWOffset], (int32_t)0, poolVars.VL_NUM, repeat, repeatCopyParams);
        PipeBarrier<PIPE_V>();
        for (int i = kerHStartIdx + 1; i < kerHEndIdx; i++) {
            auto nexCmpOffset = poolVars.VL_NUM * (i - kerHStartIdxTotal);
            Compare(cmpMask, mulWUb[nexCmpOffset], mulHUb[mulHOffset], CMPMODE::GT, mask, repeat, repeatParams);
            Compare(cmpMask2, mulWUb[nexCmpOffset], mulWUb[nexCmpOffset], CMPMODE::EQ, mask, repeat, repeatParams2);
            PipeBarrier<PIPE_V>();
            Not(cmpMask2, cmpMask2, 128);
            PipeBarrier<PIPE_V>();
            Or(cmpMask, cmpMask, cmpMask2, 128);
            PipeBarrier<PIPE_V>();
            Select(
                mulHUb[mulHOffset], cmpMask, mulWUb[nexCmpOffset], mulHUb[mulHOffset], selMode, mask, repeat,
                repeatParams);
            Select(
                mulHIdxCastUb[mulHOffset], cmpMask, mulWIdxCastUb[nexCmpOffset], mulHIdxCastUb[mulHOffset], selMode,
                mask, repeat, repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::MaxPoolD(
    const uint8_t diFactor, const uint8_t hiFactor, const uint8_t curWoFactor, int64_t curDoIdx, int64_t curDoFactor)
{
    auto woFactorAlign = Ceil(curWoFactor, 8) * 8;

    int32_t kerDStartIdxTotal = ((curDoIdx * poolVars.doFactor) * poolVars.Di) / poolVars.Do;
    LocalTensor<int32_t> cmpIdx = nextCmpBuffer.Get<int32_t>();
    auto cmpIdxTmp = cmpIdx.ReinterpretCast<float>();
    LocalTensor<uint16_t> cmpMask = cmpMaskBuffer.Get<uint16_t>();
    LocalTensor<uint16_t> cmpMask2 = cmpNanMaskBuffer.Get<uint16_t>();
    uint64_t mask = 256 / sizeof(float);
    auto repeat = poolVars.hoFactor * woFactorAlign;
    UnaryRepeatParams repeatCopyParams{1, 1, 8, (uint8_t)(poolVars.VL_NUM / 8 * diFactor)};
    BinaryRepeatParams repeatParams{1, 1, 1, 8, (uint8_t)(8 * diFactor), 8};
    BinaryRepeatParams repeatParams2{1, 1, 1, 8, (uint8_t)(8 * diFactor), (uint8_t)(8 * diFactor)};
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
    LocalTensor<int32_t> mulWBufferInt = mulWBuffer.Get<int32_t>();
    LocalTensor<int32_t> mulHIdxUb = mulWBufferInt[8 * 1024]; // use mulWBuffer last 32K as index buff
    auto mulHIdxCastUb = mulHIdxUb.ReinterpretCast<float>();

    LocalTensor<float> mulDUb = mulWBuffer.Get<float>();
    LocalTensor<int32_t> mulDIdxUb = mulWIdxBuffer.Get<int32_t>();
    auto mulDIdxCastUb = mulDIdxUb.ReinterpretCast<float>();

    for (int kernelIdx = 0; kernelIdx < curDoFactor; kernelIdx++) {
        int32_t kerDStartIdx = ((kernelIdx + curDoIdx * poolVars.doFactor) * poolVars.Di) / poolVars.Do;
        int32_t kerDEndIdx = Ceil((kernelIdx + curDoIdx * poolVars.doFactor + 1) * poolVars.Di, poolVars.Do);
        auto mulDOffset = kernelIdx * repeat * poolVars.VL_NUM;
        auto mulHOffset = poolVars.VL_NUM * (kerDStartIdx - kerDStartIdxTotal);

        Adds(mulDUb[mulDOffset], mulHUb[mulHOffset], (float)0.0, poolVars.VL_NUM, repeat, repeatCopyParams);
        Adds(mulDIdxUb[mulDOffset], mulHIdxUb[mulHOffset], (int32_t)0.0, poolVars.VL_NUM, repeat, repeatCopyParams);
        PipeBarrier<PIPE_V>();
        for (int i = kerDStartIdx + 1; i < kerDEndIdx; i++) {
            auto nexCmpOffset = poolVars.VL_NUM * (i - kerDStartIdxTotal);
            Compare(cmpMask, mulHUb[nexCmpOffset], mulDUb[mulDOffset], CMPMODE::GT, mask, repeat, repeatParams);
            Compare(cmpMask2, mulHUb[nexCmpOffset], mulHUb[nexCmpOffset], CMPMODE::EQ, mask, repeat, repeatParams2);
            PipeBarrier<PIPE_V>();
            Not(cmpMask2, cmpMask2, 128);
            PipeBarrier<PIPE_V>();
            Or(cmpMask, cmpMask, cmpMask2, 128);
            PipeBarrier<PIPE_V>();
            Select(
                mulDUb[mulDOffset], cmpMask, mulHUb[nexCmpOffset], mulDUb[mulDOffset], selMode, mask, repeat,
                repeatParams);
            Select(
                mulDIdxCastUb[mulDOffset], cmpMask, mulHIdxCastUb[nexCmpOffset], mulDIdxCastUb[mulDOffset], selMode,
                mask, repeat, repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }
}

/*
 * 功能：<fp32>类型out和<int32>类型index的转置，把[D, H, W, VL]转为[VL, D, H, W]
 * 同时会cast out为<T>类型
 */
template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::TransOutAndIdx(
    int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor, int64_t UbIdxOffset)
{
    auto curWoFactorAlign = Ceil(curWoFactor, 8) * 8;
    auto curWoFactorAlign16 = Ceil(curWoFactor, 32 / sizeof(T)) * 32 / sizeof(T);
    LocalTensor<int32_t> mulDIdxUb = mulWIdxBuffer.Get<int32_t>();
    auto mulDIdxCastUb = mulDIdxUb.ReinterpretCast<float>();
    Adds(mulDIdxUb, mulDIdxUb, (int32_t)UbIdxOffset, poolVars.hoFactor * curWoFactorAlign * poolVars.doFactor * poolVars.VL_NUM);
    LocalTensor<int32_t> indexLocal = indexQue.AllocTensor<int32_t>();
    LocalTensor<float> indexLocalTmp = indexLocal.ReinterpretCast<float>();

    PipeBarrier<PIPE_V>();
    LocalTensor<T> yLocal = maxQue.AllocTensor<T>();
    LocalTensor<float> mulDUb = mulWBuffer.Get<float>();
    LocalTensor<float> mulHUb = inputTransBuffer.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        Pool3dMemCommon::OutTranspose<T>(yLocal, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, poolVars.VL_NUM);
    } else {
        if (curWoFactorAlign == curWoFactorAlign16) {
            Pool3dMemCommon::OutTranspose<T>(mulHUb, mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, poolVars.VL_NUM);
        } else {
            UnaryRepeatParams repeatCastParams2{
                (uint16_t)(curWoFactorAlign16 / 8), (uint16_t)(curWoFactorAlign / 8),
                (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign16, 16) * 2),
                (uint8_t)(Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 2)};
            Pool3dMemCommon::OutTranspose<T>(mulHUb[4096], mulDUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, poolVars.VL_NUM);
            PipeBarrier<PIPE_V>();

            Adds(
                mulHUb, mulHUb[4096], (float)0.0, (uint8_t)(curWoFactorAlign * curDoFactor * curHoFactor), poolVars.VL_NUM,
                repeatCastParams2);
        }
        PipeBarrier<PIPE_V>();

        Cast(yLocal, mulHUb, RoundMode::CAST_ROUND, poolVars.VL_NUM * curWoFactorAlign16 * curDoFactor * curHoFactor);
    }
    maxQue.EnQue(yLocal);

    Pool3dMemCommon::OutTranspose<T>(indexLocalTmp, mulDIdxCastUb, Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16, poolVars.VL_NUM);
    indexQue.EnQue(indexLocal);
}

/*
 * 功能：搬出<T>类型out和<int32>类型index
 */
template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::CopyOutAndIdx(
    int64_t curNcFactor, int64_t curDoFactor, int64_t curHoFactor, int64_t curWoFactor, int64_t yGmOffset)
{
    auto curWoFactorAlign = Ceil(curWoFactor, 8) * 8;
    auto curWoFactorAlign16 = Ceil(curWoFactor, 32 / sizeof(T)) * 32 / sizeof(T);

    LocalTensor<T> yLocal = maxQue.DeQue<T>();

    DataCopyExtParams paramsOut2 = {
        static_cast<uint16_t>(curHoFactor), static_cast<uint32_t>(curWoFactor * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((poolVars.Wo - curWoFactor) * sizeof(T)), static_cast<uint32_t>(0)};
    for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
        for (int64_t dCopyi = 0; dCopyi < curDoFactor; dCopyi++) {
            auto dstAddr = yGmOffset + ncCopyi * poolVars.Do * poolVars.Ho * poolVars.Wo + dCopyi * poolVars.Ho * poolVars.Wo;
            auto srcAddr = ncCopyi * Ceil(curDoFactor * curHoFactor * curWoFactorAlign16, 16) * 16 +
                           dCopyi * curHoFactor * curWoFactorAlign16;
            DataCopyPad(maxGm[dstAddr], yLocal[srcAddr], paramsOut2);
        }
    }
    maxQue.FreeTensor(yLocal);

    paramsOut2.blockLen = curWoFactor * sizeof(int32_t);
    paramsOut2.dstStride = (poolVars.Wo - curWoFactor) * sizeof(int32_t);
    LocalTensor<int32_t> indexLocal = indexQue.DeQue<int32_t>();
    for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
        for (int64_t dCopyi = 0; dCopyi < curDoFactor; dCopyi++) {
            auto dstAddr = yGmOffset + ncCopyi * poolVars.Do * poolVars.Ho * poolVars.Wo + dCopyi * poolVars.Ho * poolVars.Wo;
            auto srcAddr = ncCopyi * Ceil(curDoFactor * curHoFactor * curWoFactorAlign, 16) * 16 +
                           dCopyi * curHoFactor * curWoFactorAlign;
            DataCopyPad(indicesGm[dstAddr], indexLocal[srcAddr], paramsOut2);
        }
    }
    indexQue.FreeTensor(indexLocal);
}

template <typename T>
__aicore__ inline void AdaptiveMaxPool3dSmallPool<T>::Process()
{
    if (poolVars.cBlockIdx >= poolVars.useCoreNum) {
        return;
    }

    InitReset();
    for (auto curIdx = poolVars.beginIdx; curIdx < poolVars.endIdx; curIdx++) {
        // 按照outer切分，当前在[NC, Doo, Hoo, Woo]上的第几个UB块和当前计算多少块kernel
        auto blockVar = Pool3dMemCommon::CalcBlockVar(curIdx, poolVars);
        // 按照inner切分，计算当前起始和终止位置
        int32_t kerDStartIdxTotal = ((blockVar.curDoIdx * poolVars.doFactor) * poolVars.Di) / poolVars.Do;
        int32_t kerDEndIdxTotal = Ceil((blockVar.curDoFactor + blockVar.curDoIdx * poolVars.doFactor) * poolVars.Di, poolVars.Do);
        int32_t kerHStartIdxTotal = ((blockVar.curHoIdx * poolVars.hoFactor) * poolVars.Hi) / poolVars.Ho;
        int32_t kerHEndIdxTotal = Ceil((blockVar.curHoFactor + blockVar.curHoIdx * poolVars.hoFactor) * poolVars.Hi, poolVars.Ho);
        int32_t kerWStartIdxTotal = ((blockVar.curWoIdx * poolVars.woFactor) * poolVars.Wi) / poolVars.Wo;
        int32_t kerWEndIdxTotal = Ceil((blockVar.curWoFactor + blockVar.curWoIdx * poolVars.woFactor) * poolVars.Wi, poolVars.Wo);

        const uint8_t diFactor = kerDEndIdxTotal - kerDStartIdxTotal;
        const uint8_t hiFactor = kerHEndIdxTotal - kerHStartIdxTotal;
        const uint8_t wiFactor = kerWEndIdxTotal - kerWStartIdxTotal;

        // 搬入搬出和ub内index相对gm的偏移
        auto xGmOffset = blockVar.curNcIdx * poolVars.ncFactor * poolVars.DiHiWi + kerDStartIdxTotal * poolVars.HiWi + 
                        kerHStartIdxTotal * poolVars.Wi + kerWStartIdxTotal;
        auto yGmOffset = blockVar.curNcIdx * poolVars.ncFactor * poolVars.Do * poolVars.Ho * poolVars.Wo + 
                    blockVar.curDoIdx * poolVars.doFactor * poolVars.Ho * poolVars.Wo + 
                    blockVar.curHoIdx * poolVars.hoFactor * poolVars.Wo + blockVar.curWoIdx * poolVars.woFactor;
        int64_t UbIdxOffset = kerDStartIdxTotal * poolVars.HiWi + kerHStartIdxTotal * poolVars.Wi + kerWStartIdxTotal;
        if (curIdx == poolVars.beginIdx) {
            CopyInput(blockVar.curNcFactor, diFactor, hiFactor, wiFactor, xGmOffset);
        }
        // [VL_NUM, diFactor, hiFactor, wiFactorAlign] => [diFactor, hiFactor, wiFactorAlign, VL_NUM]
        TransInput(blockVar.curNcFactor, diFactor, hiFactor, wiFactor);

        if (curIdx != poolVars.endIdx - 1) {
            CopyIn(curIdx + 1);
        }

        CalReset(diFactor, hiFactor);
        // [diFactor, hiFactor, wiFactorAlign, VL_NUM] => [woFactorAlign, diFactor, hiFactor, VL_NUM]
        MaxPoolW(diFactor, hiFactor, wiFactor, blockVar.curWoIdx, blockVar.curWoFactor);
        // [woFactorAlign, diFactor, hiFactor, VL_NUM] => [hoFactor, woFactorAlign, diFactor, VL_NUM]
        MaxPoolH(diFactor, hiFactor, blockVar.curWoFactor, blockVar.curHoIdx, blockVar.curHoFactor);
        // [hoFactor, woFactorAlign, diFactor, hiFactor, VL_NUM] => [doFactor, hoFactor, woFactorAlign, diFactor, VL_NUM]
        MaxPoolD(diFactor, hiFactor, blockVar.curWoFactor, blockVar.curDoIdx, blockVar.curDoFactor);
        // [doFactor, hoFactor, woFactorAlign, diFactor, VL_NUM] => [VL_NUM, doFactor, hoFactor, woFactorAlign, diFactor]
        TransOutAndIdx(blockVar.curDoFactor, blockVar.curHoFactor, blockVar.curWoFactor, UbIdxOffset);
        CopyOutAndIdx(blockVar.curNcFactor, blockVar.curDoFactor, blockVar.curHoFactor, blockVar.curWoFactor, yGmOffset);
    }
}
#endif // ADAPTIVE_MAX_POOL3D_SAMLL_POOL_H_