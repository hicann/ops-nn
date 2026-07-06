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
 * \file mse_loss_grad.h
 * \brief
 */
#ifndef __MSE_LOSS_GRAD_H__
#define __MSE_LOSS_GRAD_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mse_loss_grad_tiling_data.h"
#include "mse_loss_grad_tiling_key.h"

namespace NsMseLossGrad {

using namespace AscendC;

template <typename T>
class MseLossGrad {
public:
    __aicore__ inline MseLossGrad(){};
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y,
                                const MseLossGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueIN;
    TQue<QuePosition::VECOUT, 1> outQueueOUT;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    GlobalTensor<T> outGm;
    TBuf<TPosition::VECCALC> resultTmpBuf;
    TBuf<TPosition::VECCALC> calcValueLocal;
    float cof = 2.0;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    int32_t bufferNum = 1;
};

template <typename T>
__aicore__ inline void MseLossGrad<T>::Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y,
                                            const MseLossGradTilingData* tilingData)
{
    this->cof = tilingData->cof;
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    this->bufferNum = 1;
    if (static_cast<int32_t>(tilingData->usedDb) == 1) {
        this->bufferNum = 2;
    }
    xGm.SetGlobalBuffer((__gm__ T*)predict + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ T*)label + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ T*)dout + globalBufferIndex, this->coreDataNum);
    outGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueIN, this->bufferNum, this->tileDataNum * 3 * sizeof(T));
    pipe.InitBuffer(outQueueOUT, this->bufferNum, this->tileDataNum * sizeof(T));
    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        pipe.InitBuffer(resultTmpBuf, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(calcValueLocal, this->tileDataNum * 3 * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void MseLossGrad<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.AllocTensor<T>();

    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = this->processDataNum * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;

    DataCopyPad(inLocal[0], xGm[progress * this->tileDataNum], copyInParams, padParams);
    DataCopyPad(inLocal[this->tileDataNum], yGm[progress * this->tileDataNum], copyInParams, padParams);
    DataCopyPad(inLocal[2 * this->tileDataNum], zGm[progress * this->tileDataNum], copyInParams, padParams);

    inQueueIN.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MseLossGrad<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> outLocal = outQueueOUT.DeQue<T>();

    DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = this->processDataNum * sizeof(T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopy(outGm[progress * this->tileDataNum], outLocal, copyOutParams);

    outQueueOUT.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MseLossGrad<T>::Compute(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.DeQue<T>();
    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<float> calcValueLocalFP32 = calcValueLocal.Get<float>();
        Cast(calcValueLocalFP32, inLocal, RoundMode::CAST_NONE, this->tileDataNum * 3);

        LocalTensor<float> xLocal = calcValueLocalFP32;
        LocalTensor<float> yLocal = calcValueLocalFP32[this->tileDataNum];
        LocalTensor<float> zLocal = calcValueLocalFP32[2 * (this->tileDataNum)];

        LocalTensor<float> outLoclFp32 = resultTmpBuf.Get<float>();
        Sub(outLoclFp32, xLocal, yLocal, this->processDataNum);
        Muls(outLoclFp32, outLoclFp32, static_cast<float>(this->cof), this->processDataNum);
        Mul(outLoclFp32, outLoclFp32, zLocal, this->processDataNum);

        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();

        Cast(outLocal, outLoclFp32, RoundMode::CAST_RINT, this->tileDataNum);

        outQueueOUT.EnQue(outLocal);
    } else {
        LocalTensor<T> xLocal = inLocal;
        LocalTensor<T> yLocal = inLocal[this->tileDataNum];
        LocalTensor<T> zLocal = inLocal[2 * (this->tileDataNum)];
        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
        Sub(outLocal, xLocal, yLocal, this->processDataNum);
        Muls(outLocal, outLocal, static_cast<T>(this->cof), this->processDataNum);
        Mul(outLocal, outLocal, zLocal, this->processDataNum);
        outQueueOUT.EnQue(outLocal);
    }
    inQueueIN.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void MseLossGrad<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

template <typename T>
class MseLossGradScale {
public:
    __aicore__ inline MseLossGradScale(){};
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y,
                                const MseLossGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueIN;
    TQue<QuePosition::VECOUT, 1> outQueueOUT;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    GlobalTensor<T> outGm;
    TBuf<TPosition::VECCALC> resultTmpBuf;
    TBuf<TPosition::VECCALC> calcValueLocal;
    float doutScaleValue = 0.0;
    float cof = 2.0;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    int32_t bufferNum = 1;
};

template <typename T>
__aicore__ inline void MseLossGradScale<T>::Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y,
                                                 const MseLossGradTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->cof = tilingData->cof;
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    this->bufferNum = 1;
    if (static_cast<int32_t>(tilingData->usedDb) == 1) {
        this->bufferNum = 2;
    }
    xGm.SetGlobalBuffer((__gm__ T*)predict + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ T*)label + globalBufferIndex, this->coreDataNum);

    outGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    zGm.SetGlobalBuffer((__gm__ T*)dout, 1); // dout标量
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        T scaleValue = zGm.GetValue(0);
        this->doutScaleValue = ToFloat(scaleValue);
    } else if constexpr (IsSameType<T, half>::value) {
        T scaleValue = zGm.GetValue(0);
        this->doutScaleValue = (float)scaleValue;
    } else {
        this->doutScaleValue = (float)(zGm.GetValue(0));
    }

    pipe.InitBuffer(inQueueIN, this->bufferNum, this->tileDataNum * 2 * sizeof(T));
    pipe.InitBuffer(outQueueOUT, this->bufferNum, this->tileDataNum * sizeof(T));
    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        pipe.InitBuffer(resultTmpBuf, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(calcValueLocal, this->tileDataNum * 2 * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void MseLossGradScale<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.AllocTensor<T>();

    DataCopy(inLocal[0], xGm[progress * this->tileDataNum], this->processDataNum);
    DataCopy(inLocal[this->tileDataNum], yGm[progress * this->tileDataNum], this->processDataNum);

    inQueueIN.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MseLossGradScale<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> outLocal = outQueueOUT.DeQue<T>();

    DataCopy(outGm[progress * this->tileDataNum], outLocal, this->processDataNum);

    outQueueOUT.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MseLossGradScale<T>::Compute(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.DeQue<T>();
    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<float> calcValueLocalFP32 = calcValueLocal.Get<float>();

        Cast(calcValueLocalFP32, inLocal, RoundMode::CAST_NONE, this->tileDataNum * 2);

        LocalTensor<float> xLocal = calcValueLocalFP32;
        LocalTensor<float> yLocal = calcValueLocalFP32[this->tileDataNum];

        LocalTensor<float> outLoclFp32 = resultTmpBuf.Get<float>();
        Sub(outLoclFp32, xLocal, yLocal, this->processDataNum);
        Muls(outLoclFp32, outLoclFp32, static_cast<float>(this->cof), this->processDataNum);
        Muls(outLoclFp32, outLoclFp32, static_cast<float>(this->doutScaleValue), this->processDataNum);

        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();

        Cast(outLocal, outLoclFp32, RoundMode::CAST_RINT, this->tileDataNum);

        outQueueOUT.EnQue(outLocal);
    } else {
        LocalTensor<T> xLocal = inLocal;
        LocalTensor<T> yLocal = inLocal[this->tileDataNum];
        LocalTensor<T> zLocal = inLocal[2 * (this->tileDataNum)];
        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
        Sub(outLocal, xLocal, yLocal, this->processDataNum);
        Muls(outLocal, outLocal, static_cast<T>(this->cof), this->processDataNum);
        Muls(outLocal, outLocal, static_cast<T>(this->doutScaleValue), this->processDataNum);
        outQueueOUT.EnQue(outLocal);
    }
    inQueueIN.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void MseLossGradScale<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsMseLossGrad
#endif // MSE_LOSS_GRAD_H