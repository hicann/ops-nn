/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad_v2.h
 * \brief
 */
#ifndef MSE_LOSS_GRAD_V2_H
#define MSE_LOSS_GRAD_V2_H
#pragma once
#include "mse_loss_grad_v2_base.h"
using namespace AscendC;

template <typename inType>
class KernelMseLossGrad910 : public KernelMseLossGradBase<inType>
{
public:
    __aicore__ inline KernelMseLossGrad910()
    {}
    __aicore__ inline void Init(
        GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, float cof, uint64_t totalLength, uint64_t tileNum,
        uint64_t blockLength, uint64_t padLength, uint64_t usedDb)
    {
        this->cof = cof;
        this->totalLength = static_cast<int32_t>(totalLength);
        this->blockLength = blockLength;
        uint32_t blockNum = GetBlockNum();
        this->gmSize = blockLength;
        this->tileNum = tileNum;
        this->bufferNum = 1;
        if (static_cast<int32_t>(usedDb) == 1) {
            // 2 means using double buffer
            this->bufferNum = 2;
            this->tileNum = this->tileNum * this->bufferNum;
        }
        this->tileLength = (this->blockLength + this->tileNum - 1) / this->tileNum;
        if (static_cast<int32_t>(GetBlockIdx()) == static_cast<int32_t>(blockNum) - 1 &&
            static_cast<int32_t>(padLength) != 0) {
            this->tileNum = 1;
            this->bufferNum = 1;
            this->tileLength = padLength;
            this->gmSize = padLength;
        }
        this->tileLengthAlign = this->CeilAlign(this->tileLength);
        this->tileLengthPtr = this->tileLength;

        uint64_t gmOffset = this->blockLength * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ inType*)predict + gmOffset, this->gmSize);
        yGm.SetGlobalBuffer((__gm__ inType*)label + gmOffset, this->gmSize);
        zGm.SetGlobalBuffer((__gm__ inType*)dout + gmOffset, this->gmSize);
        outGm.SetGlobalBuffer((__gm__ inType*)y + gmOffset, this->gmSize);

        // 3 means there are 3 inputs
        pipe.InitBuffer(inQueueIN, this->bufferNum, this->tileLengthAlign * 3 * sizeof(inType));
        pipe.InitBuffer(outQueueOUT, this->bufferNum, this->tileLengthAlign * sizeof(inType));

        if constexpr (IsSameType<inType, bfloat16_t>::value || IsSameType<inType, half>::value) {
            pipe.InitBuffer(resultTmpBuf, this->tileLengthAlign * sizeof(float));
            // 3 means there are 3 inputs
            pipe.InitBuffer(calcValueLocal, this->tileLengthAlign * 3 * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1) {
                this->tileLength = this->gmSize - this->tileLengthPtr * i;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<inType> inLocal = inQueueIN.AllocTensor<inType>();
        DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(inType)), 0, 0, 0};
        DataCopyPadExtParams<inType> padParams = {false, 0, 0, 0};
        DataCopyPad<inType>(inLocal[0], xGm[progress * this->tileLengthPtr], copyParams, padParams);
        DataCopyPad<inType>(inLocal[this->tileLengthAlign], yGm[progress * this->tileLengthPtr], copyParams, padParams);
        DataCopyPad<inType>(
            inLocal[2 * this->tileLengthAlign], // 2 means two tileLengthAlign offsets are needed
            zGm[progress * this->tileLengthPtr], copyParams, padParams);
        inQueueIN.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<inType> inLocal = inQueueIN.DeQue<inType>();
        if constexpr (IsSameType<inType, bfloat16_t>::value || IsSameType<inType, half>::value) {
            // convert bf16 to fp32
            LocalTensor<float> calcValueLocalFP32 = calcValueLocal.Get<float>();
            // 3 means there are 3 inputs
            Cast(calcValueLocalFP32, inLocal, RoundMode::CAST_NONE, this->tileLengthAlign * 3);
            LocalTensor<float> xLocal = calcValueLocalFP32;
            LocalTensor<float> yLocal = calcValueLocalFP32[this->tileLengthAlign];
            LocalTensor<float> zLocal = calcValueLocalFP32[2 * (this->tileLengthAlign)];
            LocalTensor<float> outLoclFp32 = resultTmpBuf.Get<float>();
            Sub(outLoclFp32, xLocal, yLocal, this->tileLength);
            Muls(outLoclFp32, outLoclFp32, static_cast<float>(this->cof), this->tileLength);
            Mul(outLoclFp32, outLoclFp32, zLocal, this->tileLength);
            // convert fp32 to bf16
            LocalTensor<inType> outLocal = outQueueOUT.AllocTensor<inType>();
            // 3 means there are 3 inputs
            Cast(outLocal, outLoclFp32, RoundMode::CAST_RINT, this->tileLengthAlign * 3);
            outQueueOUT.EnQue(outLocal);
        } else {
            LocalTensor<inType> xLocal = inLocal;
            LocalTensor<inType> yLocal = inLocal[this->tileLengthAlign];
            LocalTensor<inType> zLocal = inLocal[2 * (this->tileLengthAlign)];
            LocalTensor<inType> outLocal = outQueueOUT.AllocTensor<inType>();
            Sub(outLocal, xLocal, yLocal, this->tileLength);
            Muls(outLocal, outLocal, static_cast<inType>(this->cof), this->tileLength);
            Mul(outLocal, outLocal, zLocal, this->tileLength);
            outQueueOUT.EnQue(outLocal);
        }
        inQueueIN.FreeTensor(inLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<inType> outLocal = outQueueOUT.DeQue<inType>();
        DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(inType)), 0, 0, 0};
        DataCopyPad<inType>(outGm[progress * this->tileLengthPtr], outLocal, copyParams);

        outQueueOUT.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1, &conf> inQueueIN;
    TQue<QuePosition::VECOUT, 1, &conf> outQueueOUT;
    GlobalTensor<inType> xGm;
    GlobalTensor<inType> yGm;
    GlobalTensor<inType> zGm;
    GlobalTensor<inType> outGm;
    TBuf<TPosition::VECCALC> resultTmpBuf;
    TBuf<TPosition::VECCALC> calcValueLocal;
};
#endif // _MSE_LOSS_GRAD_V2_H
