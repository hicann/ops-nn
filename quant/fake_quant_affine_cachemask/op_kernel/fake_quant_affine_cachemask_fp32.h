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
 * \file fake_quant_affine_cachemask_fp32.h
 * \brief
 */
#ifndef _FAKE_QUANT_AFFINE_CACHEMASK_FP32_H_
#define _FAKE_QUANT_AFFINE_CACHEMASK_FP32_H_
#include "fake_quant_affine_cachemask_base.h"

namespace FakeQuantAffineCachemaskN {
using namespace AscendC;
using namespace std;

template <typename yType>
class FakeQuantAffineCachemaskFp32 : public FakeQuantAffineCachemaskBase<yType>
{
public:
    __aicore__ inline FakeQuantAffineCachemaskFp32()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR scale, GM_ADDR zero_point, GM_ADDR y, GM_ADDR mask,
        const FakeQuantAffineCachemaskTilingData* tilingData)
    {
        this->BaseMemberDataInit(tilingData);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(x) + this->offset, this->blockLength);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(scale) + this->scaleOffset, this->circleNum);
        zeroGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_ZERO_POINT*>(zero_point) + this->scaleOffset, this->circleNum);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask) + this->offset, this->blockLength);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueData, BUFFER_NUM, this->tileLength * sizeof(yType));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileLength * sizeof(yType));
        pipe.InitBuffer(outQueueMask, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(calcBuf, BUFFER_NUM * this->tileLength * sizeof(yType));
        pipe.InitBuffer(calcInt32Buf, BUFFER_NUM * this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(calcHf16Buf, BUFFER_NUM * this->tileLength * sizeof(half));
        pipe.InitBuffer(maskBuf, BUFFER_NUM * this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(selectBuf, BUFFER_NUM * this->tileLength * sizeof(yType));

        pipe.InitBuffer(quantMinQueueBuf, this->tileLength * sizeof(yType));
        pipe.InitBuffer(quantMaxQueueBuf, this->tileLength * sizeof(yType));
        pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
        pipe.InitBuffer(infBuf, this->tileLength * sizeof(yType));
        pipe.InitBuffer(oneBuf, this->tileLength * sizeof(yType));
    }

    __aicore__ inline void Process()
    {
        this->CommonBufferGet(
            infBuf, zeroBuf, oneBuf, quantMinQueueBuf, quantMaxQueueBuf, infTensor, zeroTensor, oneTensor,
            quantMinTensor, quantMaxTensor, this->tileLength);
        for (uint32_t i = 0; i < this->circleNum; i++) {
            scaleValue = static_cast<float>(scaleGm.GetValue(i));
            zeroPointValue = zeroGm.GetValue(i);

            uint32_t dataOffset = i * this->calcLength;
            for (uint32_t j = 0; j < this->tileNum; j++) {
                uint32_t calcOffset = dataOffset + j * this->tileLength;
                repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
                this->CommonCopyIn(inQueueData, xGm, calcOffset, this->tileLength);
                PipeBarrier<PIPE_ALL>();
                Compute(this->tileLength);
                PipeBarrier<PIPE_ALL>();
                this->CommonCopyOut(outQueueOut, outQueueMask, yGm, maskGm, calcOffset, this->tileLength);
                PipeBarrier<PIPE_ALL>();
            }

            if (this->lastTileLength > 0) {
                uint32_t calcOffset = dataOffset + this->tileNum * this->tileLength;
                repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
                this->CommonCopyIn(inQueueData, xGm, calcOffset, this->lastActulTileLength);
                PipeBarrier<PIPE_ALL>();
                Compute(this->tileLength);
                PipeBarrier<PIPE_ALL>();
                this->CommonCopyOut(outQueueOut, outQueueMask, yGm, maskGm, calcOffset, this->lastActulTileLength);
            }
        }
    }

private:
    __aicore__ inline void Compute(uint32_t calCount)
    {
        LocalTensor<yType> xLocal = inQueueData.DeQue<yType>();
        LocalTensor<yType> yLocal = outQueueOut.AllocTensor<yType>();
        LocalTensor<uint8_t> maskLocal = outQueueMask.AllocTensor<uint8_t>();

        curTemp = calcBuf.Get<yType>();
        curInt32Temp = calcInt32Buf.Get<int32_t>();
        curHf16Temp = calcHf16Buf.Get<half>();
        selectTemp = selectBuf.Get<yType>();
        maskTemp = maskBuf.Get<uint8_t>();

        Muls(curTemp, xLocal, static_cast<yType>(1.0f / scaleValue), calCount);
        if constexpr (IsSameType<DTYPE_ZERO_POINT, int32_t>::value) {
            // tmp = x / scale + zero_point
            Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);
            Cast(curTemp, curInt32Temp, RoundMode::CAST_NONE, calCount);
            Adds(curTemp, curTemp, static_cast<yType>(zeroPointValue), calCount);
            Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);
        } else {
            // tmp = lrintf(x / scale + zero_point)
            Adds(curTemp, curTemp, static_cast<yType>(zeroPointValue), calCount);
            Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);
            Cast(curTemp, curInt32Temp, RoundMode::CAST_NONE, calCount);
        }        
        PipeBarrier<PIPE_ALL>();

        // maskTemp = (round(curTemp) >= quant_min) & (round(curTemp) <= quant_max)
        Compare(maskTemp, curTemp, quantMinTensor, CMPMODE::GE, calCount);
        BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
        Select(
            selectTemp, maskTemp, oneTensor, zeroTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes,
            repeatParams);
        Compare(maskTemp, curTemp, quantMaxTensor, CMPMODE::LE, calCount);
        Select(
            curTemp, maskTemp, oneTensor, zeroTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes,
            repeatParams);
        // use Mul to replace &
        Mul(curTemp, selectTemp, curTemp, calCount);
        // fp32 -> fp16 -> uint8
        Cast(curHf16Temp, curTemp, RoundMode::CAST_RINT, calCount);
        Cast(maskLocal, curHf16Temp, RoundMode::CAST_RINT, calCount);
        PipeBarrier<PIPE_ALL>();

        // output = (Mins(Maxs(tmp, quant_min), quant_max) - zero_point) * scale
        Maxs(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMin), calCount);
        Mins(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMax), calCount);
        Cast(curTemp, curInt32Temp, RoundMode::CAST_ROUND, calCount);
        Compare(maskTemp, xLocal, xLocal, CMPMODE::EQ, calCount);
        Select(
            curTemp, maskTemp, curTemp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
            this->mask, repeatTimes, repeatParams);
        Adds(curTemp, curTemp, static_cast<yType>(-1 * static_cast<float>(zeroPointValue)), calCount);
        Muls(yLocal, curTemp, static_cast<yType>(scaleValue), calCount);
        PipeBarrier<PIPE_ALL>();

        outQueueOut.EnQue<yType>(yLocal);
        outQueueMask.EnQue<uint8_t>(maskLocal);
        inQueueData.FreeTensor(xLocal);
    }

private:
    TPipe pipe;

    DTYPE_ZERO_POINT zeroPointValue = 1;
    float scaleValue = 1.0;
    uint8_t repeatTimes = 0;

    GlobalTensor<yType> xGm, scaleGm, yGm;
    GlobalTensor<DTYPE_ZERO_POINT> zeroGm;
    GlobalTensor<uint8_t> maskGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut, outQueueMask;
    TBuf<QuePosition::VECCALC> calcBuf, calcInt32Buf, calcHf16Buf, maskBuf, quantMinQueueBuf, quantMaxQueueBuf, infBuf,
        zeroBuf, oneBuf, selectBuf;

    LocalTensor<yType> curTemp, infTensor, zeroTensor, oneTensor, quantMinTensor, quantMaxTensor, selectTemp;
    LocalTensor<int32_t> curInt32Temp;
    LocalTensor<half> curHf16Temp;
    LocalTensor<uint8_t> maskTemp;
};
} // namespace FakeQuantAffineCachemaskN
#endif // _FAKE_QUANT_AFFINE_CACHEMASK_FP32_H_