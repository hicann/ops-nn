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
 * \file cross_v2.cpp
 * \brief y = cross(x1, x2, dim)
 */
#include "kernel_operator.h"
namespace CrossV2NS {
using namespace AscendC;
const int32_t BUFFER_NUM = 2;
const int64_t INPUT_VAR_NUM = 2;
const int64_t MAX_SHAPE_DIM = 8;
const int64_t SPECIFIED_DIM_SIZE = 3;
const uint32_t UINT8_ALIGN_SIZE = 256;
const uint32_t BLOCK_SIZE = 32;
const uint32_t NUM_PER_BLK_32 = 8;
const uint32_t NUM_PER_BLK_16 = 16;
const uint32_t IDX_0 = 0;
const uint32_t IDX_1 = 1;
const uint32_t IDX_2 = 2;
const half HALF_ZERO = 0;
const half HALF_256 = 256;
const uint16_t GATHER_MASK_UINT16_0 = 0b1001'0010'0100'1001;                     // 每个repeat内每三个元素取第一个元素
const uint16_t GATHER_MASK_UINT16_1 = 0b0010'0100'1001'0010;                     // 每个repeat内每三个元素取第二个元素
const uint16_t GATHER_MASK_UINT16_2 = 0b0100'1001'0010'0100;                     // 每个repeat内每三个元素取第三个元素
const uint32_t GATHER_MASK_UINT32_0 = 0b0100'1001'0010'0100'1001'0010'0100'1001; // 每个repeat内每三个元素取第一个元素
const uint32_t GATHER_MASK_UINT32_1 = 0b1001'0010'0100'1001'0010'0100'1001'0010; // 每个repeat内每三个元素取第二个元素
const uint32_t GATHER_MASK_UINT32_2 = 0b0010'0100'1001'0010'0100'1001'0010'0100; // 每个repeat内每三个元素取第三个元素

__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return x;
    }
    return (x + y - 1) / y * y;
}

template <typename T>
class KernelCross
{
public:
    __aicore__ inline KernelCross()
    {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int64_t stepSize, int64_t tileNum, int64_t tileNumPerBatch,
        int64_t tileDataNum, int64_t tailDataNum)
    {
        this->coreIdx = GetBlockIdx();
        this->coreNum = GetBlockNum();
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        this->stepSize = stepSize;
        this->tileNum = tileNum;
        this->tileNumPerBatch = tileNumPerBatch;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = tailDataNum;

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueY1, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueZ1, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueY2, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueZ2, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueX3, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueY3, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueZ3, BUFFER_NUM, this->tileDataNum * sizeof(T));
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(float));
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(half));
        } else {
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t loopCount = this->tileNum;
        for (int64_t i = coreIdx; i < loopCount; i += this->coreNum) {
            this->processDataNum = this->tileDataNum;
            int64_t batchIdx = i / this->tileNumPerBatch;
            int64_t tileIdx = i % this->tileNumPerBatch;
            if (tileIdx == this->tileNumPerBatch - 1) {
                this->processDataNum = this->tailDataNum;
            }
            int64_t offset1 = (batchIdx * SPECIFIED_DIM_SIZE + IDX_0) * this->stepSize + tileIdx * this->tileDataNum;
            int64_t offset2 = (batchIdx * SPECIFIED_DIM_SIZE + IDX_1) * this->stepSize + tileIdx * this->tileDataNum;
            int64_t offset3 = (batchIdx * SPECIFIED_DIM_SIZE + IDX_2) * this->stepSize + tileIdx * this->tileDataNum;
            CopyIn(offset1, offset2, offset3);
            Compute();
            CopyOut(offset1, offset2, offset3);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset1, int64_t offset2, int64_t offset3)
    {
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> y1Local = inQueueY1.AllocTensor<T>();
        LocalTensor<T> z1Local = inQueueZ1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        LocalTensor<T> y2Local = inQueueY2.AllocTensor<T>();
        LocalTensor<T> z2Local = inQueueZ2.AllocTensor<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x1Local, x1Gm[offset1], copyParams, padParams);
        DataCopyPad(y1Local, x1Gm[offset2], copyParams, padParams);
        DataCopyPad(z1Local, x1Gm[offset3], copyParams, padParams);
        DataCopyPad(x2Local, x2Gm[offset1], copyParams, padParams);
        DataCopyPad(y2Local, x2Gm[offset2], copyParams, padParams);
        DataCopyPad(z2Local, x2Gm[offset3], copyParams, padParams);

        inQueueX1.EnQue(x1Local);
        inQueueY1.EnQue(y1Local);
        inQueueZ1.EnQue(z1Local);
        inQueueX2.EnQue(x2Local);
        inQueueY2.EnQue(y2Local);
        inQueueZ2.EnQue(z2Local);
    }

    __aicore__ inline void ComputeBit16(
        const LocalTensor<T>& x1Local, const LocalTensor<T>& y1Local, const LocalTensor<T>& z1Local,
        const LocalTensor<T>& x2Local, const LocalTensor<T>& y2Local, const LocalTensor<T>& z2Local,
        const LocalTensor<T>& x3Local, const LocalTensor<T>& y3Local, const LocalTensor<T>& z3Local)
    {
        LocalTensor<float> xayb = bufXaYb.Get<float>();
        LocalTensor<float> xbya = bufXbYa.Get<float>();
        LocalTensor<float> x1 = x1Buf.Get<float>();
        LocalTensor<float> y1 = y1Buf.Get<float>();
        LocalTensor<float> z1 = z1Buf.Get<float>();
        LocalTensor<float> x2 = x2Buf.Get<float>();
        LocalTensor<float> y2 = y2Buf.Get<float>();
        LocalTensor<float> z2 = z2Buf.Get<float>();
        LocalTensor<float> x3 = x3Buf.Get<float>();
        LocalTensor<float> y3 = y3Buf.Get<float>();
        LocalTensor<float> z3 = z3Buf.Get<float>();
        Cast(x1, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(y1, y1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(z1, z1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(x2, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(y2, y2Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(z2, z2Local, RoundMode::CAST_NONE, this->processDataNum);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        Cast(x3Local, x3, RoundMode::CAST_RINT, this->processDataNum);
        Cast(y3Local, y3, RoundMode::CAST_RINT, this->processDataNum);
        Cast(z3Local, z3, RoundMode::CAST_RINT, this->processDataNum);
    }

    __aicore__ inline void ComputeBit8(
        const LocalTensor<T>& x1Local, const LocalTensor<T>& y1Local, const LocalTensor<T>& z1Local,
        const LocalTensor<T>& x2Local, const LocalTensor<T>& y2Local, const LocalTensor<T>& z2Local,
        const LocalTensor<T>& x3Local, const LocalTensor<T>& y3Local, const LocalTensor<T>& z3Local)
    {
        LocalTensor<half> xayb = bufXaYb.Get<half>();
        LocalTensor<half> xbya = bufXbYa.Get<half>();
        LocalTensor<half> x1 = x1Buf.Get<half>();
        LocalTensor<half> y1 = y1Buf.Get<half>();
        LocalTensor<half> z1 = z1Buf.Get<half>();
        LocalTensor<half> x2 = x2Buf.Get<half>();
        LocalTensor<half> y2 = y2Buf.Get<half>();
        LocalTensor<half> z2 = z2Buf.Get<half>();
        LocalTensor<half> x3 = x3Buf.Get<half>();
        LocalTensor<half> y3 = y3Buf.Get<half>();
        LocalTensor<half> z3 = z3Buf.Get<half>();
        Cast(x2, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(y2, y2Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(z2, z2Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(x1, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(y1, y1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(z1, z1Local, RoundMode::CAST_NONE, this->processDataNum);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        if constexpr (std::is_same_v<T, uint8_t>) {
            LocalTensor<uint8_t> maskX = x1Buf.Get<uint8_t>();
            LocalTensor<uint8_t> maskY = y1Buf.Get<uint8_t>();
            LocalTensor<uint8_t> maskZ = z1Buf.Get<uint8_t>();
            LocalTensor<half> x3Tmp = x2Buf.Get<half>();
            LocalTensor<half> y3Tmp = y2Buf.Get<half>();
            LocalTensor<half> z3Tmp = z2Buf.Get<half>();
            uint32_t calCount =
                RoundUp(static_cast<uint32_t>(this->processDataNum * sizeof(half)), UINT8_ALIGN_SIZE) / sizeof(half);
            CompareScalar(maskX, x3, HALF_ZERO, CMPMODE::LT, calCount);
            CompareScalar(maskY, y3, HALF_ZERO, CMPMODE::LT, calCount);
            CompareScalar(maskZ, z3, HALF_ZERO, CMPMODE::LT, calCount);
            Adds(x3Tmp, x3, HALF_256, this->processDataNum);
            Adds(y3Tmp, y3, HALF_256, this->processDataNum);
            Adds(z3Tmp, z3, HALF_256, this->processDataNum);
            Select(x3, maskX, x3Tmp, x3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            Select(y3, maskY, y3Tmp, y3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            Select(z3, maskZ, z3Tmp, z3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        }
        Cast(x3Local, x3, RoundMode::CAST_NONE, this->processDataNum);
        Cast(y3Local, y3, RoundMode::CAST_NONE, this->processDataNum);
        Cast(z3Local, z3, RoundMode::CAST_NONE, this->processDataNum);
    }

    __aicore__ inline void ComputeNotCast(
        const LocalTensor<T>& x1Local, const LocalTensor<T>& y1Local, const LocalTensor<T>& z1Local,
        const LocalTensor<T>& x2Local, const LocalTensor<T>& y2Local, const LocalTensor<T>& z2Local,
        const LocalTensor<T>& x3Local, const LocalTensor<T>& y3Local, const LocalTensor<T>& z3Local)
    {
        LocalTensor<T> xayb = bufXaYb.Get<T>();
        LocalTensor<T> xbya = bufXbYa.Get<T>();
        Mul(xayb, y1Local, z2Local, this->processDataNum);
        Mul(xbya, y2Local, z1Local, this->processDataNum);
        Sub(x3Local, xayb, xbya, this->processDataNum);
        Mul(xayb, x2Local, z1Local, this->processDataNum);
        Mul(xbya, x1Local, z2Local, this->processDataNum);
        Sub(y3Local, xayb, xbya, this->processDataNum);
        Mul(xayb, x1Local, y2Local, this->processDataNum);
        Mul(xbya, x2Local, y1Local, this->processDataNum);
        Sub(z3Local, xayb, xbya, this->processDataNum);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> y1Local = inQueueY1.DeQue<T>();
        LocalTensor<T> z1Local = inQueueZ1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<T> y2Local = inQueueY2.DeQue<T>();
        LocalTensor<T> z2Local = inQueueZ2.DeQue<T>();
        LocalTensor<T> x3Local = outQueueX3.AllocTensor<T>();
        LocalTensor<T> y3Local = outQueueY3.AllocTensor<T>();
        LocalTensor<T> z3Local = outQueueZ3.AllocTensor<T>();

        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            ComputeBit16(x1Local, y1Local, z1Local, x2Local, y2Local, z2Local, x3Local, y3Local, z3Local);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            ComputeBit8(x1Local, y1Local, z1Local, x2Local, y2Local, z2Local, x3Local, y3Local, z3Local);
        } else {
            ComputeNotCast(x1Local, y1Local, z1Local, x2Local, y2Local, z2Local, x3Local, y3Local, z3Local);
        }
        outQueueX3.EnQue<T>(x3Local);
        outQueueY3.EnQue<T>(y3Local);
        outQueueZ3.EnQue<T>(z3Local);
        inQueueX1.FreeTensor(x1Local);
        inQueueY1.FreeTensor(y1Local);
        inQueueZ1.FreeTensor(z1Local);
        inQueueX2.FreeTensor(x2Local);
        inQueueY2.FreeTensor(y2Local);
        inQueueZ2.FreeTensor(z2Local);
    }

    __aicore__ inline void CopyOut(int64_t offset1, int64_t offset2, int64_t offset3)
    {
        LocalTensor<T> x3Local = outQueueX3.DeQue<T>();
        LocalTensor<T> y3Local = outQueueY3.DeQue<T>();
        LocalTensor<T> z3Local = outQueueZ3.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[offset1], x3Local, copyParams);
        DataCopyPad(yGm[offset2], y3Local, copyParams);
        DataCopyPad(yGm[offset3], z3Local, copyParams);
        outQueueX3.FreeTensor(x3Local);
        outQueueY3.FreeTensor(y3Local);
        outQueueZ3.FreeTensor(z3Local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueZ1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueZ2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueX3;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY3;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ3;
    TBuf<QuePosition::VECCALC> bufXaYb;
    TBuf<QuePosition::VECCALC> bufXbYa;
    TBuf<QuePosition::VECCALC> x1Buf;
    TBuf<QuePosition::VECCALC> y1Buf;
    TBuf<QuePosition::VECCALC> z1Buf;
    TBuf<QuePosition::VECCALC> x2Buf;
    TBuf<QuePosition::VECCALC> y2Buf;
    TBuf<QuePosition::VECCALC> z2Buf;
    TBuf<QuePosition::VECCALC> x3Buf;
    TBuf<QuePosition::VECCALC> y3Buf;
    TBuf<QuePosition::VECCALC> z3Buf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;
    int64_t stepSize;
    int64_t tileNum;
    int64_t tileNumPerBatch;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    uint32_t coreIdx;
    uint32_t coreNum;
};

template <typename T>
class KernelCrossOneStep
{
public:
    __aicore__ inline KernelCrossOneStep()
    {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int64_t tileNum, int64_t tileDataNum, int64_t tailDataNum)
    {
        this->coreIdx = GetBlockIdx();
        this->coreNum = GetBlockNum();
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        this->tileNum = tileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = tailDataNum;

        pipe.InitBuffer(inQueueSelf, BUFFER_NUM, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueOther, BUFFER_NUM, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(T));
        pipe.InitBuffer(srcOffsetBuf, SPECIFIED_DIM_SIZE * BLOCK_SIZE);
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int16_t> || std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(selfBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(float));
            pipe.InitBuffer(otherBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(float));
            pipe.InitBuffer(outBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(float));
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(selfBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(half));
            pipe.InitBuffer(otherBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(half));
            pipe.InitBuffer(outBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(half));
        } else {
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t>) {
            LocalTensor<uint16_t> xOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, 0);
            LocalTensor<uint16_t> yOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, BLOCK_SIZE);
            LocalTensor<uint16_t> zOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, IDX_2 * BLOCK_SIZE);
            xOffset.SetValue(IDX_0, GATHER_MASK_UINT16_0);
            xOffset.SetValue(IDX_1, GATHER_MASK_UINT16_2);
            xOffset.SetValue(IDX_2, GATHER_MASK_UINT16_1);
            yOffset.SetValue(IDX_0, GATHER_MASK_UINT16_1);
            yOffset.SetValue(IDX_1, GATHER_MASK_UINT16_0);
            yOffset.SetValue(IDX_2, GATHER_MASK_UINT16_2);
            zOffset.SetValue(IDX_0, GATHER_MASK_UINT16_2);
            zOffset.SetValue(IDX_1, GATHER_MASK_UINT16_1);
            zOffset.SetValue(IDX_2, GATHER_MASK_UINT16_0);
        } else {
            LocalTensor<uint32_t> xOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, 0);
            LocalTensor<uint32_t> yOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, BLOCK_SIZE);
            LocalTensor<uint32_t> zOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, IDX_2 * BLOCK_SIZE);
            xOffset.SetValue(IDX_0, GATHER_MASK_UINT32_0);
            yOffset.SetValue(IDX_0, GATHER_MASK_UINT32_1);
            zOffset.SetValue(IDX_0, GATHER_MASK_UINT32_2);
        }

        int64_t loopCount = this->tileNum;
        for (int64_t i = coreIdx; i < loopCount; i += this->coreNum) {
            this->processDataNum = this->tileDataNum;
            int64_t batchIdx = i * this->tileDataNum;
            if (i == loopCount - 1) {
                this->processDataNum = this->tailDataNum;
            }
            int64_t offset = batchIdx * SPECIFIED_DIM_SIZE;
            CopyInOneStep(offset);
            ComputeOneStep();
            CopyOutOneStep(offset);
        }
    }

private:
    __aicore__ inline void CopyInOneStep(int64_t offset)
    {
        LocalTensor<T> selfLocal = inQueueSelf.AllocTensor<T>();
        LocalTensor<T> otherLocal = inQueueOther.AllocTensor<T>();

        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(SPECIFIED_DIM_SIZE * this->processDataNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(selfLocal, x1Gm[offset], copyParams, padParams);
        DataCopyPad(otherLocal, x2Gm[offset], copyParams, padParams);

        inQueueSelf.EnQue(selfLocal);
        inQueueOther.EnQue(otherLocal);
    }

    __aicore__ inline void ComputeBit16(
        const LocalTensor<T>& selfLocal, const LocalTensor<T>& otherLocal, const LocalTensor<T>& outLocal)
    {
        using U = float;
        LocalTensor<uint32_t> xOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, 0);
        LocalTensor<uint32_t> yOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, BLOCK_SIZE);
        LocalTensor<uint32_t> zOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, IDX_2 * BLOCK_SIZE);
        LocalTensor<U> self = selfBuf.Get<U>();
        LocalTensor<U> other = otherBuf.Get<U>();
        LocalTensor<U> out = outBuf.Get<U>();
        Cast(self, selfLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        Cast(other, otherLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        LocalTensor<U> x1 = x1Buf.Get<U>();
        LocalTensor<U> y1 = y1Buf.Get<U>();
        LocalTensor<U> z1 = z1Buf.Get<U>();
        LocalTensor<U> x2 = x2Buf.Get<U>();
        LocalTensor<U> y2 = y2Buf.Get<U>();
        LocalTensor<U> z2 = z2Buf.Get<U>();
        LocalTensor<U> x3 = x3Buf.Get<U>();
        LocalTensor<U> y3 = y3Buf.Get<U>();
        LocalTensor<U> z3 = z3Buf.Get<U>();
        LocalTensor<U> xayb = bufXaYb.Get<U>();
        LocalTensor<U> xbya = bufXbYa.Get<U>();
        uint8_t repeatTimes = this->tileDataNum / NUM_PER_BLK_32;
        uint32_t mask = SPECIFIED_DIM_SIZE * NUM_PER_BLK_32;
        uint64_t rsvdCnt = 0;
        GatherMask(x1, self, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y1, self, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z1, self, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(x2, other, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y2, other, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z2, other, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        for (uint32_t i = 0; i < this->processDataNum; i++) {
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_0, x3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_1, y3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_2, z3.GetValue(i));
        }
        Cast(outLocal, out, RoundMode::CAST_RINT, SPECIFIED_DIM_SIZE * this->processDataNum);
    }

    __aicore__ inline void ComputeBit8(
        const LocalTensor<T>& selfLocal, const LocalTensor<T>& otherLocal, const LocalTensor<T>& outLocal)
    {
        using U = half;
        LocalTensor<uint16_t> xOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, 0);
        LocalTensor<uint16_t> yOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, BLOCK_SIZE);
        LocalTensor<uint16_t> zOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, IDX_2 * BLOCK_SIZE);
        LocalTensor<U> self = selfBuf.Get<U>();
        LocalTensor<U> other = otherBuf.Get<U>();
        LocalTensor<U> out = outBuf.Get<U>();
        Cast(self, selfLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        Cast(other, otherLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        LocalTensor<U> x1 = x1Buf.Get<U>();
        LocalTensor<U> y1 = y1Buf.Get<U>();
        LocalTensor<U> z1 = z1Buf.Get<U>();
        LocalTensor<U> x3 = x3Buf.Get<U>();
        LocalTensor<U> y3 = y3Buf.Get<U>();
        LocalTensor<U> z3 = z3Buf.Get<U>();
        LocalTensor<U> x2 = x2Buf.Get<U>();
        LocalTensor<U> y2 = y2Buf.Get<U>();
        LocalTensor<U> z2 = z2Buf.Get<U>();
        LocalTensor<U> xayb = bufXaYb.Get<U>();
        LocalTensor<U> xbya = bufXbYa.Get<U>();
        uint8_t repeatTimes = this->tileDataNum / NUM_PER_BLK_16;
        uint32_t mask = SPECIFIED_DIM_SIZE * NUM_PER_BLK_16;
        uint64_t rsvdCnt = 0;
        GatherMask(x2, other, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y2, other, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z2, other, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(x1, self, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y1, self, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z1, self, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        if constexpr (std::is_same_v<T, uint8_t>) {
            LocalTensor<uint8_t> maskX = x1Buf.Get<uint8_t>();
            LocalTensor<uint8_t> maskY = y1Buf.Get<uint8_t>();
            LocalTensor<uint8_t> maskZ = z1Buf.Get<uint8_t>();
            LocalTensor<U> x3Tmp = x2Buf.Get<U>();
            LocalTensor<U> y3Tmp = y2Buf.Get<U>();
            LocalTensor<U> z3Tmp = z2Buf.Get<U>();
            uint32_t calCount =
                RoundUp(static_cast<uint32_t>(this->processDataNum * sizeof(U)), UINT8_ALIGN_SIZE) / sizeof(U);
            CompareScalar(maskX, x3, HALF_ZERO, CMPMODE::LT, calCount);
            CompareScalar(maskY, y3, HALF_ZERO, CMPMODE::LT, calCount);
            CompareScalar(maskZ, z3, HALF_ZERO, CMPMODE::LT, calCount);
            Adds(x3Tmp, x3, HALF_256, this->processDataNum);
            Adds(z3Tmp, z3, HALF_256, this->processDataNum);
            Adds(y3Tmp, y3, HALF_256, this->processDataNum);
            Select(x3, maskX, x3Tmp, x3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            Select(y3, maskY, y3Tmp, y3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            Select(z3, maskZ, z3Tmp, z3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        }
        for (uint32_t i = 0; i < this->processDataNum; i++) {
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_0, x3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_1, y3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_2, z3.GetValue(i));
        }
        Cast(outLocal, out, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
    }

    template <typename U>
    __aicore__ inline void ComputeNotCast(
        const LocalTensor<T>& selfLocal, const LocalTensor<T>& otherLocal, const LocalTensor<T>& outLocal)
    {
        uint32_t numPerBlock = BLOCK_SIZE / sizeof(U);
        LocalTensor<U> xOffset = srcOffsetBuf.GetWithOffset<U>(numPerBlock, 0);
        LocalTensor<U> yOffset = srcOffsetBuf.GetWithOffset<U>(numPerBlock, BLOCK_SIZE);
        LocalTensor<U> zOffset = srcOffsetBuf.GetWithOffset<U>(numPerBlock, IDX_2 * BLOCK_SIZE);
        LocalTensor<T> x1 = x1Buf.Get<T>();
        LocalTensor<T> y1 = y1Buf.Get<T>();
        LocalTensor<T> z1 = z1Buf.Get<T>();
        LocalTensor<T> x2 = x2Buf.Get<T>();
        LocalTensor<T> y2 = y2Buf.Get<T>();
        LocalTensor<T> z2 = z2Buf.Get<T>();
        LocalTensor<T> x3 = x3Buf.Get<T>();
        LocalTensor<T> y3 = y3Buf.Get<T>();
        LocalTensor<T> z3 = z3Buf.Get<T>();
        LocalTensor<T> xayb = bufXaYb.Get<T>();
        LocalTensor<T> xbya = bufXbYa.Get<T>();
        uint8_t repeatTimes = this->tileDataNum / numPerBlock;
        uint32_t mask = SPECIFIED_DIM_SIZE * numPerBlock;
        uint64_t rsvdCnt = 0;
        GatherMask(x1, selfLocal, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y1, selfLocal, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z1, selfLocal, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(x2, otherLocal, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y2, otherLocal, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z2, otherLocal, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        for (uint32_t i = 0; i < this->processDataNum; i++) {
            outLocal.SetValue(i * SPECIFIED_DIM_SIZE + IDX_0, x3.GetValue(i));
            outLocal.SetValue(i * SPECIFIED_DIM_SIZE + IDX_1, y3.GetValue(i));
            outLocal.SetValue(i * SPECIFIED_DIM_SIZE + IDX_2, z3.GetValue(i));
        }
    }

    __aicore__ inline void ComputeOneStep()
    {
        LocalTensor<T> selfLocal = inQueueSelf.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueOther.DeQue<T>();
        LocalTensor<T> outLocal = outQueueOut.AllocTensor<T>();
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            ComputeBit16(selfLocal, otherLocal, outLocal);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            ComputeBit8(selfLocal, otherLocal, outLocal);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            ComputeNotCast<uint16_t>(selfLocal, otherLocal, outLocal);
        } else {
            ComputeNotCast<uint32_t>(selfLocal, otherLocal, outLocal);
        }
        outQueueOut.EnQue<T>(outLocal);
        inQueueSelf.FreeTensor(selfLocal);
        inQueueOther.FreeTensor(otherLocal);
    }

    __aicore__ inline void CopyOutOneStep(int64_t offset)
    {
        LocalTensor<T> outLocal = outQueueOut.DeQue<T>();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(SPECIFIED_DIM_SIZE * this->processDataNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[offset], outLocal, copyParams);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueOther;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    TBuf<QuePosition::VECCALC> bufXaYb;
    TBuf<QuePosition::VECCALC> bufXbYa;
    TBuf<QuePosition::VECCALC> selfBuf;
    TBuf<QuePosition::VECCALC> otherBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    TBuf<QuePosition::VECCALC> x1Buf;
    TBuf<QuePosition::VECCALC> y1Buf;
    TBuf<QuePosition::VECCALC> z1Buf;
    TBuf<QuePosition::VECCALC> x2Buf;
    TBuf<QuePosition::VECCALC> y2Buf;
    TBuf<QuePosition::VECCALC> z2Buf;
    TBuf<QuePosition::VECCALC> x3Buf;
    TBuf<QuePosition::VECCALC> y3Buf;
    TBuf<QuePosition::VECCALC> z3Buf;
    TBuf<QuePosition::VECCALC> srcOffsetBuf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    uint32_t coreIdx;
    uint32_t coreNum;
};
} // namespace CrossV2NS
extern "C" __global__ __aicore__ void cross_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using namespace CrossV2NS;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        KernelCrossOneStep<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.tileNum, tiling_data.tileDataNum, tiling_data.tailDataNum);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelCross<DTYPE_X1> op;
        op.Init(
            x1, x2, y, tiling_data.stepSize, tiling_data.tileNum, tiling_data.tileNumPerBatch, tiling_data.tileDataNum,
            tiling_data.tailDataNum);
        op.Process();
    }
}