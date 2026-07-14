/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RELU_GRAD_V3_NORMAL_H
#define RELU_GRAD_V3_NORMAL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_grad_v3_tiling_data.h"

namespace NsReluGradV3 {

using namespace AscendC;

constexpr int32_t RELU_GRAD_V3_BUFFER_NUM = 2;

template <typename T>
class ReluGradV3Normal {
public:
    __aicore__ inline ReluGradV3Normal() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const ReluGradV3TilingData* tilingData, TPipe& pipeIn)
    {
        this->pipe = &pipeIn;
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreIdx = AscendC::GetBlockIdx();
        this->totalLength = tilingData->totalLength;
        this->ubPartDataNum = tilingData->ubPartDataNum;

        if (tilingData->tailBlockNum > 0) {
            if (coreIdx < tilingData->tailBlockNum) {
                this->coreDataNum = tilingData->bigCoreDataNum;
                this->tileNum = tilingData->bigCoreLoopNum;
                this->tailDataNum = tilingData->bigCoreTailDataNum;
                this->globalOffset = coreIdx * tilingData->bigCoreDataNum;
            } else {
                this->coreDataNum = tilingData->smallCoreDataNum;
                this->tileNum = tilingData->smallCoreLoopNum;
                this->tailDataNum = tilingData->smallCoreTailDataNum;
                this->globalOffset = tilingData->tailBlockNum * tilingData->bigCoreDataNum +
                                     (coreIdx - tilingData->tailBlockNum) * tilingData->smallCoreDataNum;
            }
        } else {
            this->coreDataNum = tilingData->smallCoreDataNum;
            this->tileNum = tilingData->smallCoreLoopNum;
            this->tailDataNum = tilingData->smallCoreTailDataNum;
            this->globalOffset = coreIdx * tilingData->smallCoreDataNum;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ T*)y, this->totalLength);
        zGm.SetGlobalBuffer((__gm__ T*)z, this->totalLength);

        pipe->InitBuffer(xzQueue, RELU_GRAD_V3_BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        pipe->InitBuffer(yQueue, RELU_GRAD_V3_BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, half>) {
            pipe->InitBuffer(zQueue, RELU_GRAD_V3_BUFFER_NUM, this->ubPartDataNum * sizeof(T));
        }
        if constexpr (std::is_same_v<T, float>) {
            pipe->InitBuffer(tmpBuf0, this->ubPartDataNum * sizeof(uint8_t));
        } else if constexpr (std::is_same_v<T, half>) {
            pipe->InitBuffer(tmpBuf0, this->ubPartDataNum * sizeof(uint8_t));
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(tmpBuf0, this->ubPartDataNum * sizeof(uint8_t));
            pipe->InitBuffer(bufXFloat, this->ubPartDataNum * sizeof(float));
            pipe->InitBuffer(bufYFloat, this->ubPartDataNum * sizeof(float));
            pipe->InitBuffer(bufZFloat, this->ubPartDataNum * sizeof(float));
        } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
            pipe->InitBuffer(bufXFloat, this->ubPartDataNum * sizeof(half));
            pipe->InitBuffer(bufYFloat, this->ubPartDataNum * sizeof(half));
        }
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = static_cast<int32_t>(this->tileNum);
        uint64_t offset = this->globalOffset;
        uint32_t fullTileDataNum = static_cast<uint32_t>(this->ubPartDataNum);
        for (int32_t i = 0; i < loopCount - 1; ++i) {
            CopyIn(offset, fullTileDataNum);
            Compute(fullTileDataNum);
            CopyOut(offset, fullTileDataNum);
            offset += this->ubPartDataNum;
        }
        uint32_t tailTileDataNum = static_cast<uint32_t>(this->tailDataNum);
        CopyIn(offset, tailTileDataNum);
        Compute(tailTileDataNum);
        CopyOut(offset, tailTileDataNum);
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t dataNum)
    {
        LocalTensor<T> xLocal = xzQueue.AllocTensor<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
        DataCopy(xLocal, xGm[offset], dataNum);
        DataCopy(yLocal, yGm[offset], dataNum);
        xzQueue.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(xLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t dataNum)
    {
        LocalTensor<T> xLocal = xzQueue.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        LocalTensor<T> zLocal = xLocal;

        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, half>) {
            zLocal = zQueue.AllocTensor<T>();
        }

        if constexpr (std::is_same_v<T, int32_t>) {
            Duplicate(zLocal, static_cast<int32_t>(1), dataNum);
            Min(zLocal, xLocal, zLocal, dataNum);
            Duplicate(xLocal, static_cast<int32_t>(0), dataNum);
            Max(zLocal, zLocal, xLocal, dataNum);
            Mul(zLocal, yLocal, zLocal, dataNum);
        } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
            LocalTensor<half> xHalf = bufXFloat.Get<half>();
            LocalTensor<half> yHalf = bufYFloat.Get<half>();
            Cast(xHalf, xLocal, RoundMode::CAST_NONE, dataNum);
            Duplicate(yHalf, static_cast<half>(0.000000059604644775390625), dataNum);
            Min(xHalf, xHalf, yHalf, dataNum);
            Duplicate(yHalf, static_cast<half>(0.0), dataNum);
            Max(yHalf, xHalf, yHalf, dataNum);
            Duplicate(xHalf, static_cast<half>(4096), dataNum);
            Mul(yHalf, xHalf, yHalf, dataNum);
            Mul(yHalf, yHalf, xHalf, dataNum);
            Cast(xHalf, yLocal, RoundMode::CAST_NONE, dataNum);
            Mul(yHalf, yHalf, xHalf, dataNum);
            Cast(zLocal, yHalf, RoundMode::CAST_RINT, dataNum);
            bufXFloat.FreeTensor(xHalf);
            bufYFloat.FreeTensor(yHalf);
        } else if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint8_t> sel = tmpBuf0.Get<uint8_t>();
            T zero = 0;
            Compares(sel, xLocal, zero, CMPMODE::GT, dataNum);
            Select(zLocal, sel, yLocal, zero, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataNum);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            LocalTensor<uint8_t> sel = tmpBuf0.Get<uint8_t>();
            uint32_t n = dataNum;
            LocalTensor<float> xf = bufXFloat.Get<float>();
            LocalTensor<float> yf = bufYFloat.Get<float>();
            LocalTensor<float> zf = bufZFloat.Get<float>();
            Cast(xf, xLocal, RoundMode::CAST_NONE, n);
            Cast(yf, yLocal, RoundMode::CAST_NONE, n);
            Compares(sel, xf, 0.0f, CMPMODE::GT, n);
            Select(zf, sel, yf, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
            Cast(zLocal, zf, RoundMode::CAST_RINT, n);
            bufXFloat.FreeTensor(xf);
            bufYFloat.FreeTensor(yf);
            bufZFloat.FreeTensor(zf);
        } else {
            LocalTensor<uint8_t> sel = tmpBuf0.Get<uint8_t>();
            T zero = 0;
            uint64_t mask = dataNum >= 64 ? 64 : dataNum;
            int32_t repeat = static_cast<int32_t>((dataNum + mask - 1) / mask);
            Compares(sel, xLocal, zero, CMPMODE::GT, mask, repeat, UnaryRepeatParams{1, 1, 8, 8});
            Select(zLocal, sel, yLocal, zero, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, repeat,
                   BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        }

        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, half>) {
            zQueue.EnQue<T>(zLocal);
            xzQueue.FreeTensor(xLocal);
        } else {
            xzQueue.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(zLocal);
        }
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataNum)
    {
        LocalTensor<T> zLocal;
        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, half>) {
            zLocal = zQueue.DeQue<T>();
        } else {
            zLocal = xzQueue.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
        }
        DataCopy(zGm[offset], zLocal, dataNum);
        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, half>) {
            zQueue.FreeTensor(zLocal);
        } else {
            xzQueue.FreeTensor(zLocal);
        }
    }

    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, RELU_GRAD_V3_BUFFER_NUM> xzQueue;
    TQue<QuePosition::VECIN, RELU_GRAD_V3_BUFFER_NUM> yQueue;
    TQue<QuePosition::VECOUT, RELU_GRAD_V3_BUFFER_NUM> zQueue;
    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> bufXFloat;
    TBuf<QuePosition::VECCALC> bufYFloat;
    TBuf<QuePosition::VECCALC> bufZFloat;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    uint64_t totalLength;
    uint64_t globalOffset;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
};

} // namespace NsReluGradV3

#endif // RELU_GRAD_V3_NORMAL_H
