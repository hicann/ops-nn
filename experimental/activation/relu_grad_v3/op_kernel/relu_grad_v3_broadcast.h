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

#ifndef RELU_GRAD_V3_BROADCAST_H
#define RELU_GRAD_V3_BROADCAST_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_grad_v3_tiling_data.h"
#include "relu_grad_v3_normal.h"

namespace NsReluGradV3 {

template <typename T>
class ReluGradV3Broadcast {
public:
    __aicore__ inline ReluGradV3Broadcast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const ReluGradV3TilingData* tilingData, TPipe& pipeIn)
    {
        this->pipe = &pipeIn;
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreIdx = AscendC::GetBlockIdx();
        this->totalLength = tilingData->totalLength;
        this->ubPartDataNum = tilingData->ubPartDataNum;
        this->dimNum = tilingData->dimNum;
        this->xElementNum = tilingData->xElementNum;
        this->yElementNum = tilingData->yElementNum;
        for (uint32_t i = 0; i < 8; ++i) {
            this->outShape[i] = tilingData->outShape[i];
            this->xStrides[i] = tilingData->xStrides[i];
            this->yStrides[i] = tilingData->yStrides[i];
        }

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

        xGm.SetGlobalBuffer((__gm__ T*)x, this->xElementNum);
        yGm.SetGlobalBuffer((__gm__ T*)y, this->yElementNum);
        zGm.SetGlobalBuffer((__gm__ T*)z, this->totalLength);

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(xQueue, RELU_GRAD_V3_BUFFER_NUM, this->ubPartDataNum * sizeof(T));
            pipe->InitBuffer(yQueue, RELU_GRAD_V3_BUFFER_NUM, this->ubPartDataNum * sizeof(T));
            pipe->InitBuffer(zQueue, RELU_GRAD_V3_BUFFER_NUM, this->ubPartDataNum * sizeof(T));
            pipe->InitBuffer(tmpBuf0, this->ubPartDataNum * sizeof(uint8_t));
            pipe->InitBuffer(bufXFloat, this->ubPartDataNum * sizeof(float));
            pipe->InitBuffer(bufYFloat, this->ubPartDataNum * sizeof(float));
            pipe->InitBuffer(bufZFloat, this->ubPartDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            int32_t loopCount = static_cast<int32_t>(this->tileNum);
            for (int32_t i = 0; i < loopCount; ++i) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        } else {
            T zero = static_cast<T>(0);
            for (uint64_t i = 0; i < this->coreDataNum; ++i) {
                uint64_t outIndex = this->globalOffset + i;
                T xValue = xGm.GetValue(GetInputIndex(outIndex, this->xStrides));
                T yValue = yGm.GetValue(GetInputIndex(outIndex, this->yStrides));
                if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                    zGm.SetValue(outIndex, (xValue > zero) ? yValue : zero);
                } else {
                    zGm.SetValue(outIndex, (static_cast<float>(xValue) > 0.0f) ? yValue : zero);
                }
            }
        }
    }

private:
    __aicore__ inline uint64_t GetInputIndex(uint64_t outIndex, const uint64_t* strides)
    {
        uint64_t inputIndex = 0;
        for (int32_t i = static_cast<int32_t>(this->dimNum) - 1; i >= 0; --i) {
            uint64_t coord = outIndex % this->outShape[i];
            outIndex /= this->outShape[i];
            inputIndex += coord * strides[i];
        }
        return inputIndex;
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        this->processDataNum = (progress == static_cast<int32_t>(this->tileNum) - 1) ? this->tailDataNum :
                                                                                       this->ubPartDataNum;
        uint64_t offset = this->globalOffset + static_cast<uint64_t>(progress) * this->ubPartDataNum;
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
        for (uint64_t i = 0; i < this->processDataNum; ++i) {
            uint64_t outIndex = offset + i;
            xLocal.SetValue(i, xGm.GetValue(GetInputIndex(outIndex, this->xStrides)));
            yLocal.SetValue(i, yGm.GetValue(GetInputIndex(outIndex, this->yStrides)));
        }
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        this->processDataNum = (progress == static_cast<int32_t>(this->tileNum) - 1) ? this->tailDataNum :
                                                                                       this->ubPartDataNum;
        LocalTensor<T> xLocal = xQueue.DeQue<T>();
        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        LocalTensor<T> zLocal = zQueue.AllocTensor<T>();
        LocalTensor<uint8_t> sel = tmpBuf0.Get<uint8_t>();
        uint32_t n = static_cast<uint32_t>(this->processDataNum);
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
        zQueue.EnQue<T>(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        this->processDataNum = (progress == static_cast<int32_t>(this->tileNum) - 1) ? this->tailDataNum :
                                                                                       this->ubPartDataNum;
        uint64_t offset = this->globalOffset + static_cast<uint64_t>(progress) * this->ubPartDataNum;
        LocalTensor<T> zLocal = zQueue.DeQue<T>();
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        DataCopy(zGm[offset], zLocal, static_cast<uint32_t>(this->processDataNum));
        zQueue.FreeTensor(zLocal);
    }

    TPipe* pipe;
    TQue<QuePosition::VECIN, RELU_GRAD_V3_BUFFER_NUM> xQueue;
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
    uint64_t processDataNum;
    uint64_t dimNum;
    uint64_t xElementNum;
    uint64_t yElementNum;
    uint64_t outShape[8];
    uint64_t xStrides[8];
    uint64_t yStrides[8];
};

} // namespace NsReluGradV3

#endif // RELU_GRAD_V3_BROADCAST_H
