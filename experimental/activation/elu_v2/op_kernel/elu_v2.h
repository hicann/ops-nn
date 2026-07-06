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
 * \file elu_v2.h
 * \brief
 */
#ifndef __ELU_V2_H__
#define __ELU_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "elu_v2_tiling_data.h"
#include "elu_v2_tiling_key.h"

namespace NsEluV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class EluV2 {
public:
    __aicore__ inline EluV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t bigCoreDataNum, uint32_t smallCoreDataNum,
                                uint32_t tileDataNum, uint32_t bigCoreNum, float alpha, float scale, float inputScale);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t processDataNum);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t processDataNum);
    __aicore__ inline void Compute(uint32_t processDataNum);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> selMaskBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> castXBuf, castYBuf;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t coreDataNum;
    uint32_t tileDataNum;
    float alpha, scale, inputScale;
    float scale_alpha;
};

template <typename T>
__aicore__ inline void EluV2<T>::Init(GM_ADDR x, GM_ADDR y, uint32_t bigCoreDataNum, uint32_t smallCoreDataNum,
                                      uint32_t tileDataNum, uint32_t bigCoreNum, float alpha, float scale,
                                      float inputScale)
{
    int64_t coreIdx = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = bigCoreDataNum * coreIdx;
    if (coreIdx < bigCoreNum) {
        this->coreDataNum = bigCoreDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreIdx - bigCoreNum);
    }
    this->tileDataNum = tileDataNum;

    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(selMaskBuf, this->tileDataNum * sizeof(uint8_t));

    this->alpha = alpha;
    this->scale = scale;
    this->inputScale = inputScale;
    this->scale_alpha = this->scale * this->alpha;

    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(castXBuf, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(castYBuf, this->tileDataNum * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void EluV2<T>::CopyIn(uint32_t offset, uint32_t processDataNum)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[offset], processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void EluV2<T>::CopyOut(uint32_t offset, uint32_t processDataNum)
{
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::DataCopy(yGm[offset], yLocal, processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void EluV2<T>::Compute(uint32_t processDataNum)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> selMask = selMaskBuf.Get<uint8_t>();
    if constexpr (std::is_same_v<T, float>) {
        AscendC::CompareScalar(selMask, xLocal, 0.0f, AscendC::CMPMODE::GE, processDataNum);
        AscendC::Muls(yLocal, xLocal, inputScale, processDataNum);
        AscendC::Exp(yLocal, yLocal, processDataNum);
        AscendC::Adds(yLocal, yLocal, -1.0f, processDataNum);
        AscendC::Muls(yLocal, yLocal, scale_alpha, processDataNum); // yLocal: resNeg
        AscendC::Muls(xLocal, xLocal, scale, processDataNum);       // xLocal: resPos
        AscendC::Select(yLocal, selMask, xLocal, yLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, processDataNum);
    } else {
        AscendC::LocalTensor<float> xCast = castXBuf.Get<float>();
        AscendC::LocalTensor<float> yCast = castYBuf.Get<float>();

        AscendC::Cast(xCast, xLocal, AscendC::RoundMode::CAST_NONE, processDataNum);
        AscendC::CompareScalar(selMask, xCast, 0.0f, AscendC::CMPMODE::GE, processDataNum);
        AscendC::Muls(yCast, xCast, inputScale, processDataNum);
        AscendC::Exp(yCast, yCast, processDataNum);
        AscendC::Adds(yCast, yCast, -1.0f, processDataNum);
        AscendC::Muls(yCast, yCast, scale_alpha, processDataNum); // yCast: resNeg
        AscendC::Muls(xCast, xCast, scale, processDataNum);       // xCast: resPos
        AscendC::Select(yCast, selMask, xCast, yCast, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, processDataNum);
        AscendC::Cast(yLocal, yCast, AscendC::RoundMode::CAST_RINT, processDataNum);
    }
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
template <typename T>
__aicore__ inline void EluV2<T>::Process()
{
    uint32_t coreDataNum = this->coreDataNum;
    uint32_t tileDataNum = this->tileDataNum;

    for (uint32_t offset = 0; offset < coreDataNum; offset += tileDataNum) {
        uint32_t processDataNum = tileDataNum < (coreDataNum - offset) ? tileDataNum : (coreDataNum - offset);
        CopyIn(offset, processDataNum);
        Compute(processDataNum);
        CopyOut(offset, processDataNum);
    }
}

} // namespace NsEluV2
#endif // ELU_V2_H
