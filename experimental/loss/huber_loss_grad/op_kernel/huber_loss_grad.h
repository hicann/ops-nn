/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file huber_loss_grad.h
 * \brief HuberLossGrad算子Kernel实现
 *
 * 计算公式：
 * 设 e = predictions - targets
 * 当 |e| <= delta 时：grad = e
 * 当 |e| > delta 时：grad = delta * sign(e)
 */

#ifndef HUBER_LOSS_GRAD_H_
#define HUBER_LOSS_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "huber_loss_grad_tiling_data.h"
#include "huber_loss_grad_tiling_key.h"

namespace NsHuberLossGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelHuberLossGrad {
public:
    __aicore__ inline KernelHuberLossGrad() {}

    __aicore__ inline void Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR grad_output, GM_ADDR workspace,
                                const HuberLossGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> predictionsQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> targetsQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gradOutputQueue;

    GlobalTensor<T> predictionsGM;
    GlobalTensor<T> targetsGM;
    GlobalTensor<T> gradOutputGM;

    TBuf<TPosition::VECCALC> diffBuf;
    TBuf<TPosition::VECCALC> absBuf;
    TBuf<TPosition::VECCALC> signBuf;
    TBuf<TPosition::VECCALC> gradLargeBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> signTmpBuf;

    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
    float delta = 1.0f;
};

template <typename T>
__aicore__ inline void KernelHuberLossGrad<T>::Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR grad_output,
                                                    GM_ADDR workspace, const HuberLossGradTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->delta = tilingData->delta;
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * coreIdx;
    this->tileDataNum = tilingData->tileDataNum;

    if (coreIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (coreIdx - tilingData->tailBlockNum);
    }

    predictionsGM.SetGlobalBuffer((__gm__ T*)predictions + globalBufferIndex, this->coreDataNum);
    targetsGM.SetGlobalBuffer((__gm__ T*)targets + globalBufferIndex, this->coreDataNum);
    gradOutputGM.SetGlobalBuffer((__gm__ T*)grad_output + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(predictionsQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(targetsQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(gradOutputQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(diffBuf, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(absBuf, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(signBuf, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(gradLargeBuf, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(maskBuf, this->tileDataNum * sizeof(uint8_t));
    // signTmpBuf大小由host侧GetSignMaxMinTmpSize计算并通过tiling下发，不再硬编码sizeof(T)
    pipe.InitBuffer(signTmpBuf, tilingData->signTmpSize);
}

template <typename T>
__aicore__ inline void KernelHuberLossGrad<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

template <typename T>
__aicore__ inline void KernelHuberLossGrad<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> predictionsLocal = predictionsQueue.AllocTensor<T>();
    LocalTensor<T> targetsLocal = targetsQueue.AllocTensor<T>();
    // 使用DataCopyPad支持尾部tile非32B对齐搬运，blockLen按实际元素字节数精确搬运
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(predictionsLocal, predictionsGM[progress * this->tileDataNum], copyParams, padParams);
    DataCopyPad(targetsLocal, targetsGM[progress * this->tileDataNum], copyParams, padParams);
    predictionsQueue.EnQue(predictionsLocal);
    targetsQueue.EnQue(targetsLocal);
}

template <typename T>
__aicore__ inline void KernelHuberLossGrad<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> gradOutputLocal = gradOutputQueue.DeQue<T>();
    // 使用DataCopyPad支持尾部tile非32B对齐搬运，仅写回实际有效元素，避免覆盖后续数据
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(gradOutputGM[progress * this->tileDataNum], gradOutputLocal, copyParams);
    gradOutputQueue.FreeTensor(gradOutputLocal);
}

template <typename T>
__aicore__ inline void KernelHuberLossGrad<T>::Compute(int32_t progress)
{
    LocalTensor<T> predictionsLocal = predictionsQueue.DeQue<T>();
    LocalTensor<T> targetsLocal = targetsQueue.DeQue<T>();
    LocalTensor<T> gradOutputLocal = gradOutputQueue.AllocTensor<T>();

    LocalTensor<T> diff = diffBuf.Get<T>();
    LocalTensor<T> absDiff = absBuf.Get<T>();
    LocalTensor<T> signVal = signBuf.Get<T>();
    LocalTensor<T> gradLarge = gradLargeBuf.Get<T>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> signTmp = signTmpBuf.Get<uint8_t>();

    // e = predictions - targets
    Sub(diff, predictionsLocal, targetsLocal, this->processDataNum);

    // |e|
    Abs(absDiff, diff, this->processDataNum);

    // sign(e)
    Sign(signVal, diff, signTmp, this->processDataNum);

    // mask = |e| <= delta
    constexpr uint64_t maxMaskFloat32 = 64;
    constexpr uint64_t maxMaskFloat16 = 128;
    uint64_t maxMask = (sizeof(T) == 2) ? maxMaskFloat16 : maxMaskFloat32;
    uint64_t mask0 = (this->processDataNum >= maxMask) ? maxMask : this->processDataNum;
    int32_t repeat0 = (this->processDataNum + mask0 - 1) / mask0;
    UnaryRepeatParams repeatParams0 = {1, 1, 8, 8};
    CompareScalar(mask, absDiff, static_cast<T>(this->delta), CMPMODE::LE, mask0, repeat0, repeatParams0);

    // gradLarge = delta * sign(e)
    Muls(gradLarge, signVal, static_cast<T>(this->delta), this->processDataNum);

    // Select: mask非零时选diff，否则选gradLarge
    uint64_t mask1 = (this->processDataNum >= maxMask) ? maxMask : this->processDataNum;
    int32_t repeat1 = (this->processDataNum + mask1 - 1) / mask1;
    BinaryRepeatParams repeatParams1 = {1, 1, 1, 8, 8, 8};
    Select(gradOutputLocal, mask, diff, gradLarge, SELMODE::VSEL_TENSOR_TENSOR_MODE, mask1, repeat1, repeatParams1);

    gradOutputQueue.EnQue(gradOutputLocal);
    predictionsQueue.FreeTensor(predictionsLocal);
    targetsQueue.FreeTensor(targetsLocal);
}

} // namespace NsHuberLossGrad

// bfloat16_t 特化：IO用bf16，计算用float32
template <>
class NsHuberLossGrad::KernelHuberLossGrad<bfloat16_t> {
public:
    __aicore__ inline KernelHuberLossGrad() {}

    __aicore__ inline void Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR grad_output, GM_ADDR workspace,
                                const HuberLossGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> predictionsQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> targetsQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gradOutputQueue;

    GlobalTensor<bfloat16_t> predictionsGM;
    GlobalTensor<bfloat16_t> targetsGM;
    GlobalTensor<bfloat16_t> gradOutputGM;

    TBuf<TPosition::VECCALC> predFloatBuf;
    TBuf<TPosition::VECCALC> targetsFloatBuf;
    TBuf<TPosition::VECCALC> diffBuf;
    TBuf<TPosition::VECCALC> absBuf;
    TBuf<TPosition::VECCALC> signBuf;
    TBuf<TPosition::VECCALC> gradLargeBuf;
    TBuf<TPosition::VECCALC> gradOutputFloatBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> signTmpBuf;

    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
    float delta = 1.0f;
};

__aicore__ inline void NsHuberLossGrad::KernelHuberLossGrad<bfloat16_t>::Init(GM_ADDR predictions, GM_ADDR targets,
                                                                              GM_ADDR grad_output, GM_ADDR workspace,
                                                                              const HuberLossGradTilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->delta = tilingData->delta;
    uint32_t coreIdx = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * coreIdx;
    this->tileDataNum = tilingData->tileDataNum;

    if (coreIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (coreIdx - tilingData->tailBlockNum);
    }

    predictionsGM.SetGlobalBuffer((__gm__ bfloat16_t*)predictions + globalBufferIndex, this->coreDataNum);
    targetsGM.SetGlobalBuffer((__gm__ bfloat16_t*)targets + globalBufferIndex, this->coreDataNum);
    gradOutputGM.SetGlobalBuffer((__gm__ bfloat16_t*)grad_output + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(predictionsQueue, BUFFER_NUM, this->tileDataNum * sizeof(bfloat16_t));
    pipe.InitBuffer(targetsQueue, BUFFER_NUM, this->tileDataNum * sizeof(bfloat16_t));
    pipe.InitBuffer(gradOutputQueue, BUFFER_NUM, this->tileDataNum * sizeof(bfloat16_t));

    pipe.InitBuffer(predFloatBuf, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(targetsFloatBuf, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(diffBuf, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(absBuf, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(signBuf, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(gradLargeBuf, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(gradOutputFloatBuf, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(maskBuf, this->tileDataNum * sizeof(uint8_t));
    // signTmpBuf大小由host侧GetSignMaxMinTmpSize计算并通过tiling下发，不再硬编码sizeof(uint8_t)
    pipe.InitBuffer(signTmpBuf, tilingData->signTmpSize);
}

__aicore__ inline void NsHuberLossGrad::KernelHuberLossGrad<bfloat16_t>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

__aicore__ inline void NsHuberLossGrad::KernelHuberLossGrad<bfloat16_t>::CopyIn(int32_t progress)
{
    LocalTensor<bfloat16_t> predictionsLocal = predictionsQueue.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t> targetsLocal = targetsQueue.AllocTensor<bfloat16_t>();
    // 使用DataCopyPad支持尾部tile非32B对齐搬运，blockLen按实际元素字节数精确搬运
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> padParams{true, 0, 0, 0};
    DataCopyPad(predictionsLocal, predictionsGM[progress * this->tileDataNum], copyParams, padParams);
    DataCopyPad(targetsLocal, targetsGM[progress * this->tileDataNum], copyParams, padParams);
    predictionsQueue.EnQue(predictionsLocal);
    targetsQueue.EnQue(targetsLocal);
}

__aicore__ inline void NsHuberLossGrad::KernelHuberLossGrad<bfloat16_t>::CopyOut(int32_t progress)
{
    LocalTensor<bfloat16_t> gradOutputLocal = gradOutputQueue.DeQue<bfloat16_t>();
    // 使用DataCopyPad支持尾部tile非32B对齐搬运，仅写回实际有效元素，避免覆盖后续数据
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPad(gradOutputGM[progress * this->tileDataNum], gradOutputLocal, copyParams);
    gradOutputQueue.FreeTensor(gradOutputLocal);
}

__aicore__ inline void NsHuberLossGrad::KernelHuberLossGrad<bfloat16_t>::Compute(int32_t progress)
{
    LocalTensor<bfloat16_t> predictionsLocal = predictionsQueue.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> targetsLocal = targetsQueue.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> gradOutputLocal = gradOutputQueue.AllocTensor<bfloat16_t>();

    LocalTensor<float> predFloat = predFloatBuf.Get<float>();
    LocalTensor<float> targetsFloat = targetsFloatBuf.Get<float>();
    LocalTensor<float> diff = diffBuf.Get<float>();
    LocalTensor<float> absDiff = absBuf.Get<float>();
    LocalTensor<float> signVal = signBuf.Get<float>();
    LocalTensor<float> gradLarge = gradLargeBuf.Get<float>();
    LocalTensor<float> gradOutputFloat = gradOutputFloatBuf.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> signTmp = signTmpBuf.Get<uint8_t>();

    Cast(predFloat, predictionsLocal, RoundMode::CAST_NONE, this->processDataNum);
    Cast(targetsFloat, targetsLocal, RoundMode::CAST_NONE, this->processDataNum);

    Sub(diff, predFloat, targetsFloat, this->processDataNum);

    Abs(absDiff, diff, this->processDataNum);

    Sign(signVal, diff, signTmp, this->processDataNum);

    constexpr uint64_t maxMaskFloat32 = 64;
    uint64_t mask0 = (this->processDataNum >= maxMaskFloat32) ? maxMaskFloat32 : this->processDataNum;
    int32_t repeat0 = (this->processDataNum + mask0 - 1) / mask0;
    UnaryRepeatParams repeatParams0 = {1, 1, 8, 8};
    CompareScalar(mask, absDiff, static_cast<float>(this->delta), CMPMODE::LE, mask0, repeat0, repeatParams0);

    Muls(gradLarge, signVal, static_cast<float>(this->delta), this->processDataNum);

    uint64_t mask1 = (this->processDataNum >= maxMaskFloat32) ? maxMaskFloat32 : this->processDataNum;
    int32_t repeat1 = (this->processDataNum + mask1 - 1) / mask1;
    BinaryRepeatParams repeatParams1 = {1, 1, 1, 8, 8, 8};
    Select(gradOutputFloat, mask, diff, gradLarge, SELMODE::VSEL_TENSOR_TENSOR_MODE, mask1, repeat1, repeatParams1);

    Cast(gradOutputLocal, gradOutputFloat, RoundMode::CAST_RINT, this->processDataNum);

    gradOutputQueue.EnQue(gradOutputLocal);
    predictionsQueue.FreeTensor(predictionsLocal);
    targetsQueue.FreeTensor(targetsLocal);
}

#endif // HUBER_LOSS_GRAD_H_
