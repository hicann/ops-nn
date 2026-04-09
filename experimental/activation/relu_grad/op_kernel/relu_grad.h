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
 * \file relu_grad.h
 * \brief
 */
#ifndef RELU_GRAD_H
#define RELU_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_grad_tiling_data.h"
#include "relu_grad_tiling_key.h"

#include "kernel_operator.h"

// 需要的常量 命名规则为scalar_输入数据的数据类型_参数名
constexpr float scalar_float = 0.0;
constexpr float scalar_half = 0.0;
constexpr float scalar_bf16_A = 0.000000000000000000000000000000000000011754943508222875079687365372;
constexpr float scalar_bf16_B = 0.0;
constexpr int scalar_bf16_C = 274877906944;
constexpr int scalar_bf16_D = 17592186044416;
constexpr float scalar_uint8orint8_A = 0.000000059604644775390625;
constexpr float scalar_uint8orint8_B = 0.0;
constexpr int scalar_uint8orint8_C = 4096;
constexpr int scalar_int32_A = 0;
constexpr int scalar_int32_B = 1;

namespace NsReluGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_GRADIENTS>
class KernelReluGrad {
public:
    __aicore__ inline KernelReluGrad(){};

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue1, tmpQueue2, tmpQueueMask1;

    AscendC::GlobalTensor<TYPE_GRADIENTS> xGm, yGm, outGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelReluGrad<TYPE_GRADIENTS>::Init(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_GRADIENTS *)gradients + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_GRADIENTS *)features + globalBufferIndex, this->coreDataNum);
    outGm.SetGlobalBuffer((__gm__ TYPE_GRADIENTS *)backprops + globalBufferIndex, this->coreDataNum);
    if constexpr (std::is_same_v<TYPE_GRADIENTS, float32_t> || std::is_same_v<TYPE_GRADIENTS, float16_t>) {
        pipe.InitBuffer(tmpQueueMask1, this->tileDataNum * sizeof(uint8_t));
    } else if constexpr (std::is_same_v<TYPE_GRADIENTS, bfloat16_t>) {
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(float));
    } else if constexpr (std::is_same_v<TYPE_GRADIENTS, uint8_t> || std::is_same_v<TYPE_GRADIENTS, int8_t>) {
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(half));
    }
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADIENTS));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADIENTS));
    pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_GRADIENTS));
}

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelReluGrad<TYPE_GRADIENTS>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_GRADIENTS> xLocal = inQueueX.AllocTensor<TYPE_GRADIENTS>();
    AscendC::LocalTensor<TYPE_GRADIENTS> yLocal = inQueueY.AllocTensor<TYPE_GRADIENTS>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelReluGrad<TYPE_GRADIENTS>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_GRADIENTS> outLocal = outQueueOut.DeQue<TYPE_GRADIENTS>();
    AscendC::DataCopy(outGm[progress * this->tileDataNum], outLocal, this->processDataNum);
    outQueueOut.FreeTensor(outLocal);
}

template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelReluGrad<TYPE_GRADIENTS>::Compute(int32_t progress)
{
    if constexpr (std::is_same_v<TYPE_GRADIENTS, float32_t>) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask1.AllocTensor<uint8_t>();
        AscendC::CompareScalar(mask1Local, yLocal, static_cast<float>(scalar_float), AscendC::CMPMODE::LE, this->processDataNum);
        AscendC::Duplicate(yLocal, static_cast<float>(scalar_float), this->processDataNum);
        AscendC::Select(outLocal, mask1Local, yLocal, xLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        outQueueOut.EnQue<float>(outLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    } else {
        AscendC::LocalTensor<TYPE_GRADIENTS> xLocal = inQueueX.DeQue<TYPE_GRADIENTS>();
        AscendC::LocalTensor<TYPE_GRADIENTS> yLocal = inQueueY.DeQue<TYPE_GRADIENTS>();
        AscendC::LocalTensor<TYPE_GRADIENTS> outLocal = outQueueOut.AllocTensor<TYPE_GRADIENTS>();
        if constexpr (std::is_same_v<TYPE_GRADIENTS, float16_t>) {
            AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask1.AllocTensor<uint8_t>();
            AscendC::CompareScalar(mask1Local, yLocal, static_cast<half>(scalar_half), AscendC::CMPMODE::LE, this->processDataNum);
            AscendC::Duplicate(yLocal, static_cast<half>(scalar_half), this->processDataNum);
            AscendC::Select(outLocal, mask1Local, yLocal, xLocal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE_GRADIENTS, bfloat16_t>) {
            AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
            AscendC::Cast(tmp1Local, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Duplicate(tmp2Local, static_cast<float>(scalar_bf16_A), this->processDataNum);
            AscendC::Min(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
            AscendC::Duplicate(tmp2Local, static_cast<float>(scalar_bf16_B), this->processDataNum);
            AscendC::Max(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
            AscendC::Duplicate(tmp1Local, static_cast<float>(scalar_bf16_C), this->processDataNum);
            AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Duplicate(tmp1Local, static_cast<float>(scalar_bf16_D), this->processDataNum);
            AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Cast(tmp1Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Cast(outLocal, tmp2Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE_GRADIENTS, uint8_t> || std::is_same_v<TYPE_GRADIENTS, int8_t>) {
            AscendC::LocalTensor<half> tmp1Local = tmpQueue1.AllocTensor<half>();
            AscendC::LocalTensor<half> tmp2Local = tmpQueue2.AllocTensor<half>();
            AscendC::Cast(tmp1Local, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Duplicate(tmp2Local, static_cast<half>(scalar_uint8orint8_A), this->processDataNum);
            AscendC::Min(tmp1Local, tmp1Local, tmp2Local, this->processDataNum);
            AscendC::Duplicate(tmp2Local, static_cast<half>(scalar_uint8orint8_B), this->processDataNum);
            AscendC::Max(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
            AscendC::Duplicate(tmp1Local, static_cast<half>(scalar_uint8orint8_C), this->processDataNum);
            AscendC::Mul(tmp2Local, tmp1Local, tmp2Local, this->processDataNum);
            AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Cast(tmp1Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(tmp2Local, tmp2Local, tmp1Local, this->processDataNum);
            AscendC::Cast(outLocal, tmp2Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else {
            AscendC::Duplicate(outLocal, static_cast<int32_t>(scalar_int32_B), this->processDataNum);
            AscendC::Min(outLocal, yLocal, outLocal, this->processDataNum);
            AscendC::Duplicate(yLocal, static_cast<int32_t>(scalar_int32_A), this->processDataNum);
            AscendC::Max(outLocal, outLocal, yLocal, this->processDataNum);
            AscendC::Duplicate(yLocal, static_cast<int32_t>(scalar_int32_B), this->processDataNum);
            AscendC::Mul(outLocal, outLocal, yLocal, this->processDataNum);
            AscendC::Mul(outLocal, outLocal, yLocal, this->processDataNum);
            AscendC::Mul(outLocal, xLocal, outLocal, this->processDataNum);
        }
        outQueueOut.EnQue<TYPE_GRADIENTS>(outLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
}


template <typename TYPE_GRADIENTS>
__aicore__ inline void KernelReluGrad<TYPE_GRADIENTS>::Process()
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

} // namespace NsReluGrad
#endif // RELU_GRAD_H