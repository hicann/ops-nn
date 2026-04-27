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
 * \file threshold_grad_v2_d.h
 * \brief
 */
#ifndef THRESHOLD_GRAD_V2_D_H
#define THRESHOLD_GRAD_V2_D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "threshold_grad_v2_d_tiling_data.h"
#include "threshold_grad_v2_d_tiling_key.h"

#include "kernel_operator.h"

namespace NsThresholdGradV2D {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_INPUT_GRADIENT>
class KernelThresholdGradV2D {
public:
    __aicore__ inline KernelThresholdGradV2D(){};

    __aicore__ inline void Init(GM_ADDR input_gradient, GM_ADDR input_feature, GM_ADDR output_backprops, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float threshold);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueG, inQueueF;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueout;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue1, tmpQueue2, tmpQueueMask;

    AscendC::GlobalTensor<TYPE_INPUT_GRADIENT> input_gradientGm, input_featureGm, output_backpropsGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    float thresholdValue = 0.0f;
};

template <typename TYPE_INPUT_GRADIENT>
__aicore__ inline void KernelThresholdGradV2D<TYPE_INPUT_GRADIENT>::Init(GM_ADDR input_gradient, GM_ADDR input_feature, GM_ADDR output_backprops, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum, uint64_t tailBlockNum, float threshold)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    this->thresholdValue = threshold;
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
    input_gradientGm.SetGlobalBuffer((__gm__ TYPE_INPUT_GRADIENT *)input_gradient + globalBufferIndex, this->coreDataNum);
    input_featureGm.SetGlobalBuffer((__gm__ TYPE_INPUT_GRADIENT *)input_feature + globalBufferIndex, this->coreDataNum);
    output_backpropsGm.SetGlobalBuffer((__gm__ TYPE_INPUT_GRADIENT *)output_backprops + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueG, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_INPUT_GRADIENT));
    pipe.InitBuffer(inQueueF, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_INPUT_GRADIENT));
    pipe.InitBuffer(outQueueout, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_INPUT_GRADIENT));
    pipe.InitBuffer(tmpQueueMask, this->tileDataNum * sizeof(uint8_t));
    if (std::is_same_v<TYPE_INPUT_GRADIENT, int32_t> || std::is_same_v<TYPE_INPUT_GRADIENT, bfloat16_t>) {
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(float));
    } else if (std::is_same_v<TYPE_INPUT_GRADIENT, uint8_t> || std::is_same_v<TYPE_INPUT_GRADIENT, int8_t>) {
        pipe.InitBuffer(tmpQueue1, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(half));
    }
}

template <typename TYPE_INPUT_GRADIENT>
__aicore__ inline void KernelThresholdGradV2D<TYPE_INPUT_GRADIENT>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_INPUT_GRADIENT> gLocal = inQueueG.AllocTensor<TYPE_INPUT_GRADIENT>();
    AscendC::LocalTensor<TYPE_INPUT_GRADIENT> fLocal = inQueueF.AllocTensor<TYPE_INPUT_GRADIENT>();
    AscendC::DataCopy(gLocal, input_gradientGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(fLocal, input_featureGm[progress * this->tileDataNum], this->processDataNum);
    inQueueG.EnQue(gLocal);
    inQueueF.EnQue(fLocal);
}

template <typename TYPE_INPUT_GRADIENT>
__aicore__ inline void KernelThresholdGradV2D<TYPE_INPUT_GRADIENT>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_INPUT_GRADIENT> outLocal = outQueueout.DeQue<TYPE_INPUT_GRADIENT>();
    AscendC::DataCopy(output_backpropsGm[progress * this->tileDataNum], outLocal, this->processDataNum);
    outQueueout.FreeTensor(outLocal);
}

template <typename TYPE_INPUT_GRADIENT>
__aicore__ inline void KernelThresholdGradV2D<TYPE_INPUT_GRADIENT>::Compute(int32_t progress)
{
    if (std::is_same_v<TYPE_INPUT_GRADIENT, int8_t>) {
        AscendC::LocalTensor<int8_t> gLocal = inQueueG.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> fLocal = inQueueF.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> outLocal = outQueueout.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> tmp1Local = tmpQueue1.AllocTensor<half>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::Cast(tmp1Local, fLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp2Local, tmp1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp2Local, static_cast<float>(this->thresholdValue), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Cast(tmp1Local, gLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp2Local, tmp1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Select(tmp2Local, maskLocal, tmp2Local, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Cast(tmp1Local, tmp2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(outLocal, tmp1Local, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        outQueueout.EnQue<int8_t>(outLocal);
        inQueueG.FreeTensor(gLocal);
        inQueueF.FreeTensor(fLocal);
    } else if (std::is_same_v<TYPE_INPUT_GRADIENT, uint8_t>) {
        AscendC::LocalTensor<uint8_t> gLocal = inQueueG.DeQue<uint8_t>();
        AscendC::LocalTensor<uint8_t> fLocal = inQueueF.DeQue<uint8_t>();
        AscendC::LocalTensor<uint8_t> outLocal = outQueueout.AllocTensor<uint8_t>();
        AscendC::LocalTensor<half> tmp1Local = tmpQueue1.AllocTensor<half>();
        AscendC::LocalTensor<float> tmp2Local = tmpQueue2.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::Cast(tmp1Local, fLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp2Local, tmp1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp2Local, static_cast<float>(this->thresholdValue), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Cast(tmp1Local, gLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp2Local, tmp1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Select(tmp2Local, maskLocal, tmp2Local, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Cast(tmp1Local, tmp2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(outLocal, tmp1Local, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        outQueueout.EnQue<uint8_t>(outLocal);
        inQueueG.FreeTensor(gLocal);
        inQueueF.FreeTensor(fLocal);
    } else if (std::is_same_v<TYPE_INPUT_GRADIENT, int32_t>) {
        AscendC::LocalTensor<int32_t> gLocal = inQueueG.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> fLocal = inQueueF.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> outLocal = outQueueout.AllocTensor<int32_t>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::Cast(tmp1Local, fLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp1Local, static_cast<float>(this->thresholdValue), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Cast(tmp1Local, gLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Select(tmp1Local, maskLocal, tmp1Local, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Cast(outLocal, tmp1Local, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        outQueueout.EnQue<int32_t>(outLocal);
        inQueueG.FreeTensor(gLocal);
        inQueueF.FreeTensor(fLocal);
    } else if (std::is_same_v<TYPE_INPUT_GRADIENT, bfloat16_t>) {
        AscendC::LocalTensor<bfloat16_t> gLocal = inQueueG.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> fLocal = inQueueF.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> outLocal = outQueueout.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<float> tmp1Local = tmpQueue1.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::Cast(tmp1Local, fLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::CompareScalar(maskLocal, tmp1Local, static_cast<float>(this->thresholdValue), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Cast(tmp1Local, gLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Select(tmp1Local, maskLocal, tmp1Local, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        AscendC::Cast(outLocal, tmp1Local, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        outQueueout.EnQue<bfloat16_t>(outLocal);
        inQueueG.FreeTensor(gLocal);
        inQueueF.FreeTensor(fLocal);
    } else if (std::is_same_v<TYPE_INPUT_GRADIENT, float>) {
        AscendC::LocalTensor<float> gLocal = inQueueG.DeQue<float>();
        AscendC::LocalTensor<float> fLocal = inQueueF.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueout.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::CompareScalar(maskLocal, fLocal, static_cast<float>(this->thresholdValue), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Select(outLocal, maskLocal, gLocal, static_cast<float>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        outQueueout.EnQue<float>(outLocal);
        inQueueG.FreeTensor(gLocal);
        inQueueF.FreeTensor(fLocal);
    } else {
        AscendC::LocalTensor<half> gLocal = inQueueG.DeQue<half>();
        AscendC::LocalTensor<half> fLocal = inQueueF.DeQue<half>();
        AscendC::LocalTensor<half> outLocal = outQueueout.AllocTensor<half>();
        AscendC::LocalTensor<uint8_t> maskLocal = tmpQueueMask.AllocTensor<uint8_t>();
        AscendC::CompareScalar(maskLocal, fLocal, static_cast<half>(this->thresholdValue), AscendC::CMPMODE::GT, this->processDataNum);
        AscendC::Select(outLocal, maskLocal, gLocal, static_cast<half>(0.0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        outQueueout.EnQue<half>(outLocal);
        inQueueG.FreeTensor(gLocal);
        inQueueF.FreeTensor(fLocal);
    }
}

template <typename TYPE_INPUT_GRADIENT>
__aicore__ inline void KernelThresholdGradV2D<TYPE_INPUT_GRADIENT>::Process()
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

} // namespace NsThresholdGradV2D
#endif // THRESHOLD_GRAD_V2_D_H