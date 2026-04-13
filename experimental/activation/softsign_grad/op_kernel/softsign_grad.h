/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file softsign_grad.h
 * \brief SoftsignGrad 算子 kernel 实现 (arch35)
 *
 * 数学公式: output = gradients / (1 + |features|)^2
 *
 * 实现策略：
 *   - FP32 路径: 直接 FP32 计算 (Abs -> Adds -> Mul -> Div)
 *   - FP16/BF16 路径: Cast 到 FP32 计算后 Cast 回原精度 (REQ-8.1)
 *
 * 模板参数：
 *   - T: 数据类型 (half / float / bfloat16_t)
 *   - BUFFER_MODE: 缓冲模式 (0=单缓冲, 1=双缓冲)
 *
 * 全量覆盖（迭代三完成）：6 个 TilingKey 组合
 *   T=half/float/bfloat16_t, BUFFER_MODE=0/1
 */

#ifndef SOFTSIGN_GRAD_H
#define SOFTSIGN_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softsign_grad_tiling_data.h"
#include "softsign_grad_tiling_key.h"

namespace NsSoftsignGrad {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class SoftsignGrad {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    // 是否需要 Cast 到 FP32 计算：非 float 类型都需要提升
    static constexpr bool NEEDS_CAST = !std::is_same_v<T, float>;

public:
    __aicore__ inline SoftsignGrad() {}

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR features, GM_ADDR output,
                                const SoftsignGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueGrad;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueFeat;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    // Cast/tmp buffers only used when NEEDS_CAST == true
    TBuf<QuePosition::VECCALC> castBufGrad;
    TBuf<QuePosition::VECCALC> castBufFeat;
    TBuf<QuePosition::VECCALC> tmpBuffer;

    GlobalTensor<T> gradientsGm;
    GlobalTensor<T> featuresGm;
    GlobalTensor<T> outputGm;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignGrad<T, BUFFER_MODE>::Init(
    GM_ADDR gradients, GM_ADDR features, GM_ADDR output,
    const SoftsignGradTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    // 最后一个核 remainderLength 可能为负，clamp 为 0
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }

    gradientsGm.SetGlobalBuffer(
        (__gm__ T*)gradients + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    featuresGm.SetGlobalBuffer(
        (__gm__ T*)features + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGm.SetGlobalBuffer(
        (__gm__ T*)output + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // 输入输出队列：以原始类型 T 分配
    pipe.InitBuffer(inputQueueGrad, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueFeat, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    // FP16/BF16 路径需要 FP32 的 Cast buffer 和临时计算 buffer
    if constexpr (NEEDS_CAST) {
        pipe.InitBuffer(castBufGrad, ubLength_ * sizeof(float));
        pipe.InitBuffer(castBufFeat, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuffer, ubLength_ * sizeof(float));
    } else {
        // FP32 路径：只需要一个 tmp buffer 做中间计算
        pipe.InitBuffer(tmpBuffer, ubLength_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignGrad<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> gradLocal = inputQueueGrad.template AllocTensor<T>();
    AscendC::LocalTensor<T> featLocal = inputQueueFeat.template AllocTensor<T>();

    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    AscendC::DataCopyPad(gradLocal, gradientsGm[progress * ubLength_], copyParams, {false, 0, 0, 0});
    AscendC::DataCopyPad(featLocal, featuresGm[progress * ubLength_], copyParams, {false, 0, 0, 0});

    inputQueueGrad.EnQue(gradLocal);
    inputQueueFeat.EnQue(featLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignGrad<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> gradLocal = inputQueueGrad.template DeQue<T>();
    AscendC::LocalTensor<T> featLocal = inputQueueFeat.template DeQue<T>();
    AscendC::LocalTensor<T> outLocal = outputQueue.template AllocTensor<T>();

    if constexpr (NEEDS_CAST) {
        // FP16/BF16 路径：Cast 到 FP32 -> 计算 -> Cast 回原精度
        AscendC::LocalTensor<float> gradFp32 = castBufGrad.Get<float>();
        AscendC::LocalTensor<float> featFp32 = castBufFeat.Get<float>();
        AscendC::LocalTensor<float> tmpFp32 = tmpBuffer.Get<float>();

        // 1. Cast: T -> float (half->float / bf16->float 均为精度提升, 使用 CAST_NONE)
        AscendC::Cast(gradFp32, gradLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        AscendC::Cast(featFp32, featLocal, AscendC::RoundMode::CAST_NONE, currentNum);

        // 2. tmpFp32 = |features|
        AscendC::Abs(tmpFp32, featFp32, currentNum);
        // 3. tmpFp32 = 1 + |features|
        AscendC::Adds(tmpFp32, tmpFp32, 1.0f, currentNum);
        // 4. tmpFp32 = (1 + |features|)^2
        AscendC::Mul(tmpFp32, tmpFp32, tmpFp32, currentNum);
        // 5. gradFp32 = gradients / (1 + |features|)^2
        AscendC::Div(gradFp32, gradFp32, tmpFp32, currentNum);

        // 6. Cast: float -> T (精度降低, 使用 CAST_ROUND)
        AscendC::Cast(outLocal, gradFp32, AscendC::RoundMode::CAST_ROUND, currentNum);
    } else {
        // FP32 直接计算路径
        AscendC::LocalTensor<float> tmpFp32 = tmpBuffer.Get<float>();

        // 1. tmpFp32 = |features|
        AscendC::Abs(tmpFp32, featLocal, currentNum);
        // 2. tmpFp32 = 1 + |features|
        AscendC::Adds(tmpFp32, tmpFp32, 1.0f, currentNum);
        // 3. tmpFp32 = (1 + |features|)^2
        AscendC::Mul(tmpFp32, tmpFp32, tmpFp32, currentNum);
        // 4. outLocal = gradients / (1 + |features|)^2
        AscendC::Div(outLocal, gradLocal, tmpFp32, currentNum);
    }

    outputQueue.template EnQue<T>(outLocal);
    inputQueueGrad.FreeTensor(gradLocal);
    inputQueueFeat.FreeTensor(featLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignGrad<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> outLocal = outputQueue.template DeQue<T>();

    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    AscendC::DataCopyPad(outputGm[progress * ubLength_], outLocal, copyParams);
    outputQueue.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignGrad<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSoftsignGrad

#endif // SOFTSIGN_GRAD_H
