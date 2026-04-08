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

/**
 * \file hard_swish.h
 * \brief HardSwish 算子 kernel 类定义（arch35 - Ascend950）
 *
 * 计算公式：y = x * clamp(x + 3, 0, 6) / 6
 *
 * 使用双缓冲（BUFFER_NUM=2）实现流水线并行：
 *   CopyIn:  GM -> UB（输入数据搬运）
 *   Compute: UB 执行 HardSwish 计算
 *   CopyOut: UB -> GM（输出数据搬运）
 *
 * 模板参数 T：float（仅用于 fp32 路径）
 * fp16 使用独立类 HardSwishFp16（提升至 fp32 计算，提高精度）
 * bf16 使用独立类 HardSwishBf16（提升至 fp32 计算）
 *
 * ISSUE-001 修复：x=-inf 时，x*clamp(x+3,0,6) = -inf*0 = nan。
 * 解决方案：在乘法前将 x clamp 到 >= -3，即 Maxs(xLocal, xLocal, -3)。
 * 当 x <= -3 时 clamp 结果为 0，因此 max(x,-3)*0 = -3*0 = 0（正确）。
 */
#ifndef HARD_SWISH_H
#define HARD_SWISH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_swish_tiling_data.h"
#include "hard_swish_tiling_key.h"

namespace NsHardSwish {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// ============================================================
// HardSwish<T>: fp32 通用模板（仅 T=float 使用）
// fp16 已改为 HardSwishFp16 专用类以提升精度
// ============================================================
template <typename T>
class HardSwish {
public:
    __aicore__ inline HardSwish() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void HardSwish<T>::Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void HardSwish<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void HardSwish<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> tmpLocal = tmpQueue.AllocTensor<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // Step 1: tmpLocal = x + 3
    Adds(tmpLocal, xLocal, static_cast<T>(3.0f), currentNum);
    // Step 2: tmpLocal = min(tmpLocal, 6)
    Mins(tmpLocal, tmpLocal, static_cast<T>(6.0f), currentNum);
    // Step 3: tmpLocal = max(tmpLocal, 0) => clamp(x+3, 0, 6)
    Maxs(tmpLocal, tmpLocal, static_cast<T>(0.0f), currentNum);
    // Step 4 (ISSUE-001 fix): clamp x to >= -3, so -inf becomes -3.
    //   When x <= -3, clamp result is 0, so max(x,-3)*0 = -3*0 = 0 (not NaN).
    Maxs(xLocal, xLocal, static_cast<T>(-3.0f), currentNum);
    // Step 5: yLocal = max(x,-3) * clamp(x+3, 0, 6)
    Mul(yLocal, xLocal, tmpLocal, currentNum);
    // Step 6: yLocal = yLocal * (1/6)
    Muls(yLocal, yLocal, static_cast<T>(1.0f / 6.0f), currentNum);

    outputQueueY.EnQue<T>(yLocal);
    tmpQueue.FreeTensor(tmpLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void HardSwish<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void HardSwish<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

// ============================================================
// HardSwishFp16: fp16 专用类
// fp16 需要提升为 fp32 计算后再转回 fp16（与 bf16 路径相同策略）
// 包含 ISSUE-001 修复（-inf 边界处理）
// ============================================================
class HardSwishFp16 {
public:
    __aicore__ inline HardSwishFp16() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputFp16Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputFp32Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputFp16Queue;

    GlobalTensor<half> inputGMX;
    GlobalTensor<half> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

__aicore__ inline void HardSwishFp16::Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ half*)x + tilingData->blockFactor * GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ half*)y + tilingData->blockFactor * GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputFp16Queue, BUFFER_NUM, ubLength_ * sizeof(half));
    pipe.InitBuffer(inputFp32Queue, BUFFER_NUM, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpQueue, BUFFER_NUM, ubLength_ * sizeof(float));
    pipe.InitBuffer(outputFp16Queue, BUFFER_NUM, ubLength_ * sizeof(half));
}

__aicore__ inline void HardSwishFp16::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<half> xFp16Local = inputFp16Queue.AllocTensor<half>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(half);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xFp16Local, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputFp16Queue.EnQue(xFp16Local);
}

__aicore__ inline void HardSwishFp16::Compute(int64_t currentNum)
{
    LocalTensor<half> xFp16Local = inputFp16Queue.DeQue<half>();
    LocalTensor<float> xFp32Local = inputFp32Queue.AllocTensor<float>();
    LocalTensor<float> tmpLocal = tmpQueue.AllocTensor<float>();
    LocalTensor<half> yFp16Local = outputFp16Queue.AllocTensor<half>();

    // fp16 -> fp32
    Cast(xFp32Local, xFp16Local, RoundMode::CAST_NONE, currentNum);

    // HardSwish in fp32
    Adds(tmpLocal, xFp32Local, 3.0f, currentNum);
    Mins(tmpLocal, tmpLocal, 6.0f, currentNum);
    Maxs(tmpLocal, tmpLocal, 0.0f, currentNum);
    // ISSUE-001 fix: clamp x to >= -3 to avoid -inf * 0 = NaN
    Maxs(xFp32Local, xFp32Local, -3.0f, currentNum);
    Mul(xFp32Local, xFp32Local, tmpLocal, currentNum);
    Muls(xFp32Local, xFp32Local, 1.0f / 6.0f, currentNum);

    // fp32 -> fp16
    Cast(yFp16Local, xFp32Local, RoundMode::CAST_ROUND, currentNum);

    outputFp16Queue.EnQue<half>(yFp16Local);
    tmpQueue.FreeTensor(tmpLocal);
    inputFp32Queue.FreeTensor(xFp32Local);
    inputFp16Queue.FreeTensor(xFp16Local);
}

__aicore__ inline void HardSwishFp16::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<half> yFp16Local = outputFp16Queue.DeQue<half>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(half);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yFp16Local, copyParams);
    outputFp16Queue.FreeTensor(yFp16Local);
}

__aicore__ inline void HardSwishFp16::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

// ============================================================
// HardSwishBf16: bf16 专用类
// bf16 需要提升为 fp32 计算后再转回 bf16
// 包含 ISSUE-001 修复（-inf 边界处理）
// ============================================================
class HardSwishBf16 {
public:
    __aicore__ inline HardSwishBf16() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputBf16Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputFp32Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputBf16Queue;

    GlobalTensor<bfloat16_t> inputGMX;
    GlobalTensor<bfloat16_t> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

__aicore__ inline void HardSwishBf16::Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ bfloat16_t*)x + tilingData->blockFactor * GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ bfloat16_t*)y + tilingData->blockFactor * GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputBf16Queue, BUFFER_NUM, ubLength_ * sizeof(bfloat16_t));
    pipe.InitBuffer(inputFp32Queue, BUFFER_NUM, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpQueue, BUFFER_NUM, ubLength_ * sizeof(float));
    pipe.InitBuffer(outputBf16Queue, BUFFER_NUM, ubLength_ * sizeof(bfloat16_t));
}

__aicore__ inline void HardSwishBf16::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<bfloat16_t> xBf16Local = inputBf16Queue.AllocTensor<bfloat16_t>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(bfloat16_t);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xBf16Local, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputBf16Queue.EnQue(xBf16Local);
}

__aicore__ inline void HardSwishBf16::Compute(int64_t currentNum)
{
    LocalTensor<bfloat16_t> xBf16Local = inputBf16Queue.DeQue<bfloat16_t>();
    LocalTensor<float> xFp32Local = inputFp32Queue.AllocTensor<float>();
    LocalTensor<float> tmpLocal = tmpQueue.AllocTensor<float>();
    LocalTensor<bfloat16_t> yBf16Local = outputBf16Queue.AllocTensor<bfloat16_t>();

    // bf16 -> fp32
    Cast(xFp32Local, xBf16Local, RoundMode::CAST_NONE, currentNum);

    // HardSwish in fp32
    Adds(tmpLocal, xFp32Local, 3.0f, currentNum);
    Mins(tmpLocal, tmpLocal, 6.0f, currentNum);
    Maxs(tmpLocal, tmpLocal, 0.0f, currentNum);
    // ISSUE-001 fix: clamp x to >= -3 to avoid -inf * 0 = NaN
    Maxs(xFp32Local, xFp32Local, -3.0f, currentNum);
    Mul(xFp32Local, xFp32Local, tmpLocal, currentNum);
    Muls(xFp32Local, xFp32Local, 1.0f / 6.0f, currentNum);

    // fp32 -> bf16
    Cast(yBf16Local, xFp32Local, RoundMode::CAST_ROUND, currentNum);

    outputBf16Queue.EnQue<bfloat16_t>(yBf16Local);
    tmpQueue.FreeTensor(tmpLocal);
    inputFp32Queue.FreeTensor(xFp32Local);
    inputBf16Queue.FreeTensor(xBf16Local);
}

__aicore__ inline void HardSwishBf16::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<bfloat16_t> yBf16Local = outputBf16Queue.DeQue<bfloat16_t>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(bfloat16_t);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yBf16Local, copyParams);
    outputBf16Queue.FreeTensor(yBf16Local);
}

__aicore__ inline void HardSwishBf16::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsHardSwish
#endif // HARD_SWISH_H
