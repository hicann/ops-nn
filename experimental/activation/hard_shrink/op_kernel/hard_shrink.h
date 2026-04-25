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
 * \file hard_shrink.h
 * \brief HardShrink 算子 Kernel 类定义（arch35 架构，Ascend950）
 *
 * 计算公式: HardShrink(x) = x if |x| > lambd, else 0
 * 实现方案: 两次 Select（避免 Or API 兼容性风险）
 *   1. Compare(x > lambd) → mask1, Select(mask1 ? x : 0) → tmp
 *   2. Compare(x < -lambd) → mask2, Select(mask2 ? x : tmp) → out
 *
 * 模板参数：
 *   - T: 计算数据类型 (half/float)
 *   - BUFFER_MODE: 0=单缓冲, 1=双缓冲
 *   - IS_BF16: 是否为 bf16 输入（需 Cast bf16<->float）
 */
#ifndef HARD_SHRINK_H
#define HARD_SHRINK_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_shrink_tiling_data.h"
#include "hard_shrink_tiling_key.h"

namespace NsHardShrink {

using namespace AscendC;

template <typename T, int BUFFER_MODE, int IS_BF16>
class HardShrink {
    // IO 类型: IS_BF16 ? bfloat16_t : T
    using IO_T = typename std::conditional<IS_BF16 == 1, bfloat16_t, T>::type;
    // 计算类型: IS_BF16 ? float : T（bf16 需要 cast 到 float 进行计算）
    using COMPUTE_T = typename std::conditional<IS_BF16 == 1, float, T>::type;
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline HardShrink() {};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out,
        const HardShrinkTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> lambdBuf;      // lambd 常量 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> negLambdBuf;   // -lambd 常量 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> tmpBuf;         // 中间结果（两次 Select）(COMPUTE_T)
    TBuf<QuePosition::VECCALC> tmp2Buf;        // bf16 路径额外中间结果 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> cmpMaskBuf;     // Compare 输出 bit mask

    GlobalTensor<IO_T> selfGM;
    GlobalTensor<IO_T> outGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 0.5f;
};

template <typename T, int BUFFER_MODE, int IS_BF16>
__aicore__ inline void HardShrink<T, BUFFER_MODE, IS_BF16>::Init(
    GM_ADDR self, GM_ADDR out, const HardShrinkTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;

    selfGM.SetGlobalBuffer((__gm__ IO_T*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outGM.SetGlobalBuffer((__gm__ IO_T*)out + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // 初始化 Queue：输入输出 buffer
    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(IO_T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(IO_T));

    // 初始化计算辅助 buffer（使用计算类型 COMPUTE_T）
    pipe.InitBuffer(lambdBuf, ubLength_ * sizeof(COMPUTE_T));
    pipe.InitBuffer(negLambdBuf, ubLength_ * sizeof(COMPUTE_T));
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(COMPUTE_T));
    if constexpr (IS_BF16 == 1) {
        pipe.InitBuffer(tmp2Buf, ubLength_ * sizeof(COMPUTE_T));
    }
    // cmpMask 大小 = ubLength / 8 字节，向上 32B 对齐
    int64_t maskBytes = ((ubLength_ / 8) + 31) / 32 * 32;
    pipe.InitBuffer(cmpMaskBuf, maskBytes);

    // 一次性填充 lambd / -lambd 常量
    LocalTensor<COMPUTE_T> lambdLocal = lambdBuf.Get<COMPUTE_T>();
    LocalTensor<COMPUTE_T> negLambdLocal = negLambdBuf.Get<COMPUTE_T>();
    AscendC::Duplicate(lambdLocal, static_cast<COMPUTE_T>(lambd_), ubLength_);
    AscendC::Duplicate(negLambdLocal, static_cast<COMPUTE_T>(-lambd_), ubLength_);
}

template <typename T, int BUFFER_MODE, int IS_BF16>
__aicore__ inline void HardShrink<T, BUFFER_MODE, IS_BF16>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<IO_T> inputLocal = inputQueue.template AllocTensor<IO_T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(IO_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(inputLocal, selfGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(inputLocal);
}

template <typename T, int BUFFER_MODE, int IS_BF16>
__aicore__ inline void HardShrink<T, BUFFER_MODE, IS_BF16>::Compute(
    int64_t currentNum)
{
    AscendC::LocalTensor<IO_T> inputLocal = inputQueue.template DeQue<IO_T>();
    AscendC::LocalTensor<IO_T> outputLocal = outputQueue.template AllocTensor<IO_T>();

    // 获取计算辅助 buffer
    AscendC::LocalTensor<COMPUTE_T> lambdLocal = lambdBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<COMPUTE_T> negLambdLocal = negLambdBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<COMPUTE_T> tmpLocal = tmpBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<uint8_t> maskLocal = cmpMaskBuf.Get<uint8_t>();

    // 对齐 currentNum 到 256B 边界用于 Compare/Select（硬件要求 count 所占空间 256B 对齐）
    // 对于 bf16 路径，计算类型是 float(4B)，对齐粒度 = 256/4 = 64
    // 对于 fp16 路径，计算类型是 half(2B)，对齐粒度 = 256/2 = 128
    // 对于 fp32 路径，计算类型是 float(4B)，对齐粒度 = 256/4 = 64
    int64_t alignElems = 256 / static_cast<int64_t>(sizeof(COMPUTE_T));
    int64_t alignedNum = (currentNum + alignElems - 1) / alignElems * alignElems;
    // alignedNum 不超过 ubLength_（buffer 大小保证安全）
    if (alignedNum > ubLength_) {
        alignedNum = ubLength_;
    }

    if constexpr (IS_BF16 == 1) {
        // bf16 路径：Cast bf16 → float → 计算 → Cast float → bf16
        AscendC::LocalTensor<COMPUTE_T> tmp2Local = tmp2Buf.Get<COMPUTE_T>();

        // Cast inputLocal(bf16) → tmpLocal(float)
        AscendC::Cast(tmpLocal, inputLocal, AscendC::RoundMode::CAST_NONE, currentNum);

        // Step 1: mask = (floatInput > lambd)
        AscendC::Compare(maskLocal, tmpLocal, lambdLocal, AscendC::CMPMODE::GT, alignedNum);
        // Select: mask ? floatInput : 0.0f → tmp2Local
        AscendC::Select(tmp2Local, maskLocal, tmpLocal, static_cast<COMPUTE_T>(0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);

        // Step 2: mask = (floatInput < -lambd)
        AscendC::Compare(maskLocal, tmpLocal, negLambdLocal, AscendC::CMPMODE::LT, alignedNum);
        // Select: mask ? floatInput : tmp2Local → tmpLocal
        AscendC::Select(tmpLocal, maskLocal, tmpLocal, tmp2Local,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);

        // Cast float result → bf16 output
        AscendC::Cast(outputLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, currentNum);
    } else {
        // 非 bf16 路径（fp16/fp32）：直接使用 COMPUTE_T 类型计算
        // inputLocal 和 outputLocal 类型都是 COMPUTE_T（=T=IO_T）

        // Step 1: mask = (input > lambd)
        AscendC::Compare(maskLocal, inputLocal, lambdLocal, AscendC::CMPMODE::GT, alignedNum);
        // Select: mask ? input : 0 → tmpLocal
        AscendC::Select(tmpLocal, maskLocal, inputLocal, static_cast<COMPUTE_T>(0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);

        // Step 2: mask = (input < -lambd)
        AscendC::Compare(maskLocal, inputLocal, negLambdLocal, AscendC::CMPMODE::LT, alignedNum);
        // Select: mask ? input : tmpLocal → outputLocal
        AscendC::Select(outputLocal, maskLocal, inputLocal, tmpLocal,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
    }

    outputQueue.template EnQue<IO_T>(outputLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T, int BUFFER_MODE, int IS_BF16>
__aicore__ inline void HardShrink<T, BUFFER_MODE, IS_BF16>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<IO_T> outputLocal = outputQueue.template DeQue<IO_T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(IO_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outGM[progress * ubLength_], outputLocal, copyParams);
    outputQueue.FreeTensor(outputLocal);
}

template <typename T, int BUFFER_MODE, int IS_BF16>
__aicore__ inline void HardShrink<T, BUFFER_MODE, IS_BF16>::Process()
{
    if (blockLength_ <= 0) {
        return;  // 空 Tensor 或当前核无任务
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsHardShrink
#endif // HARD_SHRINK_H
