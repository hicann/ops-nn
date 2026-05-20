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
 * \file softshrink.h
 * \brief Softshrink 算子 Kernel 类定义（arch35 架构，Ascend950）
 *
 * 计算公式: Softshrink(x) =
 *   x - lambd, if x > lambd
 *   x + lambd, if x < -lambd
 *   0,         otherwise
 *
 * 实现方案: 三次 Select（分别处理正侧、负侧、置零）
 *   1. Compare(x > lambd) → mask1, Select(mask1 ? x - lambd : 0) → tmp
 *   2. Compare(x < -lambd) → mask2, Select(mask2 ? x + lambd : tmp) → out
 *
 * 模板参数：
 *   - T: IO 数据类型 (half/float)
 *   - BUFFER_MODE: 0=单缓冲, 1=双缓冲
 *   - NEED_UPCAST: 是否在内部把 IO 升精到 fp32 计算
 *     · 0: 直接以 T 计算（fp32 路径）
 *     · 1: Cast IO_T → float → 计算 → Cast float → IO_T（fp16 / bf16 路径）
 *     · 当 T == bfloat16_t 时，硬件不直接支持 bf16 计算，必须设为 1
 *     · 当 T == half 时，为了与 PyTorch CPU fp16/bf16 路径对齐（升精到 fp32 计算），设为 1
 *
 * 与 PyTorch CPU 实现对齐：
 *   PyTorch CPU 对 fp16/bf16 走 vectorized 升精路径 (cpu/Activation.cpp:642-663)：
 *     vector_func: Vectorized<scalar_t> → Vectorized<float> → 计算 → Vectorized<scalar_t>
 *   PyTorch CUDA 对 fp16 直接以 scalar_t 计算（不升精）；本实现按 CPU 升精路径对齐。
 *
 * NaN 行为说明：与 PyTorch 不一致——
 *   PyTorch CUDA 显式透传 NaN（at::_isnan(a) ? a : ...）；
 *   本实现中 (NaN > lambd) 和 (NaN < -lambd) 在 AscendC 上均为 false，
 *   NaN 会被静默置 0。此为已知差异，详见 benchmark_report.md。
 */
#ifndef SOFTSHRINK_H
#define SOFTSHRINK_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softshrink_tiling_data.h"
#include "softshrink_tiling_key.h"

namespace NsSoftshrink {

using namespace AscendC;

template <typename T, int BUFFER_MODE, int NEED_UPCAST>
class Softshrink {
    // IO 类型: 由模板参数 T 直接决定（half / float / bfloat16_t）
    using IO_T = T;
    // 计算类型: NEED_UPCAST ? float : T
    //   · fp16 / bf16 → COMPUTE_T = float（与 PyTorch CPU 升精路径对齐）
    //   · fp32        → COMPUTE_T = float（保持不变）
    using COMPUTE_T = typename std::conditional<NEED_UPCAST == 1, float, T>::type;
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Softshrink() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
        const SoftshrinkTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> lambdBuf;       // lambd 常量 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> negLambdBuf;    // -lambd 常量 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> xSubBuf;        // x - lambd 中间结果 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> xAddBuf;        // x + lambd 中间结果 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> tmpBuf;         // 两次 Select 之间的中间结果 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> floatInBuf;     // 升精路径的 fp32 输入缓冲 (COMPUTE_T)
    TBuf<QuePosition::VECCALC> cmpMaskBuf;     // Compare 输出 bit mask

    GlobalTensor<IO_T> xGM;
    GlobalTensor<IO_T> yGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 0.5f;
};

template <typename T, int BUFFER_MODE, int NEED_UPCAST>
__aicore__ inline void Softshrink<T, BUFFER_MODE, NEED_UPCAST>::Init(
    GM_ADDR x, GM_ADDR y, const SoftshrinkTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;

    xGM.SetGlobalBuffer((__gm__ IO_T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    yGM.SetGlobalBuffer((__gm__ IO_T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // 初始化 Queue：输入输出 buffer
    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(IO_T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(IO_T));

    // 初始化计算辅助 buffer（使用计算类型 COMPUTE_T）
    pipe.InitBuffer(lambdBuf, ubLength_ * sizeof(COMPUTE_T));
    pipe.InitBuffer(negLambdBuf, ubLength_ * sizeof(COMPUTE_T));
    pipe.InitBuffer(xSubBuf, ubLength_ * sizeof(COMPUTE_T));
    pipe.InitBuffer(xAddBuf, ubLength_ * sizeof(COMPUTE_T));
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(COMPUTE_T));
    if constexpr (NEED_UPCAST == 1) {
        pipe.InitBuffer(floatInBuf, ubLength_ * sizeof(COMPUTE_T));
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

template <typename T, int BUFFER_MODE, int NEED_UPCAST>
__aicore__ inline void Softshrink<T, BUFFER_MODE, NEED_UPCAST>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<IO_T> inputLocal = inputQueue.template AllocTensor<IO_T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(IO_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(inputLocal, xGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(inputLocal);
}

template <typename T, int BUFFER_MODE, int NEED_UPCAST>
__aicore__ inline void Softshrink<T, BUFFER_MODE, NEED_UPCAST>::Compute(
    int64_t currentNum)
{
    AscendC::LocalTensor<IO_T> inputLocal = inputQueue.template DeQue<IO_T>();
    AscendC::LocalTensor<IO_T> outputLocal = outputQueue.template AllocTensor<IO_T>();

    // 获取计算辅助 buffer
    AscendC::LocalTensor<COMPUTE_T> lambdLocal = lambdBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<COMPUTE_T> negLambdLocal = negLambdBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<COMPUTE_T> xSubLocal = xSubBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<COMPUTE_T> xAddLocal = xAddBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<COMPUTE_T> tmpLocal = tmpBuf.Get<COMPUTE_T>();
    AscendC::LocalTensor<uint8_t> maskLocal = cmpMaskBuf.Get<uint8_t>();

    // 对齐 currentNum 到 256B 边界用于 Compare/Select（硬件要求 count 所占空间 256B 对齐）
    // 升精路径 (fp16/bf16 NEED_UPCAST=1)：COMPUTE_T=float(4B)，对齐粒度 = 256/4 = 64
    // 直通路径 (fp32 NEED_UPCAST=0)：     COMPUTE_T=float(4B)，对齐粒度 = 256/4 = 64
    int64_t alignElems = 256 / static_cast<int64_t>(sizeof(COMPUTE_T));
    int64_t alignedNum = (currentNum + alignElems - 1) / alignElems * alignElems;
    if (alignedNum > ubLength_) {
        alignedNum = ubLength_;
    }

    if constexpr (NEED_UPCAST == 1) {
        // 升精路径（fp16/bf16）：Cast IO_T → fp32 → 计算 → Cast fp32 → IO_T
        // 与 PyTorch CPU fp16/bf16 vectorized 升精路径完全对齐 (cpu/Activation.cpp:642-663)
        AscendC::LocalTensor<COMPUTE_T> floatInLocal = floatInBuf.Get<COMPUTE_T>();

        // Cast inputLocal(IO_T) → floatInLocal(float)
        AscendC::Cast(floatInLocal, inputLocal, AscendC::RoundMode::CAST_NONE, currentNum);

        // 计算 xSub = x - lambd, xAdd = x + lambd（fp32 域）
        AscendC::Sub(xSubLocal, floatInLocal, lambdLocal, currentNum);
        AscendC::Add(xAddLocal, floatInLocal, lambdLocal, currentNum);

        // Step 1: mask = (x > lambd), tmp = mask ? (x - lambd) : 0
        AscendC::Compare(maskLocal, floatInLocal, lambdLocal, AscendC::CMPMODE::GT, alignedNum);
        AscendC::Select(tmpLocal, maskLocal, xSubLocal, static_cast<COMPUTE_T>(0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);

        // Step 2: mask = (x < -lambd), out_fp32 = mask ? (x + lambd) : tmp
        AscendC::Compare(maskLocal, floatInLocal, negLambdLocal, AscendC::CMPMODE::LT, alignedNum);
        AscendC::Select(tmpLocal, maskLocal, xAddLocal, tmpLocal,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);

        // Cast fp32 result → IO_T output
        // 对 bfloat16_t 使用 CAST_RINT（round-to-nearest-even），对 half 使用 CAST_NONE（默认 RTNE）
        if constexpr (std::is_same<IO_T, bfloat16_t>::value) {
            AscendC::Cast(outputLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, currentNum);
        } else {
            AscendC::Cast(outputLocal, tmpLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        }
    } else {
        // 直通路径（fp32）：COMPUTE_T = T = IO_T，inputLocal 和 outputLocal 直接参与计算

        // 计算 xSub = x - lambd, xAdd = x + lambd
        AscendC::Sub(xSubLocal, inputLocal, lambdLocal, currentNum);
        AscendC::Add(xAddLocal, inputLocal, lambdLocal, currentNum);

        // Step 1: mask = (x > lambd), tmp = mask ? (x - lambd) : 0
        AscendC::Compare(maskLocal, inputLocal, lambdLocal, AscendC::CMPMODE::GT, alignedNum);
        AscendC::Select(tmpLocal, maskLocal, xSubLocal, static_cast<COMPUTE_T>(0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);

        // Step 2: mask = (x < -lambd), out = mask ? (x + lambd) : tmp
        AscendC::Compare(maskLocal, inputLocal, negLambdLocal, AscendC::CMPMODE::LT, alignedNum);
        AscendC::Select(outputLocal, maskLocal, xAddLocal, tmpLocal,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedNum);
    }

    outputQueue.template EnQue<IO_T>(outputLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T, int BUFFER_MODE, int NEED_UPCAST>
__aicore__ inline void Softshrink<T, BUFFER_MODE, NEED_UPCAST>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<IO_T> outputLocal = outputQueue.template DeQue<IO_T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(IO_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(yGM[progress * ubLength_], outputLocal, copyParams);
    outputQueue.FreeTensor(outputLocal);
}

template <typename T, int BUFFER_MODE, int NEED_UPCAST>
__aicore__ inline void Softshrink<T, BUFFER_MODE, NEED_UPCAST>::Process()
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

} // namespace NsSoftshrink
#endif // SOFTSHRINK_H
