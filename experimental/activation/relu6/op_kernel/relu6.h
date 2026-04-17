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
 * \file relu6.h
 * \brief Relu6 Kernel 类实现（arch35 = Ascend950）
 *
 * 公式: y = min(max(x, 0), 6)
 * 迭代一：单核 + 单 dtype（float16）骨架
 * 数据流: GM -> UB(inputLocal) -> UB(tmpLocal=Maxs) -> UB(outputLocal=Mins) -> GM
 * UB 使用: 3 个 LocalTensor（inputLocal + tmpLocal + outputLocal）
 */

#ifndef RELU6_H
#define RELU6_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu6_tiling_data.h"
#include "relu6_tiling_key.h"

namespace NsRelu6 {

using namespace AscendC;

/// \brief Relu6 Kernel 类
/// \tparam T 数据类型（迭代一使用 half，迭代二扩展为模板参数）
template <typename T>
class Relu6 {
public:
    __aicore__ inline Relu6() {}

    /// \brief 初始化 GlobalBuffer 和 UB Buffer
    /// \param x 输入 GM 地址
    /// \param y 输出 GM 地址
    /// \param tilingData Tiling 参数
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Relu6TilingData* tilingData);

    /// \brief 执行主循环：CopyIn -> Compute -> CopyOut
    __aicore__ inline void Process();

private:
    /// \brief 从 GM 搬运数据到 UB
    /// \param progress 当前循环进度
    /// \param currentNum 本次搬运元素数
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);

    /// \brief 从 UB 搬运结果到 GM
    /// \param progress 当前循环进度
    /// \param currentNum 本次搬运元素数
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

    /// \brief UB 内计算: max(x, 0) -> min(result, 6)
    /// \param currentNum 本次计算元素数
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQueue;   // 双缓冲：2 个 input buffer 支持乒乓调度
    TQue<QuePosition::VECOUT, 1> outputQueue;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;  // 当前 Core 需处理的元素数
    int64_t ubLength_ = 0;     // 单次 UB 循环处理的元素数
};

template <typename T>
__aicore__ inline void Relu6<T>::Init(GM_ADDR x, GM_ADDR y, const Relu6TilingData* tilingData)
{
    // 计算当前 AI Core 需要处理的元素数量
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    // 设置 Global Buffer，根据 blockIdx 偏移
    inputGM.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // 初始化 UB Buffer：每个 LocalTensor 大小为 ubLength_ * sizeof(T)
    // 32 字节对齐由 AscendC AllocTensor 自动保证
    // inputQueue: 2 个 buffer（双缓冲），支持 CopyIn 与 Compute/CopyOut 流水并行
    // outputQueue: 动态管理 2 个 LocalTensor（tmpLocal + outputLocal）
    pipe.InitBuffer(inputQueue, 2, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, 2, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Relu6<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    // DataCopyPad 自动处理末尾不足 32 字节对齐的尾部数据
    AscendC::DataCopyPad(inputLocal, inputGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Relu6<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], outputLocal, copyParams);
    outputQueue.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Relu6<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
    AscendC::LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

    // Step 1: tmpLocal = max(inputLocal, 0)
    // 使用 Maxs API 实现 Relu 的下界
    AscendC::LocalTensor<T> tmpLocal = outputQueue.template AllocTensor<T>();
    AscendC::Maxs(tmpLocal, inputLocal, static_cast<T>(0), currentNum);

    // Step 2: outputLocal = min(tmpLocal, 6)
    // 使用 Mins API 实现 Relu6 的上界
    AscendC::Mins(outputLocal, tmpLocal, static_cast<T>(6), currentNum);

    // 释放 tmpLocal
    outputQueue.FreeTensor(tmpLocal);

    outputQueue.EnQue(outputLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Relu6<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    if (loopCount == 0) {
        return;
    }

    // 单循环场景：直接串行执行，无需流水
    if (loopCount == 1) {
        int64_t currentNum = blockLength_;
        CopyIn(0, currentNum);
        Compute(currentNum);
        CopyOut(0, currentNum);
        return;
    }

    // 多循环场景：双缓冲流水调度
    // 第 0 轮：仅 CopyIn，启动输入搬运
    int64_t currentNum = ubLength_;
    CopyIn(0, currentNum);

    for (int64_t i = 0; i < loopCount; i++) {
        // 计算当前轮的元素数
        currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;

        // Compute + CopyOut：处理当前轮数据
        Compute(currentNum);
        CopyOut(i, currentNum);

        // CopyIn：预取下一轮数据（与当前轮的 Compute/CopyOut 流水并行）
        if (i < loopCount - 1) {
            int64_t nextNum = ubLength_;
            CopyIn(i + 1, nextNum);
        }
    }
}

} // namespace NsRelu6

#endif // RELU6_H
