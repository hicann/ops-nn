/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_P_ZERO_H_
#define _RENORM_P_ZERO_H_

#include "kernel_operator.h"
#include "common/renorm_common.h"

/*
 * ===================== p = 0 范数计算（0-范数，非零元素计数）=====================
 *
 * 【数学定义】
 *   0-范数 = 非零元素的个数（严格来说不是范数，但常被这样称呼）。
 *   即统计子张量中有多少个元素不等于 0。
 *
 * 【实现方法】
 *   Ascend C 没有直接"数非零个数"的指令，需要分三步实现：
 *     步骤1：Compare(x != 0) 生成 mask（uint8 数组，非零位置为 1，零位置为 0）
 *     步骤2：Select(mask, 1.0f, 0.0f) 把 mask 转成 FP32 的 0/1 数组（1=非零, 0=零）
 *     步骤3：ReduceSum 对 0/1 数组求和，结果就是非零元素个数
 *   数据流：GM →(MTE2)→ UB(FP32) →(V:Compare)→ mask →(V:Select)→ UB(FP32 0/1) →(V:ReduceSum)→ 标量 count
 *
 * 【256 字节对齐要求】
 *   Compare API 硬件要求输入操作数地址 256 字节对齐（即 FP32 下 64 个元素），
 *   因此需要在 32 字节对齐(8元素)的基础上进一步 pad 到 64 元素对齐。
 *   补零的元素在 Compare 后 mask 为 0（因为 0==0），不影响非零计数。
 */

namespace NsRenorm {

using namespace AscendC;

// Compare API 要求 256 字节对齐
// 256 字节 = 64 个 FP32 元素（256 / 4 = 64），所以对齐粒度是 64
constexpr int64_t COMPARE_ALIGN_FP32 = 64; // 256 / sizeof(float) = 64

// 计算 p = 0 场景的范数（非零元素计数，0-范数）
// 遍历子张量的所有连续段，统计非零元素个数
template <typename D_T_X>
__aicore__ inline float ComputeNormPZero(LocalTensor<float>& workBuf, LocalTensor<D_T_X>& dataBuf,
                                         LocalTensor<uint8_t>& maskBuf, LocalTensor<float>& zerosBuf,
                                         LocalTensor<float>& onesBuf, LocalTensor<uint8_t>& reduceBuf,
                                         GlobalTensor<D_T_X>& inputGM, int64_t sliceIdx, int64_t blockSize,
                                         int64_t numBlocks, int64_t sliceCount, int64_t tileLength)
{
    float count = 0.0f;
    LocalTensor<float> reduceLocal = reduceBuf.ReinterpretCast<float>();

    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;

        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;

            // 加载数据并 Cast 到 FP32
            int64_t alignedLen8 = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);

            // Padding 到 256 字节对齐（Compare API 要求）
            int64_t alignedLen64 = (alignedLen8 + COMPARE_ALIGN_FP32 - 1) / COMPARE_ALIGN_FP32 * COMPARE_ALIGN_FP32;
            if (alignedLen64 > alignedLen8) {
                Duplicate(workBuf[alignedLen8], 0.0f, static_cast<int32_t>(alignedLen64 - alignedLen8));
                PipeBarrier<PIPE_V>();
            }

            // 准备 zerosBuf 和 onesBuf
            Duplicate(zerosBuf, 0.0f, static_cast<int32_t>(alignedLen64));
            Duplicate(onesBuf, 1.0f, static_cast<int32_t>(alignedLen64));
            PipeBarrier<PIPE_V>();

            // 步骤1：Compare(workBuf != zerosBuf) → mask (uint8_t: 1=非零, 0=零)
            // CMPMODE::NE 表示 Not-Equal（不等于）：x != 0 时 mask=1，x == 0 时 mask=0
            // maskBuf 是 uint8_t 类型的 LocalTensor，每个元素 1 字节表示一个 bool 结果
            Compare(maskBuf, workBuf, zerosBuf, CMPMODE::NE, static_cast<int32_t>(alignedLen64));
            PipeBarrier<PIPE_V>();

            // 步骤2：Select: mask ? ones(1.0f) : zeros(0.0f)
            // 将 mask（0/1 bool）转成 FP32 的 0.0/1.0 数组，这样后续才能用 ReduceSum 求和计数。
            // mask=1(非零) → 选 onesBuf(1.0f)；mask=0(零) → 选 zerosBuf(0.0f)
            // 为什么需要这一步：ReduceSum 只能对数值求和，不能直接对 uint8 mask 求和。
            Select(workBuf, maskBuf, onesBuf, zerosBuf, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(alignedLen64));
            PipeBarrier<PIPE_V>();

            // ReduceSum → 非零计数
            ReduceSum<float>(reduceLocal, workBuf, workBuf, static_cast<uint32_t>(alignedLen64));
            {
                TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(eventID);
                WaitFlag<HardEvent::V_S>(eventID);
            }
            count += reduceLocal.GetValue(0);

            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }

    return count;
}

} // namespace NsRenorm

#endif // _RENORM_P_ZERO_H_
