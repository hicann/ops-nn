/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_P_INF_H_
#define _RENORM_P_INF_H_

#include "kernel_operator.h"
#include "common/renorm_common.h"

/*
 * ===================== p = inf 范数计算（无穷范数，最大绝对值）=====================
 *
 * 【数学定义】
 *   inf-范数 = max(|x_i|)，即子张量中所有元素绝对值的最大值。
 *
 * 【实现方法】
 *   由于 UB 容量有限，子张量的数据需要分 chunk 处理：
 *     1. 每个 chunk：GM →(MTE2)→ UB →(V:Abs)→ |x| →(V:ReduceMax)→ 标量 chunkMax
 *     2. 跨 chunk：用标量比较 if (chunkMax > maxVal) 更新全局最大值 maxVal
 *   最终 maxVal 就是整个子张量的最大绝对值。
 *   逐 chunk 取 ReduceMax 再跨 chunk 标量比较，等价于对全体元素取 ReduceMax。
 */

namespace NsRenorm {

using namespace AscendC;

// 计算 p = inf 场景的范数（最大绝对值，inf-范数）
// 遍历子张量的所有连续段，取最大绝对值
template <typename D_T_X>
__aicore__ inline float ComputeNormPInf(LocalTensor<float>& workBuf, LocalTensor<D_T_X>& dataBuf,
                                        LocalTensor<uint8_t>& reduceBuf, GlobalTensor<D_T_X>& inputGM, int64_t sliceIdx,
                                        int64_t blockSize, int64_t numBlocks, int64_t sliceCount, int64_t tileLength)
{
    float maxVal = 0.0f;
    LocalTensor<float> reduceLocal = reduceBuf.ReinterpretCast<float>();

    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;

        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;

            // 加载数据并 Cast 到 FP32
            int64_t alignedLen = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);

            // Abs → workBuf = |x|
            Abs(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();

            // ReduceMax → chunk 内最大值
            ReduceMax<float>(reduceLocal, workBuf, workBuf, static_cast<uint32_t>(alignedLen));
            {
                TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(eventID);
                WaitFlag<HardEvent::V_S>(eventID);
            }
            float chunkMax = reduceLocal.GetValue(0);
            // 为什么用标量比较更新 maxVal：ReduceMax 只能求单个 chunk 内的最大值，
            // 但子张量被切分成多个 chunk，需要跨 chunk 累积全局最大值。
            // 这里用 S 流水线的标量 if 比较逐个 chunk 的最大值，保留最大的那个，
            // 最终 maxVal 就是整个子张量所有元素的最大绝对值。
            if (chunkMax > maxVal) {
                maxVal = chunkMax;
            }

            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }

    return maxVal;
}

} // namespace NsRenorm

#endif // _RENORM_P_INF_H_
