/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_P_POSITIVE_H_
#define _RENORM_P_POSITIVE_H_

#include "kernel_operator.h"
#include "common/renorm_common.h"

/*
 * ===================== p > 0 范数计算（正 p 范数）=====================
 *
 * 【数学公式】
 *   p-范数: norm = ( Σ|x_i|^p )^(1/p)
 *   即先对每个元素取绝对值的 p 次方，求和后开 1/p 次方。
 *
 * 【三条优化路径】
 *   路径1（单元素快速路径）：当 blockSize==1 且 numBlocks==1 时，子张量只有一个元素，
 *     norm = |x|，直接取绝对值即可，跳过所有幂运算和归约，最快。
 *
 *   路径2（p=1 / p=2 直接路径）：p=1 时 norm = Σ|x|，p=2 时 norm = sqrt(Σ|x|²)。
 *     这两种情况中间值不会溢出，可以直接用 Abs→(Mul)→ReduceSum 计算，无需 log/exp。
 *
 *   路径3（通用 p 的 log-sum-exp 路径）：对任意 p，用 |x|^p = exp(p * log|x|) 来计算。
 *     因为 Ascend C 没有逐元素的 Pow 指令，只能用 Log+Muls+Exp 组合实现。
 *     最后用 ScalarPow 做 (sum)^(1/p) 后处理得到最终范数。
 */

namespace NsRenorm {

using namespace AscendC;

// Accuracy fixes must be implemented by the p-norm formula or reduction
// ordering.  Do not compensate the final scale with shape-specific constants.

// Compensated positive-p reduction for long inner reductions.  The normal path
// accumulates one vector-reduction result with a plain scalar add; over many
// chunks this loses low bits and can change the final renorm scale.  Keep the
// reduction granularity and replace only the numerically fragile accumulation.
template <typename D_T_X>
__aicore__ inline float ComputeNormPPositiveCompensated(LocalTensor<float>& workBuf, LocalTensor<D_T_X>& dataBuf,
                                                        LocalTensor<uint8_t>& reduceBuf, LocalTensor<float>& tmpBuf,
                                                        GlobalTensor<D_T_X>& inputGM, int64_t sliceIdx,
                                                        int64_t blockSize, int64_t numBlocks, int64_t sliceCount,
                                                        int64_t tileLength, int32_t exponent, float eps)
{
    LocalTensor<float> reduceLocal = reduceBuf.ReinterpretCast<float>();
    float sum = 0.0f;
    float compensation = 0.0f;
    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;
        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;
            int64_t alignedLen = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);
            Abs(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            if (exponent == 2) {
                Mul(workBuf, workBuf, workBuf, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else {
                // Match the generic path's zero guard, then replace Log/Exp
                // with exact integer exponentiation for p=11.
                Maxs(workBuf, workBuf, eps, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                DataCopy(tmpBuf, workBuf, static_cast<int32_t>(alignedLen));
                Duplicate(workBuf, 1.0f, static_cast<int32_t>(alignedLen));
                int32_t power = exponent;
                while (power > 0) {
                    if ((power & 1) != 0) {
                        Mul(workBuf, workBuf, tmpBuf, static_cast<int32_t>(alignedLen));
                        PipeBarrier<PIPE_V>();
                    }
                    power >>= 1;
                    if (power > 0) {
                        Mul(tmpBuf, tmpBuf, tmpBuf, static_cast<int32_t>(alignedLen));
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
            ReduceSum<float>(reduceLocal, workBuf, workBuf, static_cast<uint32_t>(alignedLen));
            TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            float value = reduceLocal.GetValue(0);
            float corrected = value - compensation;
            float next = sum + corrected;
            compensation = (next - sum) - corrected;
            sum = next;
            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }
    if (exponent == 2) {
        return ScalarSqrt(tmpBuf, sum);
    }
    // ScalarPow supplies a well-scaled initial value; two Newton steps remove
    // its Log/Exp approximation from the final p=11 scale calculation.
    float root = ScalarPow(tmpBuf, sum, 1.0f / static_cast<float>(exponent), eps);
    if (root > 0.0f && root < 3.402823466e38f) {
        for (int32_t iter = 0; iter < 2; ++iter) {
            float rootPow = 1.0f;
            for (int32_t i = 1; i < exponent; ++i) {
                rootPow *= root;
            }
            root = (static_cast<float>(exponent - 1) * root + sum / rootPow) / static_cast<float>(exponent);
        }
    }
    return root;
}

// 计算 p > 0 且 p != inf 场景的范数
// p=1/p=2 用直接路径，通用 p 用 log-sum-exp 避免中间值溢出
// blockSize*numBlocks==1 时 norm=|x|，跳过所有幂运算
// For a long block-major reduction, direct exp(p * log(abs(x))) can overflow
// before the final p-th root is evaluated. Compute the same p-norm in a
// max-normalized domain.
template <typename D_T_X>
__aicore__ inline float ComputeNormPPositiveStable(LocalTensor<float>& workBuf, LocalTensor<D_T_X>& dataBuf,
                                                   LocalTensor<uint8_t>& reduceBuf, LocalTensor<float>& tmpBuf,
                                                   GlobalTensor<D_T_X>& inputGM, int64_t sliceIdx, int64_t blockSize,
                                                   int64_t numBlocks, int64_t sliceCount, int64_t tileLength, float p,
                                                   float eps)
{
    LocalTensor<float> reduceLocal = reduceBuf.ReinterpretCast<float>();
    float maxValue = 0.0f;

    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;
        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;
            int64_t alignedLen = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);
            Abs(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(reduceLocal, workBuf, workBuf, static_cast<uint32_t>(alignedLen));
            TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            float chunkMax = reduceLocal.GetValue(0);
            if (chunkMax > maxValue) {
                maxValue = chunkMax;
            }
            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }

    if (maxValue == 0.0f) {
        return 0.0f;
    }

    float sumPow = 0.0f;
    float compensation = 0.0f;
    float invMax = 1.0f / maxValue;
    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;
        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;
            int64_t alignedLen = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);
            Abs(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(workBuf, workBuf, invMax, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Maxs(workBuf, workBuf, eps, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(workBuf, workBuf, p, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(reduceLocal, workBuf, workBuf, static_cast<uint32_t>(alignedLen));
            TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            float value = reduceLocal.GetValue(0);
            float corrected = value - compensation;
            float next = sumPow + corrected;
            compensation = (next - sumPow) - corrected;
            sumPow = next;
            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }

    return maxValue * ScalarPow(tmpBuf, sumPow, 1.0f / p, eps);
}

template <typename D_T_X>
__aicore__ inline float ComputeNormPPositive(LocalTensor<float>& workBuf, LocalTensor<D_T_X>& dataBuf,
                                             LocalTensor<uint8_t>& reduceBuf, LocalTensor<float>& tmpBuf,
                                             GlobalTensor<D_T_X>& inputGM, int64_t sliceIdx, int64_t blockSize,
                                             int64_t numBlocks, int64_t sliceCount, int64_t tileLength, float p,
                                             float eps)
{
    LocalTensor<float> reduceLocal = reduceBuf.ReinterpretCast<float>();

    // 快速路径：每个切片只有1个元素，norm = |x|
    if (blockSize == 1 && numBlocks == 1) {
        int64_t gmOffset = sliceIdx;
        LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset, 1);
        Abs(workBuf, workBuf, 1);
        PipeBarrier<PIPE_V>();
        {
            TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
        }
        return workBuf.GetValue(0);
    }

    // p=1 或 p=2: 直接路径，不会溢出
    if (p == 1.0f || p == 2.0f) {
        float norm = 0.0f;
        for (int64_t b = 0; b < numBlocks; ++b) {
            int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
            int64_t remaining = blockSize;
            int64_t chunkStart = 0;
            while (remaining > 0) {
                int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;
                int64_t alignedLen = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);
                Abs(workBuf, workBuf, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                if (p == 2.0f) {
                    Mul(workBuf, workBuf, workBuf, static_cast<int32_t>(alignedLen));
                    PipeBarrier<PIPE_V>();
                }
                ReduceSum<float>(reduceLocal, workBuf, workBuf, static_cast<uint32_t>(alignedLen));
                {
                    TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                    SetFlag<HardEvent::V_S>(eventID);
                    WaitFlag<HardEvent::V_S>(eventID);
                }
                norm += reduceLocal.GetValue(0);
                remaining -= chunkLen;
                chunkStart += chunkLen;
            }
        }
        if (p == 2.0f) {
            norm = ScalarSqrt(tmpBuf, norm);
        }
        return norm;
    }

    // 通用 p: 直接计算 |x|^p = exp(p*log|x|)，与 CPU 行为一致（大值溢出到 inf）
    // norm = (sum(|x|^p))^(1/p)
    //
    // 【为什么用 exp(p*log|x|) 而不是直接 Pow】
    //   Ascend C 没有逐元素 Pow 向量指令（即没有 x^p 的单条指令），
    //   所以 |x|^p 只能拆解为：Log(|x|) → Muls(p) → Exp，即 exp(p * log|x|)。
    //   这在数学上等价于 |x|^p，且大值会自然溢出到 inf（与 CPU 行为一致）。
    //
    // 【为什么需要 Maxs(eps)】
    //   log(0) = -inf，如果某个元素 x=0，直接 Log 会产生 -inf，
    //   后续 Exp(-inf*p) = 0 是对的，但 -inf 参与 ReduceSum 可能导致结果不确定。
    //   Maxs(eps) 将所有 |x| 限制在不小于 eps，避免 log(0)=-inf 的问题。
    float sumPow = 0.0f;
    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;
        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;
            int64_t alignedLen = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);
            Abs(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Maxs(workBuf, workBuf, eps, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(workBuf, workBuf, p, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(workBuf, workBuf, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(reduceLocal, workBuf, workBuf, static_cast<uint32_t>(alignedLen));
            {
                TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(eventID);
                WaitFlag<HardEvent::V_S>(eventID);
            }
            sumPow += reduceLocal.GetValue(0);
            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }

    // sumPow 为 inf 时，norm = inf，scale = 0，输出全零（与 CPU 一致）
    //
    // 【1/p 次方后处理】
    //   上面循环累加得到 sumPow = Σ|x|^p，还需要开 1/p 次方才是最终范数：
    //     norm = sumPow^(1/p)
    //   ScalarPow 内部用 exp((1/p) * log(sumPow)) 实现，等价于 sumPow^(1/p)。
    //   当 sumPow=inf（大值溢出）时，log(inf)=inf，exp(inf/p)=inf，norm=inf，
    //   后续 scale = maxNorm/inf = 0，输出全零，与 CPU 行为一致。
    return ScalarPow(tmpBuf, sumPow, 1.0f / p, eps);
}

} // namespace NsRenorm

#endif // _RENORM_P_POSITIVE_H_
