/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_kernel.h
 * \brief 310P/m200(__CCE_AICORE__ == 200) 标量实现；910B 向量实现见 op_kernel 顶层。
 */

#include "kernel_operator.h"
#include "../common/sync_batch_norm_gather_stats_tiling_key.h"
#include "../common/sync_batch_norm_gather_stats_kernel_base.h"

using namespace AscendC;

namespace SyncBatchNormGatherStats {

template <typename T, typename CountT>
class SyncBatchNormGatherStatsScalar : public SyncBatchNormGatherStatsKernelBase<T, CountT> {
    SYNC_BATCH_NORM_GATHER_STATS_USE_BASE_MEMBERS;

public:
    __aicore__ inline SyncBatchNormGatherStatsScalar(TPipe* pipeIn) { (void)pipeIn; }

    __aicore__ inline void InitFull(const SyncBatchNormGatherStatsTilingData* tilingData) { SetFullTiling(tilingData); }

    __aicore__ inline void InitNotFull(const SyncBatchNormGatherStatsNNotFullLoadTilingData* tilingData)
    {
        SetNotFullTiling(tilingData);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockDim || nTotal == 0 || cLen == 0 || cLoop == 0) {
            return;
        }
        ProcessScalarGeneric();
    }

private:
    __aicore__ inline void ProcessScalarGeneric()
    {
        float countAll = 0.0f;
        for (uint64_t i = 0; i < nTotal; ++i) {
            countAll += static_cast<float>(sampleCountGm.GetValue(i));
        }
        const float reciprocalCount = (countAll == 0.0f) ? 0.0f : (1.0f / countAll);
        const float oneSubMomentum = 1.0f - momentum;
        // countAll<=1 退化为有偏估计，防 (countAll-1)=0 产生 NaN 写入 running_var
        const float unbiasedMomentum = (countAll > 1.0f) ? (countAll / (countAll - 1.0f) * momentum) : momentum;
        uint64_t cCount = (cLoop - 1) * cFactor + cTail;
        if (cBase + cCount > cLen) {
            cCount = cLen - cBase;
        }
        if (nTotal < SCALAR_COUNT_N_MAX) {
            ProcessSmallN(cCount, reciprocalCount, oneSubMomentum, unbiasedMomentum);
        } else {
            ProcessChunkedC(cCount, reciprocalCount, oneSubMomentum, unbiasedMomentum);
        }
    }

    // 小 N 路径：逐通道标量累加，每写满一个 cache line 周期 flush，规避 310P tail-channel 漏写。
    __aicore__ inline void ProcessSmallN(uint64_t cCount, float reciprocalCount, float oneSubMomentum,
                                         float unbiasedMomentum)
    {
        // flush 步长 32B：与 tiling 侧每核 >= 64B 下限共同保证 310P 正确性，勿放大
        constexpr uint64_t flushStrideElems = 32 / sizeof(T);
        for (uint64_t i = 0; i < cCount; ++i) {
            const uint64_t c = cBase + i;
            float sum = 0.0f;
            float squareSum = 0.0f;
            for (uint64_t n = 0; n < nTotal; ++n) {
                sum += static_cast<float>(totalSumGm.GetValue(n * cLen + c));
                squareSum += static_cast<float>(totalSquareSumGm.GetValue(n * cLen + c));
            }
            WriteChannel(c, sum, squareSum, reciprocalCount, oneSubMomentum, unbiasedMomentum);
            if ((i + 1) % flushStrideElems == 0) {
                FlushOutputs();
            }
        }
        FlushOutputs();
    }

    // 大 N 路径：按 SCALAR_C_CHUNK 分块累加，每块写完 flush 一次，规避大 C tail-channel 漏写。
    __aicore__ inline void ProcessChunkedC(uint64_t cCount, float reciprocalCount, float oneSubMomentum,
                                           float unbiasedMomentum)
    {
        for (uint64_t cOffset = 0; cOffset < cCount; cOffset += SCALAR_C_CHUNK) {
            const uint64_t currentC = Min(SCALAR_C_CHUNK, cCount - cOffset);
            float sum[SCALAR_C_CHUNK] = {0.0f};
            float squareSum[SCALAR_C_CHUNK] = {0.0f};

            for (uint64_t n = 0; n < nTotal; ++n) {
                const uint64_t gmBase = n * cLen + cBase + cOffset;
                for (uint64_t j = 0; j < currentC; ++j) {
                    sum[j] += static_cast<float>(totalSumGm.GetValue(gmBase + j));
                    squareSum[j] += static_cast<float>(totalSquareSumGm.GetValue(gmBase + j));
                }
            }

            for (uint64_t j = 0; j < currentC; ++j) {
                WriteChannel(cBase + cOffset + j, sum[j], squareSum[j], reciprocalCount, oneSubMomentum,
                             unbiasedMomentum);
            }
            // 每块写完即 flush，规避 310P tail-channel 漏写
            FlushOutputs();
        }
        FlushOutputs();
    }

    // 单通道 fp32 标量公式 + 四路输出 SetValue，小 N 与分块路径共用。
    __aicore__ inline void WriteChannel(uint64_t c, float sum, float squareSum, float reciprocalCount,
                                        float oneSubMomentum, float unbiasedMomentum)
    {
        const float bMean = sum * reciprocalCount;
        const float bVar = squareSum * reciprocalCount - bMean * bMean;
        const float bInvstd = 1.0f / sqrt(bVar + eps);
        const float rMean = static_cast<float>(runningMeanGm.GetValue(c)) * oneSubMomentum + momentum * bMean;
        const float rVar = static_cast<float>(runningVarGm.GetValue(c)) * oneSubMomentum + unbiasedMomentum * bVar;
        batchMeanGm.SetValue(c, static_cast<T>(bMean));
        batchInvstdGm.SetValue(c, static_cast<T>(bInvstd));
        runningMeanUpdateGm.SetValue(c, static_cast<T>(rMean));
        runningVarUpdateGm.SetValue(c, static_cast<T>(rVar));
    }

    // 310P/m200 标量写经 GM cache，SetValue 后 flush 四路输出，防遗留脏 cache line
    __aicore__ inline void FlushOutputs()
    {
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(batchMeanGm);
    }
};
} // namespace SyncBatchNormGatherStats

using namespace SyncBatchNormGatherStats;

SYNC_BATCH_NORM_GATHER_STATS_KERNEL_ENTRY(SyncBatchNormGatherStatsScalar)
