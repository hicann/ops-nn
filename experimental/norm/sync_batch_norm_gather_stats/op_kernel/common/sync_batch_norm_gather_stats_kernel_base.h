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
 * \file sync_batch_norm_gather_stats_kernel_base.h
 * \brief 910B 向量与 310P 标量实现共享的 kernel 基类：GM 句柄、分核标量、tiling 解析。
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_KERNEL_BASE_H
#define SYNC_BATCH_NORM_GATHER_STATS_KERNEL_BASE_H

#include "kernel_operator.h"
#include "sync_batch_norm_gather_stats_tiling_key.h"
#include "../sync_batch_norm_gather_stats_tiling_data.h"

namespace SyncBatchNormGatherStats {
using namespace AscendC;

constexpr uint64_t SCALAR_COUNT_N_MAX = 32;
constexpr uint64_t SCALAR_C_CHUNK = 8;

// 910B 向量实现与 310P 标量实现共享的 GM 句柄、分核标量与 tiling 解析样板。
template <typename T, typename CountT>
class SyncBatchNormGatherStatsKernelBase {
public:
    __aicore__ inline void InitGm(GM_ADDR totalSumAddr, GM_ADDR totalSquareSumAddr, GM_ADDR sampleCountAddr,
                                  GM_ADDR runningMeanAddr, GM_ADDR runningVarAddr, GM_ADDR batchMeanAddr,
                                  GM_ADDR batchInvstdAddr, GM_ADDR runningMeanUpdateAddr, GM_ADDR runningVarUpdateAddr)
    {
        totalSumGm.SetGlobalBuffer((__gm__ T*)totalSumAddr);
        totalSquareSumGm.SetGlobalBuffer((__gm__ T*)totalSquareSumAddr);
        sampleCountGm.SetGlobalBuffer((__gm__ CountT*)sampleCountAddr);
        runningMeanGm.SetGlobalBuffer((__gm__ T*)runningMeanAddr);
        runningVarGm.SetGlobalBuffer((__gm__ T*)runningVarAddr);
        batchMeanGm.SetGlobalBuffer((__gm__ T*)batchMeanAddr);
        batchInvstdGm.SetGlobalBuffer((__gm__ T*)batchInvstdAddr);
        runningMeanUpdateGm.SetGlobalBuffer((__gm__ T*)runningMeanUpdateAddr);
        runningVarUpdateGm.SetGlobalBuffer((__gm__ T*)runningVarUpdateAddr);
    }

protected:
    __aicore__ inline void SetFullTiling(const SyncBatchNormGatherStatsTilingData* tilingData)
    {
        blockDim = tilingData->blockDim;
        nTotal = tilingData->nLen;
        cLen = tilingData->cLen;
        cFactor = tilingData->ubFormer;
        nFactor = tilingData->nLen;
        momentum = tilingData->momentum;
        eps = tilingData->eps;

        const uint64_t blockIdx = GetBlockIdx();
        cBase = tilingData->blockFormer * tilingData->ubFormer * blockIdx;
        if (blockIdx == blockDim - 1) {
            cLoop = tilingData->blockTail;
            cTail = tilingData->ubTail;
        } else {
            cLoop = tilingData->blockFormer;
            cTail = tilingData->ubFormer;
        }
    }

    __aicore__ inline void SetNotFullTiling(const SyncBatchNormGatherStatsNNotFullLoadTilingData* tilingData)
    {
        blockDim = static_cast<uint64_t>(tilingData->blockDim);
        cLen = static_cast<uint64_t>(tilingData->cLen);
        cFactor = static_cast<uint64_t>(tilingData->cFactor);
        nFactor = static_cast<uint64_t>(tilingData->nFactor);
        nTotal = (static_cast<uint64_t>(tilingData->nLoop) + static_cast<uint64_t>(tilingData->nMainFoldCount)) *
                     nFactor +
                 static_cast<uint64_t>(tilingData->nTail);
        momentum = tilingData->momentum;
        eps = tilingData->eps;

        const uint64_t blockIdx = GetBlockIdx();
        cBase = ((static_cast<uint64_t>(tilingData->cLoopMainBlock) - 1) * cFactor +
                 static_cast<uint64_t>(tilingData->cTileMainBlock)) *
                blockIdx;
        if (blockIdx == blockDim - 1) {
            cLoop = static_cast<uint64_t>(tilingData->cLoopTailBlock);
            cTail = static_cast<uint64_t>(tilingData->cTailTailBlock);
        } else {
            cLoop = static_cast<uint64_t>(tilingData->cLoopMainBlock);
            cTail = static_cast<uint64_t>(tilingData->cTileMainBlock);
        }
    }

    __aicore__ inline uint64_t Min(uint64_t a, uint64_t b) const { return a < b ? a : b; }

    GlobalTensor<T> totalSumGm;
    GlobalTensor<T> totalSquareSumGm;
    GlobalTensor<CountT> sampleCountGm;
    GlobalTensor<T> runningMeanGm;
    GlobalTensor<T> runningVarGm;
    GlobalTensor<T> batchMeanGm;
    GlobalTensor<T> batchInvstdGm;
    GlobalTensor<T> runningMeanUpdateGm;
    GlobalTensor<T> runningVarUpdateGm;

    uint64_t blockDim = 0;
    uint64_t nTotal = 0;
    uint64_t nFactor = 0;
    uint64_t cLen = 0;
    uint64_t cFactor = 0;
    uint64_t cBase = 0;
    uint64_t cLoop = 0;
    uint64_t cTail = 0;
    float momentum = 0.0f;
    float eps = 0.0f;
};
} // namespace SyncBatchNormGatherStats

// kernel 入口样板宏（GM 绑定 + tilingKey 派发），910B/310P 各展开一次
#define SYNC_BATCH_NORM_GATHER_STATS_KERNEL_ENTRY(KernelClass)                                                        \
    extern "C" __global__ __aicore__ void sync_batch_norm_gather_stats(                                               \
        GM_ADDR total_sum, GM_ADDR total_square_sum, GM_ADDR sample_count, GM_ADDR running_mean, GM_ADDR running_var, \
        GM_ADDR batch_mean, GM_ADDR batch_invstd, GM_ADDR running_mean_update, GM_ADDR running_var_update,            \
        GM_ADDR workspace, GM_ADDR tiling)                                                                            \
    {                                                                                                                 \
        AscendC::TPipe pipe;                                                                                          \
        REGISTER_TILING_DEFAULT(SyncBatchNormGatherStatsTilingData);                                                  \
        KernelClass<DTYPE_TOTAL_SUM, DTYPE_SAMPLE_COUNT> op(&pipe);                                                   \
        op.InitGm(total_sum, total_square_sum, sample_count, running_mean, running_var, batch_mean, batch_invstd,     \
                  running_mean_update, running_var_update);                                                           \
        if (TILING_KEY_IS(SYNC_BATCH_NORM_GATHER_STATS_N_FULL_LOAD)) {                                                \
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 10001", SyncBatchNormGatherStatsTilingData);             \
            GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsTilingData, tilingData, tiling);                      \
            op.InitFull(&tilingData);                                                                                 \
            op.Process();                                                                                             \
        } else if (TILING_KEY_IS(SYNC_BATCH_NORM_GATHER_STATS_N_NOT_FULL_LOAD)) {                                     \
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 20001", SyncBatchNormGatherStatsNNotFullLoadTilingData); \
            GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsNNotFullLoadTilingData, tilingData, tiling);          \
            op.InitNotFull(&tilingData);                                                                              \
            op.Process();                                                                                             \
        }                                                                                                             \
    }

// 派生类复用基类成员的 using 声明宏
#define SYNC_BATCH_NORM_GATHER_STATS_USE_BASE_MEMBERS           \
    using Base = SyncBatchNormGatherStatsKernelBase<T, CountT>; \
    using Base::batchInvstdGm;                                  \
    using Base::batchMeanGm;                                    \
    using Base::blockDim;                                       \
    using Base::cBase;                                          \
    using Base::cFactor;                                        \
    using Base::cLen;                                           \
    using Base::cLoop;                                          \
    using Base::cTail;                                          \
    using Base::eps;                                            \
    using Base::Min;                                            \
    using Base::momentum;                                       \
    using Base::nFactor;                                        \
    using Base::nTotal;                                         \
    using Base::runningMeanGm;                                  \
    using Base::runningMeanUpdateGm;                            \
    using Base::runningVarGm;                                   \
    using Base::runningVarUpdateGm;                             \
    using Base::sampleCountGm;                                  \
    using Base::SetFullTiling;                                  \
    using Base::SetNotFullTiling;                               \
    using Base::totalSquareSumGm;                               \
    using Base::totalSumGm

#endif // SYNC_BATCH_NORM_GATHER_STATS_KERNEL_BASE_H
