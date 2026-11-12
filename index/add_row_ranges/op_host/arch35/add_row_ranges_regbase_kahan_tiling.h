/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_row_ranges_regbase_kahan_tiling.h
 * \brief Ascend950 (DAV_3510) RegBase Kahan tiling for add_row_ranges operator
 *
 * 方案 arr-regvec-kahan-v2 的 host 侧：
 *   - numLaneGroups = 2 当且仅当 M*ceil(N/128) >= coreNum/2，否则 1；
 *     tileCols = numLaneGroups * 64（每 lane 组一个 64-fp32-lane VL）；
 *   - tasks = M * ceil(N / tileCols)，usedCoreNum = min(coreNum, tasks)，
 *     不使用每核最小元素门限压核；
 *   - TilingData 尾部追加 M/tileCols/numLaneGroups/numColTiles/usedCoreNum/blockRows，
 *     前三个字段 totalElements/N/K 的类型与顺序不变（fallback 只读这三项）；
 *   - TilingKey = ADD_ROW_RANGES_TPL_SCH_MODE_KAHAN（仅 Ascend950）。
 */

#ifndef ADD_ROW_RANGES_REGBASE_KAHAN_TILING_H_
#define ADD_ROW_RANGES_REGBASE_KAHAN_TILING_H_

#include <cstdint>

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "../../op_kernel/arch35/add_row_ranges_tiling_data.h"
#include "../../op_kernel/arch35/add_row_ranges_tiling_key.h"

namespace optiling {

constexpr int64_t KAHAN_LANES_PER_GROUP = 64;
constexpr int64_t KAHAN_BLOCK_ROWS = 16;
constexpr uint32_t KAHAN_DCACHE_SIZE = 128 * 1024;
// Must match the kernel's KAHAN_QUEUE_DEPTH / KAHAN_UB_ROW_BYTES.
constexpr uint32_t KAHAN_QUEUE_DEPTH = 2;
constexpr uint32_t KAHAN_UB_ROW_BYTES = 256;
// Route guard: below this per-task Kahan work the (row, tile) task setup dominates and
// the element-parallel SIMT path is faster. Values are from a 300-case shape sweep.
constexpr int64_t KAHAN_MIN_WORK_PER_TASK = 2048;
constexpr int64_t KAHAN_MIN_TASKS_PER_CORE = 4;

struct KahanShape {
    int64_t groups;
    int64_t tileCols;
    int64_t numColTiles;
    int64_t tasks;
};

inline KahanShape ComputeKahanShape(int64_t M, int64_t N, int64_t coreNum)
{
    const int64_t wideTasks = M * Ops::Base::CeilDiv(N, 2 * KAHAN_LANES_PER_GROUP);
    const int64_t groups = (wideTasks * 2 >= coreNum) ? 2 : 1;
    const int64_t tileCols = groups * KAHAN_LANES_PER_GROUP;
    const int64_t numColTiles = Ops::Base::CeilDiv(N, tileCols);
    return {groups, tileCols, numColTiles, M * numColTiles};
}

// Static UB footprint of the kernel's TPipe buffers (srcQue + xQue + yQue + stateBuf).
// xQue/yQue are row-block sized so an all-empty block is copied with one DMA pair.
inline uint64_t KahanStaticUbBytes(int64_t groups, int64_t blockRows)
{
    const uint64_t g = static_cast<uint64_t>(groups);
    const uint64_t r = static_cast<uint64_t>(blockRows);
    const uint64_t rowBytes = KAHAN_UB_ROW_BYTES;
    const uint64_t each = static_cast<uint64_t>(KAHAN_QUEUE_DEPTH) * r * g * rowBytes;
    const uint64_t stateBuf = 2ULL * g * rowBytes;
    return 3ULL * each + stateBuf;
}

// Prefer the RegBase path unless a task is too small to amortise its DMA/queue setup.
inline bool KahanRoutePreferred(int64_t M, int64_t N, int64_t K, int64_t coreNum)
{
    const KahanShape shape = ComputeKahanShape(M, N, coreNum);
    const int64_t laneCols = (N < KAHAN_LANES_PER_GROUP) ? N : KAHAN_LANES_PER_GROUP;
    const int64_t workPerTask = K * laneCols;
    if (workPerTask >= KAHAN_MIN_WORK_PER_TASK) {
        return true;
    }
    return shape.tasks <= KAHAN_MIN_TASKS_PER_CORE * coreNum;
}

inline ge::graphStatus AddRowRangesKahanTiling(gert::TilingContext* context, int64_t M, int64_t N, int64_t K,
                                               uint64_t ubSize, int64_t coreNum)
{
    const int64_t totalElements = M * N;
    OP_CHECK_IF(totalElements > static_cast<int64_t>(INT32_MAX),
                OP_LOGE(context, "totalElements %ld exceeds INT32_MAX, not supported", totalElements),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(M > static_cast<int64_t>(INT32_MAX), OP_LOGE(context, "M %ld exceeds INT32_MAX", M),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(N > static_cast<int64_t>(INT32_MAX), OP_LOGE(context, "N %ld exceeds INT32_MAX", N),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(K > static_cast<int64_t>(INT32_MAX), OP_LOGE(context, "K %ld exceeds INT32_MAX", K),
                return ge::GRAPH_FAILED);

    const KahanShape shape = ComputeKahanShape(M, N, coreNum);
    const int64_t groups = shape.groups;
    const int64_t tileCols = shape.tileCols;
    const int64_t numColTiles = shape.numColTiles;
    // Rows per task: as many as possible (<= blockRows) while still giving every core a
    // task; small-M shapes therefore keep one task per row and their parallelism.
    const int64_t rowBlocksTarget = Ops::Base::CeilDiv(coreNum, numColTiles);
    int64_t rowsPerTask = (rowBlocksTarget > 0) ? M / rowBlocksTarget : 1;
    if (rowsPerTask < 1) {
        rowsPerTask = 1;
    }
    if (rowsPerTask > KAHAN_BLOCK_ROWS) {
        rowsPerTask = KAHAN_BLOCK_ROWS;
    }
    const int64_t numRowBlocks = Ops::Base::CeilDiv(M, rowsPerTask);
    const int64_t tasks = numRowBlocks * numColTiles;
    int64_t usedCoreNum = (tasks < coreNum) ? tasks : coreNum;
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    AddRowRangesTilingData* tiling = context->GetTilingData<AddRowRangesTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AddRowRangesTilingData), 0, sizeof(AddRowRangesTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalElements = totalElements;
    tiling->N = static_cast<int32_t>(N);
    tiling->K = static_cast<int32_t>(K);
    tiling->M = static_cast<int32_t>(M);
    tiling->tileCols = static_cast<int32_t>(tileCols);
    tiling->numLaneGroups = static_cast<int32_t>(groups);
    tiling->numColTiles = static_cast<int32_t>(numColTiles);
    tiling->usedCoreNum = static_cast<int32_t>(usedCoreNum);
    tiling->blockRows = static_cast<int32_t>(KAHAN_BLOCK_ROWS);
    tiling->rowsPerTask = static_cast<int32_t>(rowsPerTask);
    tiling->numRowBlocks = static_cast<int32_t>(numRowBlocks);

    context->SetBlockDim(static_cast<int32_t>(usedCoreNum));
    // The kernel allocates srcQue/xQue/yQue/stateBuf from the local-memory pool, so the
    // pool (ubSize - D-cache) must cover that static footprint; fail instead of letting
    // the queues overrun the D-cache region on device.
    const uint64_t staticUb = KahanStaticUbBytes(groups, KAHAN_BLOCK_ROWS);
    OP_CHECK_IF(
        KAHAN_DCACHE_SIZE + staticUb > ubSize,
        OP_LOGE(context, "UB insufficient: dcache=%u + static=%lu > ubSize=%lu", KAHAN_DCACHE_SIZE, staticUb, ubSize),
        return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - KAHAN_DCACHE_SIZE));
    OP_CHECK_IF(res != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu", ubSize),
                return ge::GRAPH_FAILED);

    uint64_t tilingKey = GET_TPL_TILING_KEY(ADD_ROW_RANGES_TPL_SCH_MODE_KAHAN);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context,
            "AddRowRanges kahan route: soc=ASCEND950 M=%ld N=%ld K=%ld groups=%ld tileCols=%ld rowsPerTask=%ld "
            "numRowBlocks=%ld tasks=%ld usedCoreNum=%ld key=%lu",
            M, N, K, groups, tileCols, rowsPerTask, numRowBlocks, tasks, usedCoreNum, tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

#endif // ADD_ROW_RANGES_REGBASE_KAHAN_TILING_H_
