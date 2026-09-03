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
 * \file huber_loss_tiling_calc.h
 * \brief HuberLoss tiling arithmetic, free of any framework dependency.
 *
 * Everything here is a pure function of (numel, reduction, dtype size, UB
 * size, core count). No gert, no platform query, no CANN header -- so the tile
 * formula, the core split and the single-core threshold can be unit tested on
 * a plain host compiler.
 *
 * The framework glue in huber_loss_tiling.cpp only fetches the inputs,
 * calls this, and forwards the result to SetBlockDim / SetTilingKey /
 * SetWorkspaceSize.
 */
#ifndef HUBER_LOSS_TILING_CALC_H_
#define HUBER_LOSS_TILING_CALC_H_

#include <cstdint>
#include "../op_kernel/huber_loss_tiling_data.h"

namespace huber_loss {

// Below this element count a single core is used. The reduce path pays for
// multiple cores with a workspace and a barrier, and at small sizes that costs
// more than the parallelism returns. Tuning choice, not a measured optimum.
constexpr uint64_t HUBER_LOSS_SINGLE_CORE_THRESHOLD = 32768;

// GM copies move whole 32B blocks. Two cores must never own elements inside
// the same block, or their writes collide. Also the UB sub-buffer alignment.
constexpr uint32_t HUBER_LOSS_BLOCK_BYTES = 32;

// Core boundaries use the DMA engine's preferred GM granularity rather than
// the 32B that correctness alone would need. Costs at most one granule of
// imbalance on the tail core and no UB.
constexpr uint32_t HUBER_LOSS_GM_ALIGN_BYTES = 512;

struct HuberTilingPlan {
    HuberLossTilingData data{}; // exactly the struct the kernel receives; no translation layer
    uint32_t blockDim = 1;      // mirrors data.usedCoreNum
    uint32_t tilingKey = HUBER_LOSS_SCH_MODE_NONE;
    uint64_t workspaceSize = 0;
    bool valid = false; // false means the inputs were rejected; the caller reports the error
};

inline uint64_t CeilDiv(uint64_t a, uint64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

inline uint64_t AlignDownU64(uint64_t v, uint64_t a) { return (a == 0) ? v : (v / a) * a; }

/* Per-element UB cost, in bytes, mirroring the kernel's Init() allocations
 * term for term. If Init changes, this must change with it.
 *
 *   both paths: 2 input queues, double buffered
 *               + 2 tile-wide fp32 slots (wsBuf)
 *   none:       + 1 output queue, double buffered
 *   reduce:     + accBuf, one tile wide
 *
 * The two fp32 slots carry every fp32 role -- the upcast inputs, the compute
 * temporaries and the result -- because the compute chain only writes where
 * its sources are already dead, so this does not scale with dtype.
 */
inline uint32_t PerElementUbBytes(bool isReduce, uint32_t dtypeBytes)
{
    const uint32_t kBufferNum = 2;
    const uint32_t kFp32 = 4;
    const uint32_t kFp32Slots = 2;

    uint32_t bytes = 2 * kBufferNum * dtypeBytes; // input, target queues
    bytes += kFp32Slots * kFp32;                  // wsBuf
    if (isReduce) {
        bytes += 1 * kFp32; // accBuf: a whole tile wide, see the kernel
    } else {
        bytes += 1 * kBufferNum * dtypeBytes; // output queue
    }
    return bytes;
}

/* Fixed UB cost, independent of the tile size: the accumulator trio and the
 * scalar staging blocks, plus alignment slack. Sub-buffer offsets are rounded
 * up to 32B, so a tile can waste up to 31B per sub-buffer; the slack term
 * covers that pessimistically.
 */
inline uint32_t FixedUbBytes(bool isReduce, uint32_t dtypeBytes)
{
    const uint32_t kSubBuffers = (dtypeBytes == 4) ? 6 : 9; // conservative count of aligned regions
    uint32_t bytes = kSubBuffers * HUBER_LOSS_BLOCK_BYTES;
    // Pads between concurrently-accessed UB regions (input/target queues,
    // the two ws slots, ws and acc). See HUBER_LOSS_BANK_PAD_BYTES.
    bytes += 4 * HUBER_LOSS_BANK_PAD_BYTES;
    if (isReduce) {
        // resBuf, the scalar staging blocks and the cross-core slot staging
        // are fixed; the accumulator and the reduction scratch scale with the tile.
        bytes += static_cast<uint32_t>(HUBER_LOSS_ACC_LEN) * 4; // resBuf
        bytes += 2 * HUBER_LOSS_BLOCK_BYTES;                    // scalarBuf, scalarOutBuf
        bytes += 2 * 64 * HUBER_LOSS_SLOT_BYTES;                // slot staging, sized for the widest core count
    }
    return bytes;
}

/* Input validation, kept apart from the decisions below so each function
 * stays small enough to read at a glance. Rejects nothing the host has not
 * already validated: delta > 0 and a legal reduction are the caller's job,
 * but they are re-checked here so the pure function is safe to unit test on
 * its own.
 */
inline bool InputsValid(int32_t reduction, float delta, uint32_t dtypeBytes, uint64_t ubSize, uint32_t aivNum)
{
    if (reduction != HUBER_LOSS_REDUCE_NONE && reduction != HUBER_LOSS_REDUCE_MEAN &&
        reduction != HUBER_LOSS_REDUCE_SUM) {
        return false;
    }
    if (!(delta > 0.0f)) { // also rejects NaN
        return false;
    }
    if (dtypeBytes != 2 && dtypeBytes != 4) {
        return false;
    }
    return ubSize != 0 && aivNum != 0;
}

/* Tile size from the UB budget. Zero means even one ACC_LEN-sized tile does
 * not fit; the caller must fail rather than launch a kernel that would spin
 * on a zero tile.
 */
inline uint64_t CalcRawTile(bool isReduce, uint32_t dtypeBytes, uint64_t ubSize)
{
    const uint32_t perElem = PerElementUbBytes(isReduce, dtypeBytes);
    const uint32_t fixed = FixedUbBytes(isReduce, dtypeBytes);
    if (perElem == 0 || ubSize <= fixed) {
        return 0;
    }
    return AlignDownU64((ubSize - fixed) / perElem, static_cast<uint64_t>(HUBER_LOSS_ACC_LEN));
}

/* Core split. Both paths scale. The reduce path pays for it with a workspace
 * and a barrier, so below the threshold it stays on one core and skips both.
 * Returns the core count and fills the front/tail fields of data.
 */
inline uint32_t SplitCores(uint64_t numel, uint32_t dtypeBytes, uint32_t aivNum, HuberLossTilingData& data)
{
    if (numel < HUBER_LOSS_SINGLE_CORE_THRESHOLD) {
        data.coreDataNum = numel;
        data.frontCoreNum = 1;
        data.tailCoreDataNum = 0;
        return 1;
    }

    const uint64_t elemsPerBlock = HUBER_LOSS_GM_ALIGN_BYTES / dtypeBytes;
    const uint64_t totalBlocks = CeilDiv(numel, elemsPerBlock);
    uint64_t cores = aivNum;
    if (cores > totalBlocks) {
        cores = totalBlocks;
    }
    // Blocks per core, rounded up, then shrink the core count so the last
    // core is not left starved by the rounding.
    const uint64_t blocksPerCore = CeilDiv(totalBlocks, cores);
    cores = CeilDiv(totalBlocks, blocksPerCore);

    const uint64_t perCore = blocksPerCore * elemsPerBlock;
    data.coreDataNum = perCore;
    data.frontCoreNum = static_cast<uint32_t>(cores) - 1;
    data.tailCoreDataNum = numel - perCore * (static_cast<uint32_t>(cores) - 1);
    return static_cast<uint32_t>(cores);
}

/* A tile larger than the work one core owns just wastes UB. Never returns 0:
 * one ACC_LEN granule is the minimum, the buffers must be non-empty.
 */
inline uint64_t CapTileToCore(uint64_t tile, const HuberLossTilingData& data)
{
    uint64_t maxCoreNumel = data.coreDataNum;
    if (data.tailCoreDataNum > maxCoreNumel) {
        maxCoreNumel = data.tailCoreDataNum;
    }
    if (maxCoreNumel == 0 || tile <= maxCoreNumel) {
        return tile;
    }
    const uint64_t capped = AlignDownU64(maxCoreNumel, static_cast<uint64_t>(HUBER_LOSS_ACC_LEN));
    return (capped == 0) ? HUBER_LOSS_ACC_LEN : capped;
}

/* Workspace layout [sync region][slot region]. A single core does the whole
 * reduction in its own acc: no barrier, no workspace, and nothing to zero.
 */
inline void FillWorkspace(HuberTilingPlan& plan, bool isReduce, uint32_t aivNum)
{
    if (!isReduce || plan.data.usedCoreNum <= 1) {
        plan.data.slotRegionOffset = 0;
        plan.workspaceSize = 0;
        return;
    }

    const uint32_t syncBytes = plan.data.usedCoreNum * HUBER_LOSS_SYNC_BYTES_PER_CORE;
    // Slots carry slack for twice the platform core count. Only slots
    // [0, usedCoreNum) are written and read (the kernel guards the slot
    // store with blockIdx < usedCoreNum); the slack keeps the store
    // inside the mapped region even if GetBlockIdx() ever ranges wider
    // than planned.
    const uint32_t slotBytes = 2u * aivNum * HUBER_LOSS_SLOT_BYTES;
    plan.data.slotRegionOffset = syncBytes;
    plan.workspaceSize = static_cast<uint64_t>(syncBytes) + slotBytes;
    // core0 reads only [0, usedCoreNum). That covers every partial sum by
    // construction: the blockIdx-to-data mapping is this file's front/tail
    // formula, so the cores holding data are exactly the contiguous range
    // [0, usedCoreNum). Slots above that are written with zero and never
    // read, which is why they need no pre-zeroing pass -- and a
    // pre-zeroing pass would have cost a second barrier.
}

/* The whole tiling decision: validate, size the tile, split the cores, fill
 * the contract. Each step is one of the helpers above.
 */
inline HuberTilingPlan CalcTiling(uint64_t numel, int32_t reduction, float delta, uint32_t dtypeBytes, uint64_t ubSize,
                                  uint32_t aivNum)
{
    HuberTilingPlan plan;
    if (!InputsValid(reduction, delta, dtypeBytes, ubSize, aivNum)) {
        return plan; // valid stays false
    }

    const bool isReduce = (reduction != HUBER_LOSS_REDUCE_NONE);
    plan.tilingKey = isReduce ? HUBER_LOSS_SCH_MODE_REDUCE : HUBER_LOSS_SCH_MODE_NONE;

    uint64_t tile = CalcRawTile(isReduce, dtypeBytes, ubSize);
    if (tile == 0) {
        return plan;
    }

    const uint32_t usedCores = SplitCores(numel, dtypeBytes, aivNum, plan.data);
    tile = CapTileToCore(tile, plan.data);

    // --- fill the contract ---------------------------------------------
    plan.data.totalNumel = numel;
    plan.data.usedCoreNum = usedCores;
    plan.data.tileDataNum = static_cast<uint32_t>(tile);
    plan.data.reduction = reduction;
    plan.data.delta = delta;
    // mean divides by numel, sum by 1. none never reads it. For an empty
    // tensor under mean this is 0, and 0/0 yields the NaN the spec asks for
    // with no special case in the kernel.
    plan.data.divisor = (reduction == HUBER_LOSS_REDUCE_MEAN) ? static_cast<float>(numel) : 1.0f;

    plan.blockDim = usedCores;
    FillWorkspace(plan, isReduce, aivNum);

    plan.valid = true;
    return plan;
}

} // namespace huber_loss

#endif // HUBER_LOSS_TILING_CALC_H_
