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
 * \file huber_loss_tiling_data.h
 * \brief HuberLoss host->kernel tiling contract
 *
 * Included by the operator definition source, which a separate opbuild step
 * builds with -std=c++11 and only $CANN/include on the include path. Keep this
 * header C++11 clean and free of project includes.
 */
#ifndef HUBER_LOSS_TILING_DATA_H_
#define HUBER_LOSS_TILING_DATA_H_

#include <cstdint>

/* Reduction enum, matching aten: 0=none, 1=mean, 2=sum.
 *
 * DANGER: smooth_l1_loss_v2 in this repo uses the opposite mapping (1=sum,
 * 2=mean) inside its kernel; carrying that over swaps mean and sum while both
 * still return plausible numbers. These constants are the single definition;
 * op_host and op_kernel both include this header rather than restating them.
 */
constexpr int32_t HUBER_LOSS_REDUCE_NONE = 0;
constexpr int32_t HUBER_LOSS_REDUCE_MEAN = 1;
constexpr int32_t HUBER_LOSS_REDUCE_SUM = 2;

/* accVec width in elements (1KB of float32), also the tile alignment granule:
 * the reduce path accumulates in ACC_LEN-sized chunks.
 */
constexpr int32_t HUBER_LOSS_ACC_LEN = 256;

/* Schedule modes -- the single definition, shared by the kernel template, the
 * tiling arithmetic and the tiling-key declaration.
 *
 * Only none-vs-reduce is split at compile time, and only because the two hold
 * different buffer sets: none has three queues plus an output buffer, reduce
 * has two queues plus accVec/work/result. Splitting there buys each path an
 * independent UB budget. mean and sum share the key -- identical buffers,
 * identical dataflow, and the only difference is the value of divisor below.
 *
 * These are #define rather than constexpr on purpose: ASCENDC_TPL_* expands
 * into a text marker that a build-time tool parses out of the preprocessed
 * output, so the values have to survive preprocessing as literals.
 */
#define HUBER_LOSS_SCH_MODE_NONE 0
#define HUBER_LOSS_SCH_MODE_REDUCE 1

/* Field order is chosen so the struct packs without padding (48 bytes):
 * three uint64, then the 32-bit fields.
 */
struct HuberLossTilingData {
    uint64_t totalNumel = 0;      // total element count over the whole tensor
    uint64_t coreDataNum = 0;     // elements per front core
    uint64_t tailCoreDataNum = 0; // elements per trailing core
    uint32_t usedCoreNum = 1;     // cores actually launched; tiling-only, kernel never derives logic from GetBlockNum()
    uint32_t frontCoreNum = 1;    // cores taking coreDataNum; the remainder goes to the front
    uint32_t tileDataNum = 0;     // elements per tile. Must be > 0 even for an empty tensor: buffers are sized from it
    int32_t reduction = HUBER_LOSS_REDUCE_MEAN; // host-side validation and diagnostics only; the kernel never reads it
    float delta = 1.0f;                         // already validated > 0 in the float32 domain
    /* Reduction epilogue divisor: numel for mean, 1.0 for sum, unused for none.
     * Lets the epilogue be a branchless `total / divisor` in the float32
     * domain, before the single narrowing conversion. Two consequences:
     * dividing by 1.0 is exact, so sum pays nothing; and the empty-tensor
     * result falls out of the same expression instead of a special case --
     * 0/0 is NaN for mean, 0/1 is 0 for sum.
     * Relies on IEEE division semantics; a reciprocal-multiply optimisation
     * would cost a second rounding.
     */
    float divisor = 1.0f;
    /* Byte offset of the cross-core reduction slot region inside workspace.
     * The workspace is laid out as [sync region][slot region], never
     * interleaved, so core0 can load the slots as one contiguous run without
     * the sync bytes landing in the sum. Zero when usedCoreNum == 1, where
     * there is no barrier and no workspace at all.
     */
    uint32_t slotRegionOffset = 0;
};

/* One 32B block per core in the reduction slot region; only lane 0 carries the
 * partial sum.
 */
constexpr uint32_t HUBER_LOSS_SLOT_BYTES = 32;

// 220x bank group is 512B. Concurrent Vector operands whose UB start
// addresses differ by a multiple of 512B land in the same group and serialise
// the repeat. 256B is the official pad that breaks that alias.
constexpr uint32_t HUBER_LOSS_BANK_PAD_BYTES = 256;

/* The sync region SyncAll needs, per core. Matches the two-region-per-core
 * layout the loss operators in this repository use.
 */
constexpr uint32_t HUBER_LOSS_SYNC_BYTES_PER_CORE = 64;

#endif // HUBER_LOSS_TILING_DATA_H_
