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
 * \file wts_arq_tiling_data.h
 * \brief WtsARQ tiling data shared by host and kernel (arch35 / DAV_3510).
 *
 * Fields follow DESIGN.md §3.3. Template parameter RANK (4/8) fixes the
 * capacity of the collapsed-shape arrays; host pads leading dims with 1.
 */
#ifndef WTS_ARQ_TILING_DATA_H_
#define WTS_ARQ_TILING_DATA_H_

#include <cstdint>

constexpr int64_t kWtsArqMaxRank = 8;
constexpr int64_t kWtsArqFloatBufs = 4; // B_W / B_MIN / B_MAX / B_SCALE
constexpr int64_t kWtsArqTotalBufs = 5; // + B_CAST

// Broadcast copy mode for w_min/w_max (they share the same shape, one decision)
constexpr uint32_t WTS_ARQ_BRC_NONE = 0;  // no broadcast axis: plain DataCopyPad
constexpr uint32_t WTS_ARQ_BRC_NDDMA = 1; // NDDMA GM->UB broadcast (stride=0 axes)
constexpr uint32_t WTS_ARQ_BRC_UB = 2;    // compact copy + UB Broadcast dynamic API

// NDDMA schedule mode (only meaningful when brcMode == WTS_ARQ_BRC_NDDMA)
constexpr uint32_t WTS_ARQ_SCH_WITHOUT_LOOP = 1; // axes after ubSplitAxis <= 5
constexpr uint32_t WTS_ARQ_SCH_WITH_LOOP = 2;    // axes after ubSplitAxis > 5

template <int64_t kRank>
struct WtsArqTilingData {
    int64_t dims[kRank];       // collapsed output shape (leading-padded with 1)
    int64_t minStrides[kRank]; // w_min GM strides on collapsed axes, 0 on broadcast axes
    int64_t maxStrides[kRank]; // w_max, same pattern as w_min

    int64_t ubSplitAxis; // UB split axis index in padded collapsed shape
    int64_t ubFormer;    // tile length on split axis
    int64_t ubOuter;     // tile count on split axis
    int64_t ubTail;      // tail tile length on split axis

    int64_t fusedProduct; // total tiles = ubOuter * product(dims[0..ubSplitAxis-1])
    int64_t blockFormer;  // tiles per core (non-tail cores)
    int64_t blockNum;     // actual cores used
    int64_t blockTail;    // tiles of the last core

    int64_t perBufElems; // per-buffer element capacity (aligned)

    uint32_t shapeLen;   // collapsed (unpadded) rank, <= kRank
    uint32_t schMode;    // WTS_ARQ_SCH_*
    uint32_t brcMode;    // WTS_ARQ_BRC_*
    uint32_t offsetFlag; // attr offset_flag (0/1)
    uint32_t numBits;    // attr num_bits, host validated == 8
    uint32_t coreNum;    // actual enabled cores (== blockNum)
    float eps;           // scale lower-bound guard, 1.1920929e-07
};

#endif // WTS_ARQ_TILING_DATA_H_
