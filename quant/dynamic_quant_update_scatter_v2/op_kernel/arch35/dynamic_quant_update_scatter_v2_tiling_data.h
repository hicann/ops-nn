/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security.
 */

/*!
 * \file dynamic_quant_update_scatter_v2_tiling_data.h
 * \brief DynamicQuantUpdateScatterV2 arch35 (Ascend950) plain tiling-data struct (host/kernel shared).
 *
 * Per-row (last dim = rowLen = H) asymmetric int4 dynamic quant + scatter.
 * Each of the B batch rows is quantized and scattered to (b, indices[b]) in var/var_scale/var_offset.
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_V2_ARCH35_TILING_DATA_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_V2_ARCH35_TILING_DATA_H

#include <cstdint>

struct DynamicQuantUpdateScatterV2RegbaseTilingData {
    int64_t coreNum = 0;        // used AIV cores
    int64_t rowLen = 0;         // H (last dim of x); must be even for int4
    int64_t rowPerHeadCore = 0; // rows on each of the (coreNum-1) head cores
    int64_t rowPerTailCore = 0; // rows on the last (tail) core
    int64_t batchSize = 0;      // B (number of x rows == number of indices)
    int64_t dstSeqLen = 0;      // S (physical var/scale/offset seq dim); scatter target var[(b*S+s)]
    int64_t alignRowLen = 0;    // CeilAlign(rowLen, 64) fp32 elems for the UB tile
    int64_t outAlignLen = 0;    // CeilAlign(rowLen, 64) int4 elems for the out UB tile
    int64_t varByteLen = 0;     // visible int4-packed var bytes for inplace safety
    int64_t scaleLen = 0;       // visible fp32 scale elements for inplace safety
    int64_t offsetLen = 0;      // visible fp32 offset elements for inplace safety
};

#endif // DYNAMIC_QUANT_UPDATE_SCATTER_V2_ARCH35_TILING_DATA_H
