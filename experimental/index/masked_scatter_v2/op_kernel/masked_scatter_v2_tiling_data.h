/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MASKED_SCATTER_V2_TILING_DATA_H__
#define __MASKED_SCATTER_V2_TILING_DATA_H__

#include <cstdint>

// Host -> device tiling parameters for MaskedScatterV2.
// 由 op_host masked_scatter_v2_tiling.cpp 填写，kernel 入口经
// GET_TILING_DATA_WITH_STRUCT 反序列化后传入 Init。
//
// [FIX-BCAST] 扩展（native-broadcast）字段为 v2 新增：maskIsBcast=1 时 kernel
// 按展开 strides 分段直读广播 mask（准入判定在 tiling 侧完成）。
struct MaskedScatterV2TilingData {
    int32_t total;      // total elements of self (row-major expanded)
    int32_t sourceLen;  // number of elements in source
    int32_t coreNum;    // usedCoreNum (blockDim)
    int32_t useSync;    // 1 = multi-core soft-sync enabled
    int32_t chunksBase; // chunks per core (floor)
    int32_t chunksRem;  // first `chunksRem` cores get one extra chunk
    // [FIX-BCAST] Native-broadcast mask support (host skips expand+contiguous).
    // maskIsBcast=1 only when the broadcast's innermost dim has stride 1 in
    // the expanded space; kernel then walks per-chunk contiguous segments.
    int32_t maskIsBcast;   // 1 = kernel reads broadcast mask natively
    int32_t maskRank;      // expanded rank (= self.dim(), <= 8 when bcast)
    int32_t maskTotal;     // physical numel of the un-expanded mask
    int32_t maskSize[8];   // expanded shape (= self shape, right-aligned)
    int32_t maskStride[8]; // expanded mask strides (0 for broadcast dims)
};

#endif // __MASKED_SCATTER_V2_TILING_DATA_H__
