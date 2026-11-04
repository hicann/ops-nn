/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ELEMENTS_V2_SMALL_SHAPE_TILING_H_
#define SCATTER_ELEMENTS_V2_SMALL_SHAPE_TILING_H_

#include <cstdint>

namespace ScatterElements {
constexpr uint32_t SMALL_SHAPE_MAX_RANK = 8;

// Launch policy is shared by host tiling and the SIMT launch bound.
constexpr uint32_t SMALL_SHAPE_THREAD_LIMITS[SMALL_SHAPE_MAX_RANK] = {1024, 1024, 512, 512, 512, 512, 512, 512};

// All small-update kernels use this layout. The scatter-axis target stride
// is zero because the index value replaces that coordinate.
struct ScatterElementsV2SmallShapeTiling {
    uint32_t allAxis;
    uint32_t rank;
    uint32_t dim;
    uint32_t threads;
    // Keep the low-rank scalar arguments together in the first 32 bytes.
    uint32_t dataStride;
    uint32_t indicesStride;
    uint32_t updatesStride;
    uint32_t axisStride;
    uint32_t targetStrides[SMALL_SHAPE_MAX_RANK];
    uint32_t indicesStrides[SMALL_SHAPE_MAX_RANK - 1];
    uint32_t updatesStrides[SMALL_SHAPE_MAX_RANK - 1];
    uint32_t reserved[2];
};
static_assert(sizeof(ScatterElementsV2SmallShapeTiling) == 128);
} // namespace ScatterElements

#endif
