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
 * \file max_pool_v3_util.h
 * \brief Shared utility functions and constants for max_pool_v3 — DivRtn,
 *        CalculateUpdateDim, validation helpers, and attribute index constants
 *        used by both InferShape and Tiling.
 */

#ifndef __MAX_POOL_V3_UTIL_H__
#define __MAX_POOL_V3_UTIL_H__

#include <cstdint>
#include <string>

// ============================================================================
// Attribute index constants shared by InferShape and Tiling.
// WARNING: These indices depend on the attribute registration order in
// max_pool_v3_def.cpp. The order is:
//   [0] ksize    (REQUIRED, ListInt)
//   [1] strides  (REQUIRED, ListInt)
//   [2] pads     (OPTIONAL, ListInt, default {0,0,0,0})
//   [3] ceil_mode(OPTIONAL, Bool,   default false)
// If you add/remove/reorder attributes in the OpDef, update these constants.
// ============================================================================
static constexpr size_t INDEX_KSIZE = 0;
static constexpr size_t INDEX_STRIDES = 1;
static constexpr size_t INDEX_PADS = 2;
static constexpr size_t INDEX_CEIL_MODE = 3;

static constexpr size_t SHAPE_4D_SIZE = 4;
static constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

// Pad array layout: [pad_top, pad_bottom, pad_left, pad_right]
static constexpr size_t PAD_TOP = 0;
static constexpr size_t PAD_BOTTOM = 1;
static constexpr size_t PAD_LEFT = 2;
static constexpr size_t PAD_RIGHT = 3;

// Default pads value used when the pads attribute is not explicitly set
// (matches the OpDef default: {0, 0, 0, 0})
static constexpr int64_t DEFAULT_PADS[4] = {0, 0, 0, 0};

// Integer division rounding to -Infinity (floor division).
// Caller must validate y > 0 via ValidateSpatialDims before calling.
static inline int64_t DivRtn(int64_t x, int64_t y)
{
    int64_t q = 0;
    int64_t r = 0;

    if (y > 0) {
        q = x / y;
        r = x % y;
        if (r < 0) {
            --q;
        }
    }

    return q;
}

// Calculate output dimension for one spatial axis.
static inline int64_t CalculateUpdateDim(int64_t ksize, int64_t padL, int64_t padR, int64_t stride, bool ceil_mode,
                                         int64_t dim_size)
{
    if (dim_size == UNKNOWN_DIM_VALUE) {
        return UNKNOWN_DIM_VALUE;
    }
    int64_t outputSize = DivRtn(dim_size + padL + padR - ksize + (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        if ((outputSize - 1) * stride >= dim_size + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

// Validate spatial dimensions of ksize and strides are > 0.
static inline bool ValidateSpatialDims(const int64_t* ksizeData, const int64_t* stridesData, size_t hDim, size_t wDim,
                                       const char* opName)
{
    if (ksizeData[hDim] <= 0 || ksizeData[wDim] <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "ksize[h,w]", (std::to_string(ksizeData[hDim]) + ", " + std::to_string(ksizeData[wDim])).c_str(),
            "ksize spatial dims must be greater than 0");
        return false;
    }
    if (stridesData[hDim] <= 0 || stridesData[wDim] <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "strides[h,w]",
            (std::to_string(stridesData[hDim]) + ", " + std::to_string(stridesData[wDim])).c_str(),
            "strides must be greater than 0");
        return false;
    }
    return true;
}

#endif // __MAX_POOL_V3_UTIL_H__
