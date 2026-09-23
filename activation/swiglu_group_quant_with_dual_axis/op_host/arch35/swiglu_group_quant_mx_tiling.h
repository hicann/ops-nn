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
 * \file swiglu_group_quant_mx_tiling.h
 * \brief Dual-axis-owned MX scheduling geometry shared with the single-axis operator.
 */

#ifndef OPS_NN_SWIGLU_GROUP_QUANT_MX_HOST_TILING_H
#define OPS_NN_SWIGLU_GROUP_QUANT_MX_HOST_TILING_H

#include <cstdint>

namespace SwigluGroupQuantMxTiling {

constexpr int64_t TILE_ROWS = 64;
constexpr int64_t TILE_COLS = 256;
constexpr int64_t MX_SCALE2_BUFFER_BYTES = 896;

struct Geometry {
    int64_t usedCoreNum;
    int64_t dimNBlockNum;
    int64_t dimNTail;
};

inline Geometry MakeGeometry(int64_t dimN, int64_t coreNum)
{
    const int64_t dimNBlockNum = (dimN + TILE_COLS - 1) / TILE_COLS;
    const int64_t dimNTail = dimN % TILE_COLS == 0 ? TILE_COLS : dimN % TILE_COLS;
    const int64_t usedCoreNum = coreNum;
    return {usedCoreNum, dimNBlockNum, dimNTail};
}

} // namespace SwigluGroupQuantMxTiling

#endif // OPS_NN_SWIGLU_GROUP_QUANT_MX_HOST_TILING_H
