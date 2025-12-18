/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_tiling_data.h
 * \brief
 */
#ifndef SPARSE4TO2_QUANT_MATMUL_TILING_DATA_H
#define SPARSE4TO2_QUANT_MATMUL_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace SparseQmm {
#pragma pack(push, 8)
struct Sparse4to2QuantMatmulParams {
    uint32_t xScaleDim = 0;
    uint32_t weightScaleDim = 0;
    uint32_t ubCalcM = 0;
    uint32_t ubCalcN = 0;
    uint32_t needUbBuffer = 0;
    uint32_t biasDtype = 0; // 代替原来的isBiasBf16
    uint32_t isMClash = 0;
    uint32_t isNClash = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct L2cacheTileParams {
    uint32_t mTileCntL2 = 0;
    uint32_t nTileCntL2 = 0;
    uint32_t mTileBlock = 0;
    uint32_t nTileBlock = 0;
    uint32_t calOrder = 0;
    uint32_t reserve = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Sparse4to2QuantMatmulTilingData {
    Sparse4to2QuantMatmulParams params;
    TCubeTiling matmulTiling;
    L2cacheTileParams tileL2cacheTiling;
};
#pragma pack(pop)
} // namespace SparseQmm
#endif // SPARSE4TO2_QUANT_MATMUL_TILING_DATA_H