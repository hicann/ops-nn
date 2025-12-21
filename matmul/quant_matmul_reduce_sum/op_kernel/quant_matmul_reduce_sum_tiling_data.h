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
 * \file quant_matmul_reduce_sum_tiling_data.h
 * \brief
 */

#ifndef QUANT_MATMUL_REDUCE_SUM_TILINNG_DTATA_H
#define QUANT_MATMUL_REDUCE_SUM_TILINNG_DTATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace QUANT_MATMUL_REDUCE_SUM {
#pragma pack(push, 8)
struct alignas(8) QuantMatmulReduceSumParams { // alignas(8)确保8字节对齐
    uint32_t batchNum;
    uint32_t coreNum;
    uint32_t ubBaseK;
    uint32_t ubBaseN;
    uint32_t ubRestBytes;
    uint32_t ubCalSize;
    uint32_t isPertoken;
    uint32_t isDetermine;
    uint64_t workspaceSize;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) QuantMatmulReduceSumTilingData { // alignas(8)确保8字节对齐
    QuantMatmulReduceSumParams qbmmReduceSumParams;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)
} // namespace QUANT_MATMUL_REDUCE_SUM

#endif
