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
 * \file quant_batch_matmul_v4_tiling_data.h
 * \brief KirinX90/Kirin9030 arch31 S8S4 tiling
 */

#ifndef QUANT_BATCH_MATMUL_V4_ARCH31_TILING_DATA_H
#define QUANT_BATCH_MATMUL_V4_ARCH31_TILING_DATA_H

#include <cstdint>

#pragma pack(push, 8)
struct alignas(8) QuantBatchMatmulV4Arch31TilingData {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t groupSize;
    int32_t tilingM;
    int32_t tilingN;
    int32_t blockDim;
    int32_t hasBias;
    int32_t transposeX2;
};
#pragma pack(pop)

#endif // QUANT_BATCH_MATMUL_V4_ARCH31_TILING_DATA_H
