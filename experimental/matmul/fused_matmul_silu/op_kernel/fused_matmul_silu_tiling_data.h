/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FUSED_MATMUL_SILU_TILING_DATA_H_
#define FUSED_MATMUL_SILU_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

struct FusedMatmulSiluParams {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t usedCoreNum;
    uint32_t vectorTileElems;
};

struct FusedMatmulSiluTilingData {
    TCubeTiling matmulTiling;
    FusedMatmulSiluParams params;
};

#endif // FUSED_MATMUL_SILU_TILING_DATA_H_
