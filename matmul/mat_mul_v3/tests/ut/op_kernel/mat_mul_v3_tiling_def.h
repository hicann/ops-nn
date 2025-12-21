/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_MATMUL_V3_TILING_TILING_H_
#define _TEST_MATMUL_V3_TILING_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#include "mat_mul_v3_tiling_data.h"

inline void InitMatmulTilingData(uint8_t* tiling, MatmulTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MatmulTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    MatmulTilingData tiling_data;                                                 \
    InitMatmulTilingData(tiling_arg, &tiling_data)
#endif  // FOREACH_MINIMUM_SCALAR_TILING_DEF_H