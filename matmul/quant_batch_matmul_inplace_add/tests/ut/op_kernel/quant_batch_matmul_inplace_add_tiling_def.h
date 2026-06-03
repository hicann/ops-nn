/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_DEF_H
#define QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_data.h"

template <typename T>
inline void InitQuantBatchMatmulInplaceAddTilingData(uint8_t *tiling, T *constData)
{
    memcpy(constData, tiling, sizeof(T));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                              \
    QMMIA::QuantBatchMatmulInplaceAddTilingData tiling_data;                                                 \
    InitQuantBatchMatmulInplaceAddTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg)                                  \
    tiling_struct tiling_data;                                                                               \
    InitQuantBatchMatmulInplaceAddTilingData(tiling_arg, &tiling_data)

#endif // QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_DEF_H
