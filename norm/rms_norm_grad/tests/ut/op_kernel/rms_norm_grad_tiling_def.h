/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GE_GLU_V2_TILING_H_
#define _GE_GLU_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct RmsNormGradTilingData {
    uint32_t row;
    uint32_t col;
    float avg_factor;
    uint32_t data_type;
    uint32_t block_factor;
    uint32_t ub_split_dim;
    uint32_t ub_factor;
    uint32_t core_calc_num;
    uint32_t core_calc_tail;
    uint32_t block_dim;
    uint32_t ub_calc_num;
    uint32_t ub_calc_tail;
    uint32_t ub_calc_loop;
    uint32_t ub_calc_tail_num;
    uint32_t ub_calc_tail_tail;
    uint32_t ub_calc_tail_loop;
    uint32_t fixed_output;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                             \
    RmsNormGradTilingData tilingData;                                          \
    INIT_TILING_DATA(RmsNormGradTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).row = tilingDataPointer->row;                                 \
    (tilingData).col = tilingDataPointer->col;                                 \
    (tilingData).avg_factor = tilingDataPointer->avg_factor;                   \
    (tilingData).data_type = tilingDataPointer->data_type;                     \
    (tilingData).block_factor = tilingDataPointer->block_factor;               \
    (tilingData).ub_split_dim = tilingDataPointer->ub_split_dim;               \
    (tilingData).ub_factor = tilingDataPointer->ub_factor;                     \
    (tilingData).core_calc_num = tilingDataPointer->core_calc_num;             \
    (tilingData).core_calc_tail = tilingDataPointer->core_calc_tail;           \
    (tilingData).block_dim = tilingDataPointer->block_dim;                     \
    (tilingData).ub_calc_num = tilingDataPointer->ub_calc_num;                 \
    (tilingData).ub_calc_tail = tilingDataPointer->ub_calc_tail;               \
    (tilingData).ub_calc_loop = tilingDataPointer->ub_calc_loop;               \
    (tilingData).ub_calc_tail_num = tilingDataPointer->ub_calc_tail_num;       \
    (tilingData).ub_calc_tail_tail = tilingDataPointer->ub_calc_tail_tail;     \
    (tilingData).ub_calc_tail_loop = tilingDataPointer->ub_calc_tail_loop;     \
    (tilingData).fixed_output = tilingDataPointer->fixed_output;

#define DTYPE_DY half
#define DTYPE_GAMMA half

#endif