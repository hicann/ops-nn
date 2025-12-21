/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DEEP_NORM_TILING_H_
#define DEEP_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct DeepNormTilingData {
    uint32_t num_core;
    uint32_t num_last_dim;
    uint32_t num_first_dim;
    uint32_t nl_firstdim_per_core;
    uint32_t l_firstdim_per_core;
    uint32_t first_dim_per_times;
    uint32_t updated_last_dim;
    uint32_t updated_last_times;
    uint32_t eps_str;
    uint32_t ave_str;
    uint32_t alpha_str;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                               \
    DeepNormTilingData tilingData;                                               \
    INIT_TILING_DATA(DeepNormTilingData, tilingDataPointer, tilingPointer);      \
    (tilingData).num_core = tilingDataPointer->num_core;                         \
    (tilingData).num_last_dim = tilingDataPointer->num_last_dim;                 \
    (tilingData).num_first_dim = tilingDataPointer->num_first_dim;               \
    (tilingData).nl_firstdim_per_core = tilingDataPointer->nl_firstdim_per_core; \
    (tilingData).l_firstdim_per_core = tilingDataPointer->l_firstdim_per_core;   \
    (tilingData).first_dim_per_times = tilingDataPointer->first_dim_per_times;   \
    (tilingData).updated_last_dim = tilingDataPointer->updated_last_dim;         \
    (tilingData).updated_last_times = tilingDataPointer->updated_last_times;     \
    (tilingData).eps_str = tilingDataPointer->eps_str;                           \
    (tilingData).ave_str = tilingDataPointer->ave_str;                           \
    (tilingData).alpha_str = tilingDataPointer->alpha_str;
#endif
