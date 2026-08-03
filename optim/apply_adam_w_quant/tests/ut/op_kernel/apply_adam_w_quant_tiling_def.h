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
 * \file apply_adam_w_quant_tiling_def.h
 * \brief
 */
#ifndef _APPLY_ADAM_W_QUANT_TILING_DEF_H_
#define _APPLY_ADAM_W_QUANT_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

struct ApplyAdamWQuantTilingData {
    uint64_t use_num_core;
    uint64_t last_pre_core_row_work;
    uint64_t not_last_core_num;
    uint64_t not_last_pre_core_row_work;
    uint64_t last_core_last_block;
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float eps;
    float gnorm_scale;
    int64_t block_size;
    uint64_t one_core_do_block_num_per_row;
    uint64_t tiling_key;
    uint64_t last_block_size;
};

using ApplyAdamWQuantTilingDataTest = ApplyAdamWQuantTilingData;

inline void InitApplyAdamWQuantTilingData(uint8_t* tiling, ApplyAdamWQuantTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ApplyAdamWQuantTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    ApplyAdamWQuantTilingData tilingData;          \
    InitApplyAdamWQuantTilingData(tilingPointer, &tilingData)

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingPointer) \
    tilingStruct tilingData = *reinterpret_cast<const tilingStruct*>(tilingPointer)
#endif
#endif // _APPLY_ADAM_W_QUANT_TILING_DEF_H_
