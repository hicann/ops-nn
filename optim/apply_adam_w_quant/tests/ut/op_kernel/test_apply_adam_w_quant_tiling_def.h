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
 * \file test_apply_adam_w_quant.h
 * \brief
 */
#ifndef _APPLY_ADAM_W_QUANT_TILING_H_
#define _APPLY_ADAM_W_QUANT_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct ApplyAdamWQuantTilingDataTest {
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
    uint32_t tiling_key;
};

inline void InitApplyAdamWQuantTilingData(uint8_t* tiling, ApplyAdamWQuantTilingDataTest* const_data)
{
    memcpy(const_data, tiling, sizeof(ApplyAdamWQuantTilingDataTest));
}

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  ApplyAdamWQuantTilingDataTest tilingData;                                               \
  InitApplyAdamWQuantTilingData(tilingPointer, &tilingData)
#endif // _APPLY_ADAM_W_QUANT_TILING_H_