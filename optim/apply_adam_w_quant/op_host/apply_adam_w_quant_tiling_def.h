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

#ifndef APPLY_ADAM_W_QUANT_TILING_DEF_H
#define APPLY_ADAM_W_QUANT_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ApplyAdamWQuantTilingData)
TILING_DATA_FIELD_DEF(uint64_t, use_num_core);               // 总共使用的核数
TILING_DATA_FIELD_DEF(uint64_t, last_pre_core_row_work);     // 尾核一个核循环的个数
TILING_DATA_FIELD_DEF(uint64_t, not_last_core_num);          // 非尾核的个数
TILING_DATA_FIELD_DEF(uint64_t, not_last_pre_core_row_work); // 非尾核一个核循环的个数 298
TILING_DATA_FIELD_DEF(uint64_t, last_core_last_block);       // 最后一个个最后一次循环的block个数
TILING_DATA_FIELD_DEF(float, lr);
TILING_DATA_FIELD_DEF(float, beta1);
TILING_DATA_FIELD_DEF(float, beta2);
TILING_DATA_FIELD_DEF(float, weight_decay);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, gnorm_scale);
TILING_DATA_FIELD_DEF(int64_t, block_size);
TILING_DATA_FIELD_DEF(uint64_t, one_core_do_block_num_per_row);
TILING_DATA_FIELD_DEF(uint64_t, tiling_key);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyAdamWQuant, ApplyAdamWQuantTilingData)

struct ApplyAdamWQuantTilingParam {
    uint64_t use_num_core;               // 总共使用的核数
    uint64_t last_pre_core_row_work;     // 尾核一个核循环的个数
    uint64_t not_last_core_num;          // 非尾核的个数
    uint64_t not_last_pre_core_row_work; // 非尾核一个核循环的个数
    uint64_t last_core_last_block;       // 最后一个个最后一次循环的block个数
    uint64_t per_core_do_block_num;
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float eps;
    float gnorm_scale;
    int64_t block_size;
};

struct Tiling4ApplyAdamWQuantCompileInfo {};
} // namespace optiling
#endif // APPLY_ADAM_W_QUANT_TILING_H
