/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adam_w.cpp
 * \brief apply_adam_w
 */

#include "apply_adam_w.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class ApplyAdamWTilingKey : uint32_t {
    TILING_KEY_DB = 0,
    TILING_KEY_NDB = 1,
};

template <uint64_t schMode>
__global__ __aicore__ void apply_adam_w(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR beta1_power, GM_ADDR beta2_power,
                                        GM_ADDR lr, GM_ADDR weight_decay, GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon,
                                        GM_ADDR grad, GM_ADDR max_grad_norm, GM_ADDR var_out, GM_ADDR m_out,
                                        GM_ADDR v_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyAdamWTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamWTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (schMode == static_cast<uint32_t>(ApplyAdamWTilingKey::TILING_KEY_DB)) {
        MyApplyAdamW::KernelApplyAdamW<DTYPE_VAR, DOUBLE_BUFFER_NUM> op;
        op.Init(var, m, v, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm,
                var_out, m_out, v_out, &tilingData, &pipe); // 算子kernel实例初始化
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ApplyAdamWTilingKey::TILING_KEY_NDB)) {
        MyApplyAdamW::KernelApplyAdamW<DTYPE_VAR, SINGLE_BUFFER_NUM> op;
        op.Init(var, m, v, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm,
                var_out, m_out, v_out, &tilingData, &pipe); // 算子kernel实例初始化
        op.Process();
    }
}
