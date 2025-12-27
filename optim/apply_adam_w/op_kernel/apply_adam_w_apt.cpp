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
 * \file apply_adam_w.cpp
 * \brief apply_adam_w
 */

#include "kernel_operator.h"
#include "./arch35/apply_adam_w_dag.h"
#include "./arch35/apply_adam_w_tiling_key.h"
#include "./arch35/apply_adam_w_tiling_struct.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace ApplyAdamWOp;

template <uint64_t schMode, uint64_t amsgrad, uint64_t dType>
__global__ __aicore__ void apply_adam_w(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR beta1_power,
                                        GM_ADDR beta2_power, GM_ADDR lr, GM_ADDR weight_decay, GM_ADDR beta1,
                                        GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad, GM_ADDR max_grad_norm,
                                        GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyAdamWNs::ApplyAdamWTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamWNs::ApplyAdamWTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (static_cast<int>(amsgrad) == 0) {
        ElementwiseSch<schMode, ApplyAdamWDAG<DTYPE_VAR, float>::OpDag> sch(&(tilingData.eleBaseTilingData), &pipe);
        sch.template SetVar<float, 0>(static_cast<float>(tilingData.maximizeFactor));
        sch.Init(var, m, v, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, var_out, m_out, v_out);
        sch.Process();
    } else if constexpr (static_cast<int>(amsgrad) == 1) {
        ElementwiseSch<schMode, ApplyAdamWAmsGradDAG<DTYPE_VAR, float>::OpDag> sch(&(tilingData.eleBaseTilingData), &pipe);
        sch.template SetVar<float, 0>(static_cast<float>(tilingData.maximizeFactor));
        sch.Init(var, m, v, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, var_out, m_out, v_out);
        sch.Process();
    }
}