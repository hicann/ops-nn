/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file apply_adam_d_apt.cpp
 * \brief y =|x|
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/apply_adam_d_dag.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/apply_adam_d_tiling_struct.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void apply_adam_d(GM_ADDR var, GM_ADDR m, GM_ADDR v,
                                                   GM_ADDR beta1_power,GM_ADDR beta2_power, GM_ADDR lr,
                                                   GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad,
                                                   GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ApplyAdamDTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamDTilingData, tilingData, tiling);

    if (GetBlockIdx() >= tilingData.baseTiling.blockNum) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if (TILING_KEY_IS(101UL)) {
        if (tilingData.useNesterov > 0) {
            ElementwiseSch<0UL, ApplyAdamDDagFusionNesterov<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, var_out, m_out, v_out);
            sch.Process();
        } else {
            ElementwiseSch<0UL, ApplyAdamDDagFusion<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, var_out, m_out, v_out);
            sch.Process();
        }
    }
    
    if (TILING_KEY_IS(102UL)) {
        if (tilingData.useNesterov > 0) {
            ElementwiseSch<0UL, ApplyAdamDDagFusionNesterov<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, var_out, m_out, v_out);
            sch.Process();
        } else {
            ElementwiseSch<0UL, ApplyAdamDDagFusion<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, var_out, m_out, v_out);
            sch.Process();
        }
    }  
    
    if (TILING_KEY_IS(103UL)) {
        if (tilingData.useNesterov > 0) {
            ElementwiseSch<0UL, ApplyAdamDDagFusionNesterov<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, var_out, m_out, v_out);
            sch.Process();
        } else {
            ElementwiseSch<0UL, ApplyAdamDDagFusion<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
            sch.Init(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, var_out, m_out, v_out);
            sch.Process();
        }
    }

    return;
}