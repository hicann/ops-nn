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
 * \file apply_adam_w_v2_apt.cpp
 * \brief apply_adam_w_v2_apt
 */

#include "./arch35/apply_adam_w_v2_dag.h"
#include "./arch35/apply_adam_w_v2_tiling_key.h"
#include "./arch35/apply_adam_w_v2_tiling_data.h"
#include "kernel_operator.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace ApplyAdamWV2RegbaseOp;

template <uint64_t schMode, uint64_t amsgrad>
__global__ __aicore__ void apply_adam_w_v2(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR grad, GM_ADDR step,
                                           GM_ADDR max_grad_norm, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyAdamWV2RegbaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamWV2RegbaseTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (static_cast<int>(amsgrad) == 0) {
        ElementwiseSch<schMode, ApplyAdamWV2DAG<DTYPE_VAR, DTYPE_GRAD, DTYPE_STEP, float>::OpDag> sch(&(tilingData.elewiseTiling), &pipe);
        sch.template SetVar<float, 0>(static_cast<float>(tilingData.lr));
        sch.template SetVar<float, 1>(static_cast<float>(tilingData.beta1));
        sch.template SetVar<float, 2>(static_cast<float>(tilingData.beta2));
        sch.template SetVar<float, 3>(static_cast<float>(tilingData.weightDecay));
        sch.template SetVar<float, 4>(static_cast<float>(tilingData.eps));
        sch.template SetVar<float, 5>(static_cast<float>(tilingData.gt));
        sch.Init(var, m, v, grad, step, var, m, v);
        sch.Process();
    } else if constexpr (static_cast<int>(amsgrad) == 1) {
        ElementwiseSch<schMode, ApplyAdamWV2AmsGradDAG<DTYPE_VAR, DTYPE_GRAD, DTYPE_STEP, float>::OpDag> sch(&(tilingData.elewiseTiling), &pipe);
        sch.template SetVar<float, 0>(static_cast<float>(tilingData.lr));
        sch.template SetVar<float, 1>(static_cast<float>(tilingData.beta1));
        sch.template SetVar<float, 2>(static_cast<float>(tilingData.beta2));
        sch.template SetVar<float, 3>(static_cast<float>(tilingData.weightDecay));
        sch.template SetVar<float, 4>(static_cast<float>(tilingData.eps));
        sch.template SetVar<float, 5>(static_cast<float>(tilingData.gt));
        sch.Init(var, m, v, grad, step, max_grad_norm, var, m, v, max_grad_norm);
        sch.Process();
    }
    return;
}