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
 * \file lamb_apply_optimizer_assign.cpp
 * \brief lamb_apply_optimizer_assign.cpp
 */

#include "kernel_operator.h"
#include "arch35/lamb_apply_optimizer_assign_dag.h"
#include "arch35/lamb_apply_optimizer_assign_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void lamb_apply_optimizer_assign(
    GM_ADDR grad, GM_ADDR inputv, GM_ADDR inputm, GM_ADDR input3, GM_ADDR mul0_x, GM_ADDR mul1_x, GM_ADDR mul2_x,
    GM_ADDR mul3_x, GM_ADDR add2_y, GM_ADDR steps, GM_ADDR do_use_weight, GM_ADDR weight_decay_rate, GM_ADDR output0,
    GM_ADDR output1, GM_ADDR output2, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // half/fp32 both compute in float (div/sqrt/log/exp need float precision).
    if constexpr (std::is_same<DTYPE_GRAD, half>::value) {
        using OpDag = LambApplyOptimizerAssignOp::LambApplyOptimizerAssignCompute<half, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(
            grad, inputv, inputm, input3, mul0_x, mul1_x, mul2_x, mul3_x, add2_y, steps, do_use_weight,
            weight_decay_rate, output0, output1, output2);
    } else {
        using OpDag = LambApplyOptimizerAssignOp::LambApplyOptimizerAssignCompute<float, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(
            grad, inputv, inputm, input3, mul0_x, mul1_x, mul2_x, mul3_x, add2_y, steps, do_use_weight,
            weight_decay_rate, output0, output1, output2);
    }
}
