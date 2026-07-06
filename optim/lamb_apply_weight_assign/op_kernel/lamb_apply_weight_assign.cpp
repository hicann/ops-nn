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
 * \file lamb_apply_weight_assign.cpp
 * \brief lamb_apply_weight_assign.cpp
 */

#include "kernel_operator.h"
#include "arch35/lamb_apply_weight_assign_dag.h"
#include "arch35/lamb_apply_weight_assign_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void lamb_apply_weight_assign(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3,
                                                    GM_ADDR input_param, GM_ADDR output0, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // half/fp32 both compute in float (div needs float precision).
    if constexpr (std::is_same<DTYPE_INPUT0, half>::value) {
        using OpDag = LambApplyWeightAssignOp::LambApplyWeightAssignCompute<half, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input0, input1, input2, input3, input_param, output0);
    } else {
        using OpDag = LambApplyWeightAssignOp::LambApplyWeightAssignCompute<float, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input0, input1, input2, input3, input_param, output0);
    }
}
