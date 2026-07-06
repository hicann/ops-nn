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
 * \file relu6_grad_apt.cpp
 * \brief relu6_grad kernel: dx = (0 < x < 6) ? dy : 0, element-wise with broadcast
 */

#include "kernel_operator.h"
#include "arch35/relu6_grad_dag.h"
#include "arch35/relu6_grad_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void relu6_grad(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace,
                                      GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (std::is_same<DTYPE_GRADIENTS, half>::value || std::is_same<DTYPE_GRADIENTS, bfloat16_t>::value) {
        using OpDag = typename Relu6GradOp::Relu6GradFloatCast<DTYPE_GRADIENTS, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(gradients, features, backprops);
    } else {
        using OpDag = typename Relu6GradOp::Relu6Grad<DTYPE_GRADIENTS>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(gradients, features, backprops);
    }
}
