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
 * \file lamb_update_with_lr.cpp
 * \brief lamb_update_with_lr.cpp
 */

#include "kernel_operator.h"
#include "arch35/lamb_update_with_lr_dag.h"
#include "arch35/lamb_update_with_lr_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void lamb_update_with_lr(GM_ADDR input_greater1, GM_ADDR input_greater_realdiv,
                                               GM_ADDR input_realdiv, GM_ADDR input_mul0, GM_ADDR input_mul1,
                                               GM_ADDR input_sub, GM_ADDR greater_y, GM_ADDR select_e,
                                               GM_ADDR minimum_y, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (std::is_same<DTYPE_INPUT_GREATER1, half>::value) {
        using OpDag = LambUpdateWithLrOp::LambUpdateWithLrCompute<half, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_greater1, input_greater_realdiv, input_realdiv, input_mul0, input_mul1, input_sub, greater_y,
                    select_e, minimum_y, y);
    } else {
        using OpDag = LambUpdateWithLrOp::LambUpdateWithLrCompute<float, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_greater1, input_greater_realdiv, input_realdiv, input_mul0, input_mul1, input_sub, greater_y,
                    select_e, minimum_y, y);
    }
}
