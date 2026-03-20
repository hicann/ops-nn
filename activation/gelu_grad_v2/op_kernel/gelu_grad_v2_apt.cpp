/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_grad_v2.cpp
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/gelu_grad_v2_dag.h"
#include "arch35/gelu_grad_v2_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace GeluGradV2Op;

template <uint64_t schMode,uint64_t approximate>
__global__ __aicore__ void gelu_grad_v2(GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr(approximate == TPL_NONE) {
        if constexpr (IsSameType<DTYPE_DY, float32_t>::value) {
            using OpDag = GeluGradV2None32DAG<DTYPE_DY>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(dy, x, z);
        } else {
            using OpDag = GeluGradV2None16DAG<DTYPE_DY>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(dy, x, z);     
        }
    } else if (approximate == TPL_TANH) {
        using OpDag = GeluGradV2TanhDAG<DTYPE_DY>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(dy, x, z);
    }
    return;
}