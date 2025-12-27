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
 * \file relu.cpp
 * \brief y = Relu(x)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/relu_dag.h"
#include "atvoss/elewise/elewise_sch_with_scalar.h"
#include "atvoss/elewise/elewise_sch_16b.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {

    REGISTER_TILING_DEFAULT(EleBaseTilingData16B);
    GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData16B, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(101UL)) {
        ElementwiseSch16B<0UL, GraphRelu<half, half>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(102UL)) {
        ElementwiseSch16B<0UL, GraphRelu<bfloat16_t, float>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(103UL)) {
        ElementwiseSch16B<0UL, GraphRelu<float, float>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(104UL)) {
        ElementwiseSch16B<0UL, GraphRelu<int8_t, half>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(105UL)) {
        ElementwiseSch16B<0UL, GraphRelu<int32_t, int32_t>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if (TILING_KEY_IS(106UL)) {
        ElementwiseSch16B<0UL, GraphReluMax<int64_t>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    }
    return;
}