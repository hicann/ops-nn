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
 * \file relu_v2.cpp
 * \brief y =Relu(x)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/relu_v2_dag.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/relu_v2_tiling_struct.h"

using namespace AscendC;
using namespace ReluV2Op;
using namespace ReluV2Ns;

extern "C" __global__ __aicore__ void relu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ReluV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluV2TilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        ElementwiseSch<0UL, ReluV2DAG<half, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y, z);
        sch.Process();
    } else if (TILING_KEY_IS(102UL)) {
        ElementwiseSch<0UL, ReluV2DAG<bfloat16_t, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y, z);
        sch.Process();
    } else if (TILING_KEY_IS(103UL)) {
        ElementwiseSch<0UL, ReluV2DAG<float, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y, z);
        sch.Process();
    } else if (TILING_KEY_IS(104UL)) {
        ElementwiseSch<0UL, ReluV2DAG<int8_t, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y, z);
        sch.Process();
    } else if (TILING_KEY_IS(105UL)) {
        ElementwiseSch<0UL, ReluV2DAG<int32_t, int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y, z);
        sch.Process();
    } else if (TILING_KEY_IS(106UL)) {
        ElementwiseSch<0UL, ReluV2DAG<uint8_t, half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y, z);
        sch.Process();
    }
    return;
}