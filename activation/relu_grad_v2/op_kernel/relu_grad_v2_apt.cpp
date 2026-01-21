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
 * \file relu_grad_v2.cpp
 * \brief select(mask, gradients, 0)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/relu_grad_v2_dag.h"
#include "arch35/relu_grad_v2_tiling_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace ReluGradV2Ns;
using namespace Ops::Base;
template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void relu_grad_v2(
    GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
    GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
    TPipe pipe;

    if constexpr (dType == TPL_FP16) {
        ElementwiseSch<schMode, ReluGradV2<half>::OpDag> sch(&tilingData, &pipe);
        sch.Init(gradients, mask, backprops);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch<schMode, ReluGradV2<bfloat16_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(gradients, mask, backprops);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        ElementwiseSch<schMode, ReluGradV2<float>::OpDag> sch(&tilingData, &pipe);
        sch.Init(gradients, mask, backprops);
        sch.Process();
    } else if constexpr (dType == TPL_INT8) {
        ElementwiseSch<schMode, ReluGradV2<int8_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(gradients, mask, backprops);
        sch.Process();
    } else if constexpr (dType == TPL_UINT8) {
        ElementwiseSch<schMode, ReluGradV2<uint8_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(gradients, mask, backprops);
        sch.Process();
    } else if constexpr (dType == TPL_INT32) {
        ElementwiseSch<schMode, ReluGradV2<int32_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(gradients, mask, backprops);
        sch.Process();
    }
    return;
}