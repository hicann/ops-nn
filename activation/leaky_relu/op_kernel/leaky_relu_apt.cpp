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
 * \file leaky_relu.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/leaky_relu_dag.h"
#include "arch35/leaky_relu_struct.h"
#include "atvoss/elewise/elewise_sch_with_scalar.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace LeakyReluOp;
using namespace Ops::Base;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void leaky_relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EleBaseTilingData24B);
    GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData24B, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (dType == static_cast<uint64_t>(TPL_FP16)) {
        ElementwiseSchWithScalar<EleBaseTilingData24B, schMode, LeakyReluCastDag<half, float>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_BF16)) {
        ElementwiseSchWithScalar<EleBaseTilingData24B, schMode, LeakyReluCastDag<bfloat16_t, float>::OpDag> sch(
            tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_FP32)) {
        ElementwiseSchWithScalar<EleBaseTilingData24B, schMode, LeakyReluDag<float, float>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    }
}
