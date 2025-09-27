/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_sum.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "quant_matmul_reduce_sum_mixcore.h"
#include "quant_matmul_reduce_sum_init_output.h"

using namespace AscendC;
using namespace matmul;
using namespace QUANT_MATMUL_REDUCE_SUM;

template <bool trans = false>
using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, trans>;

template <bool trans = false>
using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, int8_t, trans>;

using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

template <bool transA, bool transB, bool sync>
__aicore__ inline void RunQuantMatmulReduceSum(const QbmmRSComputeInitParams* __restrict initParams)
{
    GET_TILING_DATA(tilingData, initParams->tiling);
    TPipe tPipeClear;
    QuantMatmulReduceSumInitOutput<DTYPE_Y> clearOp;
    clearOp.Init(initParams->y, initParams->user1, &tilingData, &tPipeClear);
    clearOp.Process();
    tPipeClear.Destroy();
    TPipe tPipe;
    using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, NZ_CFG_MDL>;
    typename matmulType::MT mm;
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.matmulTiling, &tPipe);
    }
    QuantMatmulReduceSumQuantMixCoreCompute<matmulType, sync> computeOp(mm);
    computeOp.Init(initParams, &tilingData.qbmmReduceSumParams, &tilingData.matmulTiling, &tPipe);
    QbmmRSProcess<decltype(computeOp)> op(computeOp);
    op.Init(&tilingData.qbmmReduceSumParams, &tilingData.matmulTiling);
    op.Process();
}

extern "C" __global__ __aicore__ void quant_matmul_reduce_sum(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR dims, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR yScale,
    GM_ADDR x1Offset, GM_ADDR x2Offset, GM_ADDR yOffset, GM_ADDR x2Table, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    QbmmRSComputeInitParams initParams;
    // 预留参数拦截
    if (bias != nullptr) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    initParams.user1 = GetUserWorkspace(workspace);
    if (initParams.user1 == nullptr) {
        return;
    }
    initParams.x1 = x1;
    initParams.x2 = x2;
    initParams.x1Scale = x1Scale;
    initParams.x2Scale = x2Scale;
    initParams.y = y;
    initParams.workspace = workspace;
    initParams.tiling = tiling;
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
        RunQuantMatmulReduceSum<false, false, false>(&initParams);
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        RunQuantMatmulReduceSum<false, false, false>(&initParams);
    }
}
