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
 * \file apply_gradient_descent.cpp
 * \brief apply_gradient_descent
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/apply_gradient_descent_dag.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "../inc/platform.h"
#include "arch35/apply_gradient_descent_tiling_key.h"
#include "arch35/apply_gradient_descent_tiling_struct.h"

using namespace AscendC;
using namespace ApplyGradientDescentNs;

template <uint64_t schMode>
__global__ __aicore__ void apply_gradient_descent(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyGradientDescentTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyGradientDescentTilingData, tilingData, tiling);
    TPipe pipe;
    ElementwiseSch<schMode, ApplyGradientDescentOp::ApplyGradientDescentDAG<DTYPE_VAR>::OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.Init(var, alpha, delta, var_out);
    sch.Process();
}