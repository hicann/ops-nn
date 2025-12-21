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
 * \file apply_ftrl.cpp
 * \brief apply_ftrl
 */
#include "./arch35/apply_ftrl_dag.h"
#include "./arch35/apply_ftrl_tiling_key.h"
#include "./arch35/apply_ftrl_tiling_data.h"
#include "kernel_operator.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace ApplyFtrlOp;
using namespace ApplyFtrlOpTiling;

template <uint64_t schMode>
__global__ __aicore__ void apply_ftrl(GM_ADDR var, GM_ADDR accum, GM_ADDR linear, GM_ADDR grad,
                                        GM_ADDR lr, GM_ADDR l1, GM_ADDR l2, GM_ADDR lr_power,
                                        GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ApplyFtrlRegbaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyFtrlRegbaseTilingData, tilingData, tiling);
    TPipe pipe;
    ElementwiseSch<schMode, ApplyFtrlOp::ApplyFtrlDag<DTYPE_VAR>::OpDag> sch(&(tilingData.elewiseTiling), &pipe);
    sch.Init(var, accum, linear, grad, lr, l1, l2, lr_power, var_out, accum, linear);
    sch.Process();
    return;
}
