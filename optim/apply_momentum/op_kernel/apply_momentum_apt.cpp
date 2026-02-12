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
 * \file apply_momentum.cpp
 * \brief apply_momentum
 */

#include "kernel_operator.h"
#include "./arch35/apply_momentum_dag.h"
#include "./arch35/apply_momentum_tiling_key.h"
#include "./arch35/apply_momentum_tiling_data.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace ApplyMomentumOp;

template <uint64_t schMode, uint64_t useNesterov>
__global__ __aicore__ void apply_momentum(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad,
                                          GM_ADDR momentum, GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ApplyMomentumRegbaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyMomentumRegbaseTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (static_cast<int>(useNesterov) == 0) {
        ElementwiseSch<schMode, ApplyMomentumOp::ApplyMomentumDag<DTYPE_VAR>::OpDag> sch(&(tilingData.elewiseTiling), &pipe);
        sch.Init(var, accum, lr, grad, momentum, var_out, accum);
        sch.Process();
    } else if constexpr (static_cast<int>(useNesterov) == 1) {
        ElementwiseSch<schMode, ApplyMomentumOp::ApplyNesterovMomentumDag<DTYPE_VAR>::OpDag> sch(&(tilingData.elewiseTiling), &pipe);
        sch.Init(var, accum, lr, grad, momentum, var_out, accum);
        sch.Process();
    }
    return;
}
