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
 * \file mse_loss_apt.cpp
 * \brief
 */
#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./arch35/mse_loss_dag.h"
#include "./arch35/mse_loss_tiling_key.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "atvoss/reduce/reduce_sch.h"
#include "./arch35/mse_loss_tiling_def.h"
#include "atvoss/reduce/reduce_sch.h"

using namespace ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM, uint32_t Reduction, uint32_t Dtype>
__global__ __aicore__ void mse_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MseLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(MseLossTilingData, tilingData, tiling);
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_PREDICT>::T;
    if constexpr (Reduction == 0) {
        ElementwiseSch<0UL, MseLoss::MseLossOp<DTYPE_PREDICT>::OpDag> sch(
            &(tilingData.baseTiling),
            &pipe); // 获取Schedule
        sch.Init(predict, label, y);
        sch.Process();
    } else if constexpr (Reduction == 1) {
        using Op = ReduceSch<REDUCE_TPL_VALUE, MseLoss::MseLossSumDag<DTYPE_PREDICT, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.Init(&pipe, predict, label, y, workspace);
        op.Process();
    } else if constexpr (Reduction == 2) {
        using Op = ReduceSch<REDUCE_TPL_VALUE, MseLoss::MseLossMeanDag<DTYPE_PREDICT, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<PromoteType, 0>(tilingData.reduceTiling.meanVar);
        op.Init(&pipe, predict, label, y, workspace);
        op.Process(static_cast<DTYPE_PREDICT>(NAN));
    }
}