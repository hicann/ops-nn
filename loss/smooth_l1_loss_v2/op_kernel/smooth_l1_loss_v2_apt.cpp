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
 * \file smooth_l1_loss_v2_apt.cpp
 * \brief
 */
#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./arch35/smooth_l1_loss_v2_dag.h"
#include "./arch35/smooth_l1_loss_v2_tiling_key.h"
#include "./arch35/smooth_l1_loss_v2_tilingdata.h"
#include "atvoss/elewise/elewise_sch.h" 
#include "atvoss/util/dfx.h"
#include "atvoss/reduce/reduce_sch.h"

using namespace AscendC;
using namespace SmoothL1LossV2;

template <REDUCE_TPL_PARAM, uint32_t Reduction, uint32_t Dtype>
__global__ __aicore__ void smooth_l1_loss_v2(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace,
                                             GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SmoothL1LossV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SmoothL1LossV2TilingData, tilingData, tiling);
    TPipe pipe;
    using PromoteType = Ops::Base::ReduceOpTmpl::__reduceType::GetPromoteType<DTYPE_PREDICT>::T;
    if constexpr (Reduction == 0) {
        ElementwiseSch<0UL, SmoothL1LossV2::SmoothL1LossV2OpDag<DTYPE_PREDICT>::OpDag> sch(&(tilingData.baseTiling),
                                                                                           &pipe);  // 获取Schedule
        sch.SetVar<PromoteType, 0>(tilingData.Sigma);
        sch.SetVar<PromoteType, 1>(tilingData.MultiplyValue);
        sch.SetVar<PromoteType, 2>(tilingData.AddsValue);
        sch.Init(predict, label, y);
        sch.Process();
    } else if constexpr (Reduction == 1) {
        using Op = Ops::Base::ReduceOpTmpl::ReduceSch<REDUCE_TPL_VALUE, SmoothL1LossV2::SmoothL1LossV2SumDag<DTYPE_PREDICT, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<PromoteType, 0>(tilingData.Sigma);
        op.template SetVar<PromoteType, 1>(tilingData.MultiplyValue);
        op.template SetVar<PromoteType, 2>(tilingData.AddsValue);
        op.Init(&pipe, predict, label, y, workspace);
        op.Process(static_cast<DTYPE_PREDICT>(0));
    } else if constexpr (Reduction == 2) {
        using Op = Ops::Base::ReduceOpTmpl::ReduceSch<REDUCE_TPL_VALUE,
                             SmoothL1LossV2::SmoothL1LossV2MeanDag<DTYPE_PREDICT, PromoteType>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<PromoteType, 0>(tilingData.Sigma);
        op.template SetVar<PromoteType, 1>(tilingData.MultiplyValue);
        op.template SetVar<PromoteType, 2>(tilingData.AddsValue);
        op.template SetVar<PromoteType, 3>(tilingData.reduceTiling.meanVar);
        op.Init(&pipe, predict, label, y, workspace);
        op.Process(static_cast<DTYPE_PREDICT>(NAN));
    }
}