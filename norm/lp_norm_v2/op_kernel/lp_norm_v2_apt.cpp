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
 * \file lp_norm_v2_apt.cpp
 * \brief lp_norm_v2 kernel
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/lp_norm_v2_dag.h"
#include "arch35/lp_norm_v2_tiling_key.h"
#include "arch35/lp_norm_v2_tiling_struct.h"

using namespace ReduceOpTmpl;
using namespace AscendC;
using namespace LpNormV2;

template <typename Dtype>
struct GetPromoteType {
};

// inf / -inf 场景下做max/min不需要升精度
template <>
struct GetPromoteType<half> {
    using T = half;
};

template <>
struct GetPromoteType<bfloat16_t> {
    using T = float;
};

template <>
struct GetPromoteType<float> {
    using T = float;
};

template <REDUCE_TPL_PARAM, uint32_t TemplateNum>
__global__ __aicore__ void lp_norm_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::LpNormV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(optiling::LpNormV2TilingData, tilingData, tiling);
    TPipe pipe;
    float val = (TemplateNum == TEMPLATE_P_NINF) ? INFINITY : tilingData.epsilon;
    if constexpr (TemplateNum == TEMPLATE_P0) {  // p=0
        using Op = ReduceSch<REDUCE_TPL_VALUE, LpNormV2::LpNormV2P0Dag<DTYPE_X, float, DTYPE_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<float, 0>(tilingData.epsilon);
        op.Init(&pipe, x, y, workspace);
        op.Process(val);
    } else if constexpr (TemplateNum == TEMPLATE_P1) {  // p=1
        using Op = ReduceSch<REDUCE_TPL_VALUE, LpNormV2::LpNormV2P1Dag<DTYPE_X, float, DTYPE_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<float, 0>(tilingData.epsilon);
        op.Init(&pipe, x, y, workspace);
        op.Process(val);
    } else if constexpr (TemplateNum == TEMPLATE_P2) {  // p=2
        using Op = ReduceSch<REDUCE_TPL_VALUE, LpNormV2::LpNormV2P2Dag<DTYPE_X, float, DTYPE_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<float, 0>(tilingData.epsilon);
        op.Init(&pipe, x, y, workspace);
        op.Process(val);
    } else if constexpr (TemplateNum == TEMPLATE_P3) {  // p=3
        using Op = ReduceSch<REDUCE_TPL_VALUE, LpNormV2::LpNormV2P3Dag<DTYPE_X, float, DTYPE_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<float, 0>(tilingData.epsilon);
        op.template SetVar<float, VAR_INDEX_1>(tilingData.recp);
        op.Init(&pipe, x, y, workspace);
        op.Process(val);
    } else if constexpr (TemplateNum == TEMPLATE_P_NINF) {  // p = -inf
        using promoteDtype = GetPromoteType<DTYPE_X>::T;
        using Op = ReduceSch<REDUCE_TPL_VALUE, LpNormV2::LpNormV2PNInfDag<DTYPE_X, promoteDtype, DTYPE_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<promoteDtype, 0>(static_cast<promoteDtype>(tilingData.epsilon));
        op.Init(&pipe, x, y, workspace);
        op.Process(val);
    } else if constexpr (TemplateNum == TEMPLATE_P_INF) {  // p = inf
        using promoteDtype = GetPromoteType<DTYPE_X>::T;
        using Op = ReduceSch<REDUCE_TPL_VALUE, LpNormV2::LpNormV2PInfDag<DTYPE_X, promoteDtype, DTYPE_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<promoteDtype, 0>(static_cast<promoteDtype>(tilingData.epsilon));
        op.Init(&pipe, x, y, workspace);
        op.Process(val);
    } else if constexpr (TemplateNum == TEMPLATE_P_OTHER) {  // p = other
        using Op = ReduceSch<REDUCE_TPL_VALUE, LpNormV2::LpNormV2POtherDag<DTYPE_X, float, DTYPE_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.template SetVar<float, 0>(tilingData.epsilon);
        op.template SetVar<float, VAR_INDEX_1>(tilingData.p);
        op.template SetVar<float, VAR_INDEX_2>(tilingData.recp);
        op.Init(&pipe, x, y, workspace);
        op.Process(val);
    }
}