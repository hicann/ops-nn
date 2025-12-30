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
 * \file l2_loss_tiling.cpp
 * \brief
 */
#include "l2_loss_tiling.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/l2_loss_dag.h"
#include "../../op_kernel/arch35/l2_loss_tiling_key.h"

#include "atvoss/reduce/reduce_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace optiling
{
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
class L2LossTiling
{
public:
    explicit L2LossTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus SetTilingData();
    ge::graphStatus TilingReduce();

private:
    gert::TilingContext* tilingContext;
    L2LossTilingKey key;
};

ge::graphStatus L2LossTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter SetTilingData");
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount);
    OP_LOGI(tilingContext->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu",
            key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus L2LossTiling::TilingReduce()
{
    ReduceOpInputParam opInput;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(tilingContext, opInput, 0) == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext->GetNodeName(), "ReduceOp get x input failed"),
                    return ge::GRAPH_FAILED);
    opInput.axes.resize(opInput.shape.size());
    for (size_t i = 0; i < opInput.shape.size(); i++) {
        opInput.axes[i] = i;
    }

    ge::graphStatus status = ge::GRAPH_FAILED;
    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<L2Loss::L2LossDag<float, float>::OpDag>(tilingContext, opInput, key.ReduceTiling);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<L2Loss::L2LossDag<half, float>::OpDag>(tilingContext, opInput, key.ReduceTiling);
    }

    OP_CHECK_IF((status == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext->GetNodeName(),
                                                    "ReduceOp Tiling failed, dtype shoude be in (half/bf16/float)"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus L2LossTiling::RunTiling()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for reduce"),
                    return ge::GRAPH_FAILED);
    return SetTilingData();
}

static ge::graphStatus TilingForL2Loss(gert::TilingContext* context)
{
    OP_LOGD("L2LossTiling", "Enter TilingForL2Loss");
    if (context == nullptr) {
        OP_LOGE("L2LossTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("L2LossTiling", "Enter new L2LossTiling");
    L2LossTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForL2Loss(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(L2Loss).Tiling(TilingForL2Loss).TilingParse<ReduceOpCompileInfo>(TilingPrepareForL2Loss);
}  // namespace optiling