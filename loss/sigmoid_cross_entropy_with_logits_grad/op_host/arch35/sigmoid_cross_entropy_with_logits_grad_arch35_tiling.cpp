/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid_cross_entropy_with_logits_grad_arch35_tiling.cpp
 * \brief
 */
#include "sigmoid_cross_entropy_with_logits_grad_arch35_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_grad_tiling_data.h"
#include <algorithm>

using namespace ge;

namespace optiling {

static constexpr size_t IDX_PREDICT = 0;
static constexpr size_t IDX_TARGET = 1;
static constexpr size_t IDX_DOUT = 2;
static constexpr size_t WORKSPACE_NUM = 1;

static ge::graphStatus TilingForSigmoidCrossEntropyWithLogitsGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto nodeName = context->GetNodeName();
    if (nodeName != nullptr) {
        OP_LOGI(nodeName, "Enter SigmoidCrossEntropyWithLogitsGradTilingFunc");
    }
    auto predictShapePtr = context->GetInputShape(IDX_PREDICT);
    if (predictShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto predictShape = predictShapePtr->GetStorageShape();

    int64_t totalElements = 1;
    for (size_t i = 0; i < predictShape.GetDimNum(); ++i) {
        totalElements *= predictShape.GetDim(i);
    }

    auto predictDesc = context->GetInputDesc(IDX_PREDICT);
    if (predictDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inputDtype = predictDesc->GetDataType();

    auto checkDtype = [&](size_t idx) -> bool {
        const auto* desc = context->GetInputDesc(idx);
        return desc == nullptr || desc->GetDataType() == inputDtype;
    };
    if (!checkDtype(IDX_TARGET))
        return ge::GRAPH_FAILED;
    if (!checkDtype(IDX_DOUT))
        return ge::GRAPH_FAILED;

    SigmoidCrossEntropyWithLogitsGradTilingData*
        tilingData = context->GetTilingData<SigmoidCrossEntropyWithLogitsGradTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData->totalLength = static_cast<uint64_t>(totalElements);
    tilingData->tileNum = static_cast<uint32_t>(
        std::min(static_cast<int64_t>(256), std::max(static_cast<int64_t>(1), totalElements)));

    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (ws != nullptr) {
        ws[0] = 0;
    }
    context->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseForSigmoidCrossEntropyWithLogitsGrad(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SigmoidCrossEntropyWithLogitsGrad)
    .Tiling(TilingForSigmoidCrossEntropyWithLogitsGrad)
    .TilingParse<SigmoidCrossEntropyWithLogitsGradCompileInfo>(ParseForSigmoidCrossEntropyWithLogitsGrad);

} // namespace optiling
