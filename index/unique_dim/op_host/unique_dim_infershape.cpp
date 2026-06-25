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
 * \file unique_dim_infershape.cpp
 * \brief Shape / dtype / shape-range inference for UniqueDim
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static constexpr size_t OUT_IDX = 0;
static constexpr size_t INDICES_IDX = 1;
static constexpr size_t COUNTS_IDX = 2;

static ge::graphStatus InferShape4UniqueDim(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4UniqueDim begin");

    // Validate dim attribute
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() >= 1) {
        const int64_t *dim = attrs->GetAttrPointer<int64_t>(0);
        if (dim != nullptr && *dim != 0) {
            OP_LOGE(context->GetNodeName(), "UniqueDim only supports dim=0, got dim=%ld", *dim);
            return GRAPH_FAILED;
        }
    }

    const gert::Shape *xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape *yShape = context->GetOutputShape(OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape *idxShape = context->GetOutputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    gert::Shape *countShape = context->GetOutputShape(COUNTS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, countShape);

    if (IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set shape to (-2)");
        SetUnknownRank(*yShape);
        SetUnknownRank(*idxShape);
        SetUnknownRank(*countShape);
        return GRAPH_SUCCESS;
    }

    size_t xDimNum = xShape->GetDimNum();
    int64_t xSize = xShape->GetShapeSize();
    if (xSize == 0) {
        // empty input → all outputs are empty
        yShape->SetDimNum(xDimNum);
        for (size_t i = 0; i < xDimNum; i++) {
            yShape->SetDim(i, 0);
        }
        idxShape->SetDimNum(1);
        idxShape->SetDim(0, 0);
        countShape->SetDimNum(1);
        countShape->SetDim(0, 0);
    } else {
        // y: same rank as input, dim[0] unknown (numOut depends on computation)
        yShape->SetDimNum(xDimNum);
        SetUnknownShape(xDimNum, *yShape);

        // idx (inverse): [numInp], always equals input dim[0]
        idxShape->SetDimNum(1);
        idxShape->SetDim(0, xShape->GetDim(0));

        // count: [numOut], unknown until computation
        countShape->SetDimNum(1);
        countShape->SetDim(0, -1);
    }

    OP_LOGD(context->GetNodeName(), "InferShape4UniqueDim end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4UniqueDim(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferDtype4UniqueDim begin");
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(OUT_IDX, inputDtype);
    context->SetOutputDataType(INDICES_IDX, ge::DT_INT64);
    context->SetOutputDataType(COUNTS_IDX, ge::DT_INT64);
    OP_LOGD(context->GetNodeName(), "InferDtype4UniqueDim end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4UniqueDim(gert::InferShapeRangeContext *context)
{
    auto xRange = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    auto yRange = context->GetOutputShapeRange(OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange);
    auto idxRange = context->GetOutputShapeRange(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxRange);
    auto countRange = context->GetOutputShapeRange(COUNTS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, countRange);

    OP_CHECK_NULL_WITH_CONTEXT(context, xRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, idxRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, idxRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, countRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, countRange->GetMin());

    // y range: same rank as input, dim[0] in [0, xDim0]
    yRange->GetMax()->SetDimNum(xRange->GetMax()->GetDimNum());
    yRange->GetMin()->SetDimNum(xRange->GetMin()->GetDimNum());
    for (size_t i = 0; i < xRange->GetMax()->GetDimNum(); i++) {
        yRange->GetMax()->SetDim(i, xRange->GetMax()->GetDim(i));
        yRange->GetMin()->SetDim(i, 0);
    }

    // idx range: [0, xDim0]
    idxRange->GetMax()->SetDimNum(1);
    idxRange->GetMin()->SetDimNum(1);
    idxRange->GetMax()->SetDim(0, xRange->GetMax()->GetDim(0));
    idxRange->GetMin()->SetDim(0, 0);

    // count range: [0, xDim0]
    countRange->GetMax()->SetDimNum(1);
    countRange->GetMin()->SetDimNum(1);
    countRange->GetMax()->SetDim(0, xRange->GetMax()->GetDim(0));
    countRange->GetMin()->SetDim(0, 0);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UniqueDim).InferShape(InferShape4UniqueDim)
                          .InferDataType(InferDtype4UniqueDim)
                          .InferShapeRange(InferShapeRange4UniqueDim)
                          .OutputShapeDependOnCompute({OUT_IDX, INDICES_IDX, COUNTS_IDX});

}  // namespace ops
