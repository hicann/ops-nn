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
 * \file quant_batch_matmul_inplace_add_infershape.cpp
 * \brief
 */
#include "common/op_host/matmul_common_infershape.h"
#include "runtime/infer_datatype_context.h"


namespace {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t X1_SCALE_INDEX = 2;
constexpr uint32_t YREF_INDEX = 3;
constexpr uint32_t X2_SCALE_INDEX = 4;
constexpr size_t QUANT_BATCH_MATMUL_V3_MIN_SHAPE_SIZE = 2;

static ge::graphStatus IsInputTensorNull(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(X1_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(X2_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(X2_SCALE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(YREF_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOptionalInputShape(X1_SCALE_INDEX));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4QuantBatchMatmulInplaceAdd(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_CHECK_IF(
        IsInputTensorNull(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Some of the required inputs are null."), return ge::GRAPH_FAILED);

    auto shape_x1 = context->GetInputShape(X1_INDEX);
    auto shape_x2 = context->GetInputShape(X2_INDEX);
    auto dim_a = shape_x1->GetDimNum();
    auto dim_b = shape_x2->GetDimNum();
    bool any_unknow_rank = Ops::NN::CheckIsUnknownDimNum(*shape_x1) || Ops::NN::CheckIsUnknownDimNum(*shape_x2);
    if (!any_unknow_rank &&
        (dim_a != QUANT_BATCH_MATMUL_V3_MIN_SHAPE_SIZE || dim_b != QUANT_BATCH_MATMUL_V3_MIN_SHAPE_SIZE)) {
        OP_LOGE(context->GetNodeName(), "[InferShape] The shape can only be in the range of 2.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(
        context->GetNodeName(), "x1_shape: %s", "x2_shape: %s",
        Ops::Base::ToString(*shape_x1).c_str(), Ops::Base::ToString(*shape_x2).c_str());

    auto yShape = context->GetInputShape(YREF_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *yShape;
    OP_LOGD(context->GetNodeName(), "output shape: %s", Ops::Base::ToString(*(context->GetOutputShape(0))).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4QuantBatchMatmulInplaceAdd(gert::InferDataTypeContext *context) 
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(QuantBatchMatmulInplaceAdd)
    .InferShape(InferShape4QuantBatchMatmulInplaceAdd)
    .InferDataType(InferDataType4QuantBatchMatmulInplaceAdd);
}
