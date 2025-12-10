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
 * \file dynamic_quant_update_scatter_v2_infershape.cpp
 * \brief
 */

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT_UPDATE_SCATTER_V2 = 3;

static ge::graphStatus CheckComputeNodeNumMerged(const gert::InferShapeContext* context)
{
    if (context->GetComputeNodeOutputNum() != OUTPUT_NUM_DYNAMIC_QUANT_UPDATE_SCATTER_V2) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context)
{
    if (context == nullptr || CheckComputeNodeNumMerged(context) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < OUTPUT_NUM_DYNAMIC_QUANT_UPDATE_SCATTER_V2; ++i) {
        if (context->GetOutputShape(i) == nullptr) {
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantUpdateScatterV2InferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do Infershape of DynamicQuantUpdateScatterV2InferShape.");
    if (InferShapeCheck(context) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* varShape = context->GetInputShape(2);
    const gert::Shape* varScaleShape = context->GetInputShape(3);
    const gert::Shape* varOffsetShape = context->GetInputShape(4);
    gert::Shape* varOutShape = context->GetOutputShape(0);
    gert::Shape* varScaleOutShape = context->GetOutputShape(1);
    gert::Shape* varOffsetOutShape = context->GetOutputShape(2);

    varOutShape->SetDimNum(xShape->GetDimNum());
    varScaleOutShape->SetDimNum(xShape->GetDimNum());
    varOffsetOutShape->SetDimNum(xShape->GetDimNum());
    for (uint32_t i = 0; i < xShape->GetDimNum(); ++i) {
        if (i < xShape->GetDimNum() - 1) {
            varOutShape->SetDim(i, varShape->GetDim(i));
        } else {
            varOutShape->SetDim(i, xShape->GetDim(i));
        }
        if (i == 0) {
            varScaleOutShape->SetDim(i, 1);
            varOffsetOutShape->SetDim(i, 1);
        } else {
            varScaleOutShape->SetDim(i, varScaleShape->GetDim(i - 1));
            varOffsetOutShape->SetDim(i, varOffsetShape->GetDim(i - 1));
        }
    }

    OP_LOGD(context, "End to do DynamicQuantUpdateScatterV2InferShape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantUpdateScatterV2InferDataType(gert::InferDataTypeContext* context)
{
    static constexpr int32_t OUTPUT_IDX_VAR = 0;
    static constexpr int32_t OUTPUT_IDX_SCALE = 1;
    static constexpr int32_t OUTPUT_IDX_OFFSET = 2;

    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context, "DynamicQuantUpdateScatterV2InferDataType begin");
    context->SetOutputDataType(OUTPUT_IDX_VAR, ge::DT_INT4);
    context->SetOutputDataType(OUTPUT_IDX_SCALE, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_OFFSET, ge::DT_FLOAT);
    OP_LOGD(context, "DynamicQuantUpdateScatterV2InferDataType end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicQuantUpdateScatterV2)
    .InferShape(DynamicQuantUpdateScatterV2InferShape)
    .InferDataType(DynamicQuantUpdateScatterV2InferDataType);
} // namespace ops
