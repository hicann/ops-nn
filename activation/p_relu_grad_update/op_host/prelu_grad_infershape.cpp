/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_info.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/infershape_broadcast_util.h"

using namespace ge;
namespace ops {

constexpr size_t GRADS_INDEX = 0;
constexpr size_t FEATURES_INDEX = 1;
constexpr size_t WEIGHTS_INDEX = 2;
constexpr size_t DX_INDEX = 0;
constexpr size_t DA_INDEX = 1;
constexpr size_t UPDATE_INDEX = 1;

// PReluGrad: dx = features shape/dtype, da = weights shape/dtype
static ge::graphStatus InferShape4PReluGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4PReluGrad begin");
    auto featuresShape = context->GetInputShape(FEATURES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    auto weightsShape = context->GetInputShape(WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);

    auto dxShape = context->GetOutputShape(DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    *dxShape = *featuresShape;

    auto daShape = context->GetOutputShape(DA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, daShape);
    *daShape = *weightsShape;
    OP_LOGD(context->GetNodeName(), "InferShape4PReluGrad end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4PReluGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGrad begin");
    auto featuresDtype = context->GetInputDataType(FEATURES_INDEX);
    auto weightsDtype = context->GetInputDataType(WEIGHTS_INDEX);
    context->SetOutputDataType(DX_INDEX, featuresDtype);
    context->SetOutputDataType(DA_INDEX, weightsDtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGrad end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PReluGrad).InferShape(InferShape4PReluGrad).InferDataType(InferDataType4PReluGrad);

// PReluGradUpdate: dx/update = BroadcastShape(grads, features), dx/update dtype = grads dtype
static ge::graphStatus InferShape4PReluGradUpdate(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4PReluGradUpdate begin");
    auto gradsShape = context->GetInputShape(GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsShape);
    auto featuresShape = context->GetInputShape(FEATURES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);

    auto dxShape = context->GetOutputShape(DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    auto updateShape = context->GetOutputShape(UPDATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, updateShape);

    OP_CHECK_IF(!Ops::Base::BroadcastShape(gradsShape, featuresShape, dxShape),
                OP_LOGE(context->GetNodeName(), "grads shape %s and features shape %s cannot broadcast.",
                        Ops::Base::ToString(*gradsShape).c_str(), Ops::Base::ToString(*featuresShape).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!Ops::Base::BroadcastShape(gradsShape, featuresShape, updateShape),
                OP_LOGE(context->GetNodeName(), "grads shape %s and features shape %s cannot broadcast.",
                        Ops::Base::ToString(*gradsShape).c_str(), Ops::Base::ToString(*featuresShape).c_str()),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShape4PReluGradUpdate end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4PReluGradUpdate(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGradUpdate begin");
    auto gradsDtype = context->GetInputDataType(GRADS_INDEX);
    context->SetOutputDataType(DX_INDEX, gradsDtype);
    context->SetOutputDataType(UPDATE_INDEX, gradsDtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4PReluGradUpdate end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PReluGradUpdate).InferShape(InferShape4PReluGradUpdate).InferDataType(InferDataType4PReluGradUpdate);
} // namespace ops
