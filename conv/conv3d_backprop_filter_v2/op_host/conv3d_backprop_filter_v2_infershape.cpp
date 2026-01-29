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
 * \file conv3d_backprop_filter_v2_infershape.cpp
 * \brief
 */
#include "common/op_host/conv_backprop_infershape.h"


namespace Ops {
namespace NN {
namespace Conv {

constexpr size_t kConv3dDimSizeLimit = 5;
constexpr size_t kConv2dDimSizeLimit = 4;
static ge::graphStatus InferShapeForConv3DBackpropFilter(gert::InferShapeContext* context)
{
    auto const_tensor = context->GetInputTensor(1);
    OP_CHECK_IF(
        const_tensor == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get null tensor"),
        return ge::GRAPH_FAILED);
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());

    auto ret = ge::GRAPH_SUCCESS;
    if (const_tensor_dim_num == kConv2dDimSizeLimit) {
        ret = InferShapeForConvBackpropExtend3D(context, 1, "filter_size");
    } else {
        ret = InferShapeForConvBackprop(context, 1, "filter_size", kConv3dDimSizeLimit);
    }
    return ret;
}

static ge::graphStatus InferDataTypeForConv3DBackpropFilterV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConv3DBackpropFilterV2 enter");
    ge::graphStatus ret = context->SetOutputDataType(0, ge::DT_FLOAT);
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferDataTypeForConv3DBackpropFilterV2 enter");
    return ge::GRAPH_SUCCESS;
}
}  // namespace Conv

IMPL_OP_INFERSHAPE(Conv3DBackpropFilterV2)
    .InferShape(Ops::NN::Conv::InferShapeForConv3DBackpropFilter)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConv3DBackpropFilterV2)
    .InputsDataDependency({1})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);
}  // namespace NN
}  // namespace Ops
