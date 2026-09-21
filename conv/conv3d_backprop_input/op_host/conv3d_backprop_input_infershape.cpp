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
 * \file conv3d_backprop_input_infershape.cpp
 * \brief InferShape registration for legacy Conv3DBackpropInput, which is converted to
 *        Conv3DBackpropInputV2 via a fusion pass. The shared infer helpers are reused from
 *        conv_backprop_infershape.h in conv/common.
 */

#include "common/op_host/conv_backprop_infershape.h"

namespace Ops {
namespace NN {
namespace Conv {

static constexpr size_t kConv2dDimSizeLimit = 4;
static constexpr size_t kConv3dDimSizeLimit = 5;
static constexpr size_t kInputSizeIndex = 0;

static ge::graphStatus InferShape4Conv3DBackpropInput(gert::InferShapeContext* context)
{
    auto const_tensor = context->GetInputTensor(kInputSizeIndex);
    OP_CHECK_IF(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get null tensor"),
                return ge::GRAPH_FAILED);
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());

    auto ret = ge::GRAPH_SUCCESS;
    if (const_tensor_dim_num == kConv2dDimSizeLimit) {
        ret = InferShapeForConvBackpropExtend3D(context, 0, "input_size");
    } else {
        ret = InferShapeForConvBackprop(context, 0, "input_size", kConv3dDimSizeLimit);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        auto yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            OP_LOGD(context->GetNodeName(), "[InferShape] Conv3DBackpropInput y_shape: %s",
                    Ops::Base::ToString(*yShape).c_str());
        }
    }
    return ret;
}

} // namespace Conv

IMPL_OP_INFERSHAPE(Conv3DBackpropInput)
    .InferShape(Ops::NN::Conv::InferShape4Conv3DBackpropInput)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConvBackpropInputV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);

} // namespace NN
} // namespace Ops
