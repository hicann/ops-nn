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
 * \file depthwise_conv2d_backprop_input_infershape.cpp
 * \brief InferShape registration for legacy DepthwiseConv2DBackpropInput, which is converted to
 *        Conv3DBackpropInputV2 via a fusion pass. The shared infer helpers are reused from
 *        conv_backprop_infershape.h in conv/common.
 */

#include "common/op_host/conv_backprop_infershape.h"
#include "common/op_host/conv_common_cube_util.h"

namespace Ops {
namespace NN {
namespace Conv {

static constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kInputSizeIndex = 0;
constexpr size_t kFilterIndex = 1;
constexpr size_t kOutBackpropIndex = 2;

// depthwisedx校验规格：
// input_size为非const时：filter_shape不支持-1/-2，out_backprop shape支持-1/-2
// input_size为const时：filter_shape不支持-1/-2，out_backprop shape支持-1，不支持-2
static ge::graphStatus CheckShapeForDepthwiseConv2DBackpropInput(gert::InferShapeContext* context)
{
    // filter不支持-1/-2，与input_size是否const、out_backprop是否动态均无关
    const auto filterShape = context->GetInputShape(kFilterIndex);
    OP_CHECK_IF(filterShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter shape."),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*filterShape)) {
        OP_LOGE(context->GetNodeName(), "filter shape = [-2] is not supported.");
        return ge::GRAPH_FAILED;
    }
    for (size_t idx = 0; idx < filterShape->GetDimNum(); ++idx) {
        const int64_t filterDim = filterShape->GetDim(idx);
        if (filterDim < 0) {
            OP_LOGE(context->GetNodeName(), "filter shape dim[%zu] = %ld should be positive, -1 is not supported.", idx,
                    filterDim);
            return ge::GRAPH_FAILED;
        }
    }

    // out_backprop：input_size为const时不支持[-2]（支持-1）；input_size为非const时支持-1/-2
    const auto inputSizeTensor = context->GetInputTensor(kInputSizeIndex);
    OP_CHECK_IF(inputSizeTensor == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get input_size tensor."),
                return ge::GRAPH_FAILED);
    const auto& inputSizeShape = inputSizeTensor->GetOriginShape();
    const bool isInputSizeConst = IsConstSizeTensor(inputSizeTensor) && !Ops::Base::IsUnknownRank(inputSizeShape) &&
                                  !Ops::Base::IsUnknownShape(inputSizeShape);
    if (!isInputSizeConst) {
        return ge::GRAPH_SUCCESS;
    }
    const auto outBackpropShape = context->GetInputShape(kOutBackpropIndex);
    OP_CHECK_IF(outBackpropShape == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get out_backprop shape."),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*outBackpropShape)) {
        OP_LOGE(context->GetNodeName(), "out_backprop shape = [-2] is not supported when input_size is const.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Conv2DBackpropInput(gert::InferShapeContext* context)
{
    auto ret = InferShapeForConvBackprop(context, 0, "input_size", kConv2dDimSizeLimit);
    if (ret == ge::GRAPH_SUCCESS) {
        ret = CheckShapeForDepthwiseConv2DBackpropInput(context);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        auto yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            OP_LOGD(context->GetNodeName(), "[InferShape] Conv2DBackpropInput y_shape: %s",
                    Ops::Base::ToString(*yShape).c_str());
        }
    }
    return ret;
}

} // namespace Conv

IMPL_OP_INFERSHAPE(DepthwiseConv2DBackpropInput)
    .InferShape(Ops::NN::Conv::InferShape4Conv2DBackpropInput)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConvBackpropInputV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", -1L);

} // namespace NN
} // namespace Ops
