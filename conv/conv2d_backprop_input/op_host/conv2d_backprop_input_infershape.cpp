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
 * \file conv2d_backprop_input_infershape.cpp
 * \brief InferShape registration for legacy Conv2DBackpropInput, which is converted to
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

// input_size为const且out_backprop为静态shape时，filter的H/W维度不支持-1/-2
static ge::graphStatus CheckFilterShapeForConv2DBackpropInput(gert::InferShapeContext* context)
{
    const auto inputSizeTensor = context->GetInputTensor(kInputSizeIndex);
    OP_CHECK_IF(inputSizeTensor == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get input_size tensor."),
                return ge::GRAPH_FAILED);
    const auto outBackpropShape = context->GetInputShape(kOutBackpropIndex);
    OP_CHECK_IF(outBackpropShape == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get out_backprop shape."),
                return ge::GRAPH_FAILED);
    const auto& inputSizeShape = inputSizeTensor->GetOriginShape();
    if (!IsConstSizeTensor(inputSizeTensor) || Ops::Base::IsUnknownRank(inputSizeShape) ||
        Ops::Base::IsUnknownShape(inputSizeShape) || !IsStaticShape(*outBackpropShape)) {
        return ge::GRAPH_SUCCESS;
    }

    const auto filterShape = context->GetInputShape(kFilterIndex);
    OP_CHECK_IF(filterShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter shape."),
                return ge::GRAPH_FAILED);
    const auto filterDesc = context->GetInputDesc(kFilterIndex);
    OP_CHECK_IF(filterDesc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter tensor desc."),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*filterShape)) {
        OP_LOGE(context->GetNodeName(),
                "filter shape [-2] should be 4 dims and h, w should be positive when input_size is const and "
                "out_backprop shape is static.");
        return ge::GRAPH_FAILED;
    }
    size_t hIndex = 0;
    size_t wIndex = 0;
    if (filterShape->GetDimNum() != kConv2dDimSizeLimit ||
        !GetHwDimIndex(filterDesc->GetOriginFormat(), hIndex, wIndex)) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t filterH = filterShape->GetDim(hIndex);
    const int64_t filterW = filterShape->GetDim(wIndex);
    if (filterH < 0 || filterW < 0) {
        OP_LOGE(context->GetNodeName(),
                "filter shape h = %ld, w = %ld should be positive when input_size is const and out_backprop shape is "
                "static.",
                filterH, filterW);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Conv2DBackpropInput(gert::InferShapeContext* context)
{
    auto ret = InferShapeForConvBackprop(context, 0, "input_size", kConv2dDimSizeLimit);
    if (ret == ge::GRAPH_SUCCESS) {
        ret = CheckFilterShapeForConv2DBackpropInput(context);
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

IMPL_OP_INFERSHAPE(Conv2DBackpropInput)
    .InferShape(Ops::NN::Conv::InferShape4Conv2DBackpropInput)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConvBackpropInputV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", -1L);

} // namespace NN
} // namespace Ops
