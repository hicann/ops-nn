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
 * \file depthwise_conv2d_backprop_filter_infershape.cpp
 * \brief InferShape registration for legacy DepthwiseConv2DBackpropFilter, which is converted to
 *        Conv3DBackpropFilterV2 via a fusion pass. The shared infer helpers are reused from
 *        conv_backprop_infershape.h in conv/common.
 */

#include "common/op_host/conv_backprop_infershape.h"
#include "common/op_host/conv_common_cube_util.h"

namespace Ops {
namespace NN {
namespace Conv {

static constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kInputIndex = 0;
constexpr size_t kFilterSizeIndex = 1;
constexpr size_t kOutBackpropIndex = 2;

// filter_size为const且input为静态shape时，out_backprop的H/W维度不支持-1
static ge::graphStatus CheckOutBackpropShapeForDepthwiseConv2DBackpropFilter(gert::InferShapeContext* context)
{
    const auto filterSizeTensor = context->GetInputTensor(kFilterSizeIndex);
    OP_CHECK_IF(filterSizeTensor == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter_size tensor."),
                return ge::GRAPH_FAILED);
    const auto xShape = context->GetInputShape(kInputIndex);
    OP_CHECK_IF(xShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get x shape."),
                return ge::GRAPH_FAILED);
    const auto& filterSizeShape = filterSizeTensor->GetOriginShape();
    if (!IsConstSizeTensor(filterSizeTensor) || Ops::Base::IsUnknownRank(filterSizeShape) ||
        Ops::Base::IsUnknownShape(filterSizeShape) || !IsStaticShape(*xShape)) {
        return ge::GRAPH_SUCCESS;
    }

    const auto outBackpropShape = context->GetInputShape(kOutBackpropIndex);
    OP_CHECK_IF(outBackpropShape == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get out_backprop shape."),
                return ge::GRAPH_FAILED);
    const auto outBackpropDesc = context->GetInputDesc(kOutBackpropIndex);
    OP_CHECK_IF(outBackpropDesc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get out_backprop tensor desc."),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*outBackpropShape)) {
        return ge::GRAPH_SUCCESS;
    }
    size_t hIndex = 0;
    size_t wIndex = 0;
    if (outBackpropShape->GetDimNum() != kConv2dDimSizeLimit ||
        !GetHwDimIndex(outBackpropDesc->GetOriginFormat(), hIndex, wIndex)) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t outBackpropH = outBackpropShape->GetDim(hIndex);
    const int64_t outBackpropW = outBackpropShape->GetDim(wIndex);
    if (outBackpropH < 0 || outBackpropW < 0) {
        OP_LOGE(context->GetNodeName(),
                "out_backprop shape h = %ld, w = %ld should be positive when filter_size is const and input shape is "
                "static.",
                outBackpropH, outBackpropW);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4DepthwiseConv2DBackpropFilter(gert::InferShapeContext* context)
{
    const auto filter_size_tensor = context->GetInputTensor(kFilterSizeIndex);
    OP_CHECK_IF(filter_size_tensor == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get null filter_size tensor"), return ge::GRAPH_FAILED);
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y shape is null"),
                return ge::GRAPH_FAILED);
    y_shape->SetDimNum(kConv2dDimSizeLimit);

    auto is_const_tensor = [](const gert::Tensor* t) -> bool {
        if (t == nullptr) {
            return false;
        }
        if (t->GetAddr() == nullptr) {
            return t->GetShapeSize() == 0;
        }
        return true;
    };
    if (is_const_tensor(filter_size_tensor)) {
        const auto dtype = filter_size_tensor->GetDataType();
        if (dtype == ge::DT_INT32) {
            const auto data = filter_size_tensor->GetData<int32_t>();
            for (size_t idx = 0; idx < kConv2dDimSizeLimit; ++idx) {
                y_shape->SetDim(idx, data[idx]);
            }
        } else if (dtype == ge::DT_INT64) {
            const auto data = filter_size_tensor->GetData<int64_t>();
            for (size_t idx = 0; idx < kConv2dDimSizeLimit; ++idx) {
                y_shape->SetDim(idx, data[idx]);
            }
        } else {
            CUBE_INNER_ERR_REPORT(context->GetNodeName(), "filter_size dtype %s not support, only int32/int64.",
                                  ge::TypeUtils::DataTypeToAscendString(dtype).GetString());
            return ge::GRAPH_FAILED;
        }
    } else {
        for (size_t idx = 0; idx < kConv2dDimSizeLimit; ++idx) {
            y_shape->SetDim(idx, -1);
        }
    }
    OP_LOGD(context->GetNodeName(), "[InferShape] DepthwiseConv2DBackpropFilter y_shape: %s",
            Ops::Base::ToString(*y_shape).c_str());
    return CheckOutBackpropShapeForDepthwiseConv2DBackpropFilter(context);
}

} // namespace Conv

IMPL_OP_INFERSHAPE(DepthwiseConv2DBackpropFilter)
    .InferShape(Ops::NN::Conv::InferShape4DepthwiseConv2DBackpropFilter)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConv2DBackpropFilter)
    .InputsDataDependency({1})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);

} // namespace NN
} // namespace Ops
