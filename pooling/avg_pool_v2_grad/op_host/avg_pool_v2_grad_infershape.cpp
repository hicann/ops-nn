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
 * \file avg_pool_v2_grad_infershape.cpp
 * \brief
 */

#include <string>
#include "error_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

static constexpr size_t IDX_ORIGIN_INPUT = 0;
static constexpr size_t IDX_GRAD_INPUT = 1;
static constexpr size_t IDX_OUTPUT = 0;
static constexpr size_t IDX_ZERO = 0;
static constexpr size_t IDX_ONE = 1;
static constexpr size_t IDX_THREE = 3;
static constexpr size_t CHW_DIMS = 3;
static constexpr size_t NCHW_DIMS = 4;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 4;
static constexpr size_t ATTR_KERNEL_POS = 0;
static constexpr size_t ATTR_STRIDE_POS = 1;
static constexpr size_t ATTR_PADDING_MODE_POS = 2;
static constexpr size_t ATTR_PADS_POS = 3;
static constexpr size_t ATTR_FORMAT_POS = 4;
static constexpr size_t ATTR_GLOBAL_POOLING_POS = 5;
static constexpr size_t ATTR_CEIL_MODE_POS = 6;
static constexpr size_t ATTR_EXCLUSIVE_POS = 7;
static constexpr size_t ATTR_DIVISOR_OVERRIDE_POS = 8;
static constexpr size_t ONE = 1;
static constexpr int32_t UNKNOWN_SHAPE_DIM = -1;
static constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

inline bool IsConstTensor(const gert::Tensor* inputTensor)
{
    if (inputTensor != nullptr) {
        if (inputTensor->GetAddr() == nullptr) {
            return inputTensor->GetShapeSize() == 0;
        }
        return true;
    }
    return false;
}

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape4AvgPoolV2Grad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 AvgPoolV2Grad infershape running");

    auto gradDesc = context->GetInputDesc(IDX_GRAD_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    auto gradOriFormat = gradDesc->GetOriginFormat();

    if (gradOriFormat != FORMAT_ND && gradOriFormat != FORMAT_NCHW && gradOriFormat != FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMAT(
            context->GetNodeName(), "grad", Ops::Base::ToString(gradOriFormat).c_str(), "ND, NCHW and NHWC");
        return GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(ATTR_FORMAT_POS);
    OP_LOGE_IF(dataFormatPtr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get dataFormat failed.");
    std::string dataFormatStr(dataFormatPtr);

    auto padding_mode = attrs->GetAttrPointer<char>(ATTR_PADDING_MODE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, padding_mode);
    if (strcmp(padding_mode, "SAME") != 0 && strcmp(padding_mode, "VALID") != 0 &&
        strcmp(padding_mode, "CALCULATED") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding_mode", padding_mode, "SAME, VALID and CALCULATED");
        return GRAPH_FAILED;
    }

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    if (ksize->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "ksize", std::to_string(ksize->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());

    if (dataFormatStr == "NCHW") {
        if (ksize_data[IDX_ZERO] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "ksize[0]", std::to_string(ksize_data[IDX_ZERO]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (ksize_data[IDX_ONE] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "ksize[1]", std::to_string(ksize_data[IDX_ONE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    } else if (dataFormatStr == "NHWC") {
        if (ksize_data[IDX_ZERO] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "ksize[0]", std::to_string(ksize_data[IDX_ZERO]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (ksize_data[IDX_THREE] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "ksize[3]", std::to_string(ksize_data[IDX_THREE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    }

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    if (strides->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            context->GetNodeName(), "strides", std::to_string(strides->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto strides_data = static_cast<const int64_t*>(strides->GetData());

    if (dataFormatStr == "NCHW") {
        if (strides_data[IDX_ZERO] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "strides[0]", std::to_string(strides_data[IDX_ZERO]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (strides_data[IDX_ONE] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "strides[1]", std::to_string(strides_data[IDX_ONE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    } else if (dataFormatStr == "NHWC") {
        if (strides_data[IDX_ZERO] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "strides[0]", std::to_string(strides_data[IDX_ZERO]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (strides_data[IDX_THREE] != ONE) {
            OP_LOGE_FOR_INVALID_VALUE(
                context->GetNodeName(), "strides[3]", std::to_string(strides_data[IDX_THREE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    }

    const gert::Tensor* inputShape0 = context->GetInputTensor(IDX_ORIGIN_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape0);
    size_t inputDimNum = static_cast<size_t>(inputShape0->GetOriginShape().GetShapeSize());
    const int32_t* shapeValue = inputShape0->GetData<int32_t>();
    if (inputDimNum != CHW_DIMS && inputDimNum != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "input", std::to_string(inputDimNum).c_str(), "input dim num should be 3 or 4");
        return GRAPH_FAILED;
    }
    const gert::Shape* inputShape1 = context->GetInputShape(IDX_ORIGIN_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape1);

    gert::Shape* OutShape = context->GetOutputShape(IDX_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, OutShape);

    if (Ops::Base::IsUnknownShape(*inputShape1) || !IsConstTensor(inputShape0)) {
        SetAllUnknownDim(inputDimNum, OutShape);
    }

    if (Ops::Base::IsUnknownRank(*inputShape1)) {
        Ops::Base::SetUnknownRank(*OutShape);
        return ge::GRAPH_SUCCESS;
    }

    OutShape->SetDimNum(inputDimNum);

    for (size_t idx = 0; idx < inputDimNum; ++idx) {
        OutShape->SetDim(idx, shapeValue[idx]);
    }

    OP_LOGD(context->GetNodeName(), "runtime2.0 end AvgPoolV2Grad infershape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4AvgPoolV2Grad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AvgPoolV2Grad)
    .InputsDataDependency({0})
    .InferShape(InferShape4AvgPoolV2Grad)
    .InferDataType(InferDataType4AvgPoolV2Grad);
} // namespace ops