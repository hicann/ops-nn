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
 * \file max_pool3d_grad_infershape_arch35.cpp
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
static constexpr size_t KSIZE_ATTR_INDEX = 0U;
static constexpr size_t STRIDES_ATTR_INDEX = 1U;
static constexpr size_t PADDING_ATTR_INDEX = 2U;
static constexpr size_t PADS_ATTR_INDEX = 3U;
static constexpr size_t DATA_FORMAT_ATTR_INDEX = 4U;
static constexpr size_t INDEX_OUT_MAX = 0;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 5;
static constexpr size_t INDEX_ZERO = 0;
static constexpr size_t INDEX_ONE = 1;
static constexpr size_t INDEX_TWO = 2;
static constexpr size_t CDHW_DIM_NUM = 4;
static constexpr size_t NUMBER_TWO = 2;
static constexpr size_t PARAM_NUM = 4;
static constexpr size_t SHAPE_D_DIM = 2;
static constexpr size_t SHAPE_H_DIM = 3;
static constexpr size_t SHAPE_W_DIM = 4;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttrInfo(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(ksize->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %lu must be equal 5!", ksize->GetSize()), return GRAPH_FAILED);

    auto ksize_data = reinterpret_cast<const int64_t*>(ksize->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(ksize->GetSize()); i++) {
        OP_CHECK_IF((ksize_data[i] <= 0), OP_LOGE(context->GetNodeName(), "Attr value invalid, ksize_data[%u] is %ld, should bigger than 0.", i, ksize_data[i]),
            return ge::GRAPH_FAILED);
    }

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(strides->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %lu must be equal 5!", strides->GetSize()), return GRAPH_FAILED);

    auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(strides->GetSize()); i++) {
        OP_CHECK_IF((strides_data[i] <= 0),
            OP_LOGE(context->GetNodeName(), "Attr value invalid, strides_data[%u] is %ld, should bigger than 0.", i, strides_data[i]), return ge::GRAPH_FAILED);
    }

    auto padsPtr = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    std::string padding(padsPtr);
    OP_CHECK_IF(padding != "SAME" && padding != "VALID" && padding != "CALCULATED",
             OP_LOGE(context->GetNodeName(), "Pads attritube must be 'SAME' or 'VALID' or 'CALCULATED'!"), return GRAPH_FAILED);

    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(pads->GetSize() != 0 && pads->GetSize() != 1 && pads->GetSize() != ATTR_LIST_SHAPE_SIZE,
            OP_LOGE(context->GetNodeName(), "Length of pads %lu must be equal 0, 1 or 3!", pads->GetSize()), return GRAPH_FAILED);

    auto pads_data = reinterpret_cast<const int64_t*>(pads->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(pads->GetSize()); i++) {
        OP_CHECK_IF((pads_data[i] < 0), OP_LOGE(context->GetNodeName(), "Attr value invalid, pads_data[%u] is %ld, should bigger or equal 0.", i, pads_data[i]),
            return ge::GRAPH_FAILED);
    }

    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_format);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMaxPool3DGrad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape running");
    auto inputXDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXDesc);

    auto ret =CheckAttrInfo(context);
    OP_CHECK_IF(
        ret != GRAPH_SUCCESS, OP_LOGD("InferShapeForMaxPool3DGrad", "CheckAttrInfo return unsuccess"),
        return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* origYShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, origYShape);
    const gert::Shape* gradsShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    size_t xDimNum = xShape->GetDimNum();

    if (Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownShape(*origYShape) || Ops::Base::IsUnknownShape(*gradsShape)) {
        SetAllUnknownDim(xDimNum, yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape handle unknown shape.");
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape handle unknown rank.");
        return ge::GRAPH_SUCCESS;
    }
    yShape->SetDimNum(xDimNum);
    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape run success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMaxPool3DGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(INDEX_OUT_MAX, xDtype);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPool3DGrad)
    .InferShape(InferShapeForMaxPool3DGrad)
    .InferDataType(InferDataTypeForMaxPool3DGrad);
} // namespace ops