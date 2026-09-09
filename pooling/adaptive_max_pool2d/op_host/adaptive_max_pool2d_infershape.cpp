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
 * \file adaptive_max_pool2d_infershape.cpp
 * \brief
 */

#include <string>
#include <vector>

#include "error_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t INDEX_OUT_ARGMAX = 1;
constexpr size_t INDEX_OUTPUT_SIZE = 0;
constexpr size_t INDEX_DTYPE = 1;
constexpr int64_t INT32_DTYPE = 3;
constexpr int64_t MIN_INPUT_DIMS = 3;
constexpr int64_t MAX_INPUT_DIMS = 4;
constexpr int64_t OUTPUT_SIZE_DIMS = 2;
constexpr int64_t KEEP_DIMS = 2;
constexpr const char* kOpName = "AdaptiveMaxPool2d";
} // namespace

namespace ops {
static ge::graphStatus InferShape4AdaptiveMaxPool2d(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "runtime2.0 AdaptiveMaxPool2d infershape running");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* argmax_shape = context->GetOutputShape(INDEX_OUT_ARGMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, argmax_shape);

    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    const gert::ContinuousVector* output_size_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(INDEX_OUTPUT_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_size_ptr);

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        Ops::Base::SetUnknownRank(*argmax_shape);
        return ge::GRAPH_SUCCESS;
    }

    size_t input_dim_num = x_shape->GetDimNum();
    if (Ops::Base::IsUnknownShape(*x_shape)) {
        Ops::Base::SetUnknownShape(input_dim_num, *y_shape);
        Ops::Base::SetUnknownShape(input_dim_num, *argmax_shape);
        return ge::GRAPH_SUCCESS;
    }

    if (input_dim_num != MIN_INPUT_DIMS && input_dim_num != MAX_INPUT_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "x", std::to_string(input_dim_num).c_str(), "3 or 4");
        return ge::GRAPH_FAILED;
    }

    size_t output_size_len = output_size_ptr->GetSize();
    if (output_size_len != OUTPUT_SIZE_DIMS) {
        OP_LOGE_FOR_INVALID_LISTSIZE(kOpName, "output_size", std::to_string(output_size_len).c_str(), "2");
        return ge::GRAPH_FAILED;
    }

    y_shape->SetDimNum(0);
    argmax_shape->SetDimNum(0);
    for (size_t i = 0; i < input_dim_num - KEEP_DIMS; i++) {
        y_shape->AppendDim(x_shape->GetDim(i));
        argmax_shape->AppendDim(x_shape->GetDim(i));
    }
    const int64_t* output_size = static_cast<const int64_t*>(output_size_ptr->GetData());
    for (size_t i = 0; i < OUTPUT_SIZE_DIMS; i++) {
        y_shape->AppendDim(output_size[i]);
        argmax_shape->AppendDim(output_size[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4AdaptiveMaxPool2d(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "AdaptiveMaxPool2dInferDtype enter");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType x_dtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, x_dtype);

    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    bool is_ascend950 = (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                             platform_info, optional_info) == ge::GRAPH_SUCCESS) &&
                        (platform_info.str_info.short_soc_version == "Ascend950");
    if (!is_ascend950) {
        context->SetOutputDataType(INDEX_OUT_ARGMAX, ge::DT_INT64);
        OP_LOGD(context->GetNodeName(), "AdaptiveMaxPool2dInferDtype end");
        return ge::GRAPH_SUCCESS;
    }

    // A5 (Ascend950): use argmax_dtype attribute (3 -> INT32, otherwise INT64).
    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    const int64_t* argmax_dtype = attr_ptr->GetAttrPointer<int64_t>(INDEX_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, argmax_dtype);
    ge::DataType out1_dtype = (*argmax_dtype == INT32_DTYPE) ? ge::DT_INT32 : ge::DT_INT64;
    context->SetOutputDataType(INDEX_OUT_ARGMAX, out1_dtype);
    OP_LOGD(context->GetNodeName(), "AdaptiveMaxPool2dInferDtype end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveMaxPool2d)
    .InferShape(InferShape4AdaptiveMaxPool2d)
    .InferDataType(InferDtype4AdaptiveMaxPool2d);
} // namespace ops
