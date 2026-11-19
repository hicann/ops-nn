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
 * \file max_pool_grad_infershape.cpp
 * \brief
 */
#include <string>
#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {

static constexpr size_t X1_INDEX = 0;
static constexpr size_t X2_INDEX = 1;
static constexpr size_t GRAD_INDEX = 2;
static constexpr size_t Y_INDEX = 0;

static constexpr size_t KSIZE_ATTR_INDEX = 0;
static constexpr size_t STRIDES_ATTR_INDEX = 1;
static constexpr size_t PADDING_ATTR_INDEX = 2;
static constexpr size_t DATA_FORMAT_ATTR_INDEX = 3;

static constexpr size_t NCHW_DIM_NUM = 4;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static inline void SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
}

static ge::graphStatus CheckAttrsValid(const gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    auto ksize = attrs->GetListInt(KSIZE_ATTR_INDEX);
    OP_CHECK_IF(
        ksize == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ksize",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ksize->GetSize() != NCHW_DIM_NUM,
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "ksize", std::to_string(ksize->GetSize()).c_str(), "4"),
        return GRAPH_FAILED);

    auto strides = attrs->GetListInt(STRIDES_ATTR_INDEX);
    OP_CHECK_IF(
        strides == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "strides",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(strides->GetSize() != NCHW_DIM_NUM,
                OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "strides",
                                             std::to_string(strides->GetSize()).c_str(), "4"),
                return GRAPH_FAILED);

    const char* padding = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OP_CHECK_IF(
        padding == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "padding",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    std::string paddingStr = padding;
    OP_CHECK_IF(paddingStr != "VALID" && paddingStr != "SAME",
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding", paddingStr.c_str(), "VALID or SAME"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HandleUnknownShape(const gert::InferShapeContext* context, const gert::Shape* x1Shape,
                                          const gert::Shape* x2Shape, const gert::Shape* gradShape, gert::Shape* yShape)
{
    size_t x1DimNum = x1Shape->GetDimNum();
    if (Ops::Base::IsUnknownShape(*x1Shape) || Ops::Base::IsUnknownShape(*x2Shape) ||
        Ops::Base::IsUnknownShape(*gradShape)) {
        SetAllUnknownDim(x1DimNum, yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGrad infershape handle unknown rank or shape.");
        return ge::GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownRank(*x1Shape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }
    yShape->SetDimNum(x1DimNum);
    *yShape = *x1Shape;
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGrad infershape run success.");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMaxPoolGrad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGrad infershape running");

    auto x1Desc = context->GetInputDesc(X1_INDEX);
    OP_CHECK_IF(
        x1Desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x1",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto xOriFormat = x1Desc->GetOriginFormat();
    OP_CHECK_IF(xOriFormat != FORMAT_ND && xOriFormat != FORMAT_NCHW && xOriFormat != FORMAT_NHWC,
                OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x",
                                           ge::TypeUtils::FormatToSerialString(xOriFormat).c_str(), "ND, NCHW, NHWC"),
                return GRAPH_FAILED);

    ge::graphStatus ret = CheckAttrsValid(context);
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape* x1Shape = context->GetInputShape(X1_INDEX);
    const gert::Shape* x2Shape = context->GetInputShape(X2_INDEX);
    const gert::Shape* gradShape = context->GetInputShape(GRAD_INDEX);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(
        x1Shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x1",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        x2Shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x2",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gradShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "grad",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "yShape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);

    return HandleUnknownShape(context, x1Shape, x2Shape, gradShape, yShape);
}

IMPL_OP_INFERSHAPE(MaxPoolGrad).InferShape(InferShapeForMaxPoolGrad);
} // namespace ops
