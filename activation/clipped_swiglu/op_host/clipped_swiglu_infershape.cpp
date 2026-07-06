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
 * \file clipped_swiglu_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"
using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForClippedSwiglu(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForClippedSwiglu");
    auto xShape = context->GetInputShape(GLU_IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto yShape = context->GetOutputShape(GLU_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto splitDimPtr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, splitDimPtr);
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }
    auto splitDim = *splitDimPtr;
    if (splitDim < 0) {
        splitDim += xShape->GetDimNum();
    }
    if (splitDim < 0 || splitDim >= static_cast<int64_t>(xShape->GetDimNum())) {
        OP_LOGE("ClippedSwiglu", "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].",
                xShape->GetDimNum(), xShape->GetDimNum() - 1, splitDim);
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to generate yShape");
    *yShape = *xShape;
    // dynamic shape
    if (xShape->GetDim(splitDim) == -1) {
        return ge::GRAPH_SUCCESS;
    }
    if (xShape->GetDim(splitDim) < 0 || xShape->GetDim(splitDim) % SPLIT_NUM != 0) {
        OP_LOGE("ClippedSwiglu", "The shape [%s] is not divisible by 2.", Ops::Base::ToString(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    yShape->SetDim(splitDim, xShape->GetDim(splitDim) / SPLIT_NUM);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForClippedSwiglu");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForClippedSwiglu(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForClippedSwiglu");
    const ge::DataType dtype = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForClippedSwiglu");
    return ret;
}

IMPL_OP_INFERSHAPE(ClippedSwiglu).InferShape(InferShapeForClippedSwiglu).InferDataType(InferDataTypeForClippedSwiglu);
} // namespace ops
