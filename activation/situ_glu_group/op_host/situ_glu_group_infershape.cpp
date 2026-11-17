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
 * \file situ_glu_group_infershape.cpp
 * \brief InferShape for SituGluGroup: y equals x with the dim axis halved.
 */

#include <string>
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
static ge::graphStatus InferShapeForSituGluGroup(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start InferShape for SituGluGroup");
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
    // GetDimNum() returns size_t; convert once so the negative-dim normalization and the range
    // message never rely on unsigned wraparound.
    const int64_t xDimNum = static_cast<int64_t>(xShape->GetDimNum());
    auto splitDim = *splitDimPtr;
    if (splitDim < 0) {
        splitDim += xDimNum;
    }
    if (splitDim < 0 || splitDim >= xDimNum) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dim", std::to_string(splitDim),
                                              "The value of dim must be in the range [-" + std::to_string(xDimNum) +
                                                  ", " + std::to_string(xDimNum - 1) + "]");
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Start generating yShape");
    *yShape = *xShape;
    // Dynamic shape.
    if (xShape->GetDim(splitDim) == -1) {
        return ge::GRAPH_SUCCESS;
    }
    if (xShape->GetDim(splitDim) < 0 || xShape->GetDim(splitDim) % SPLIT_NUM != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape),
                                              "The size of x at dim must be divisible by 2");
        return GRAPH_FAILED;
    }

    yShape->SetDim(splitDim, xShape->GetDim(splitDim) / SPLIT_NUM);
    OP_LOGD(context->GetNodeName(), "Finish InferShape for SituGluGroup");
    return ge::GRAPH_SUCCESS;
}

// InferDataType is implemented in op_graph/situ_glu_group_graph_infer.cpp.
IMPL_OP_INFERSHAPE(SituGluGroup).InferShape(InferShapeForSituGluGroup);
} // namespace ops
