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
 * \file swi_glu_infer.cc
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForSwiGlu(gert::InferShapeContext* context)
{
    auto x_shape = context->GetInputShape(GLU_IN_X);
    OP_CHECK_IF(
        x_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "swi_glu" : (context)->GetNodeName()), "x"),
        return ge::GRAPH_FAILED);
    auto y_shape = context->GetOutputShape(GLU_OUT_Y);
    if (y_shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "swi_glu" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y_shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y_shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "swi_glu" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    if (split_dim_ptr == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "swi_glu" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object split_dim_ptr is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object split_dim_ptr is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto split_dim = *split_dim_ptr;
    if (split_dim < 0) {
        split_dim += x_shape->GetDimNum();
    }

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        OP_LOGD("SwiGlu", "inputs have unknown rank, no need check.");
        *y_shape = *x_shape;
        return ge::GRAPH_SUCCESS;
    }

    if (split_dim < 0 || split_dim >= static_cast<int64_t>(x_shape->GetDimNum())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "SwiGlu", "dim", std::to_string(*split_dim_ptr),
            ("The value of attr [dim] must be in the range [-" + std::to_string(x_shape->GetDimNum()) + ", " +
             std::to_string(x_shape->GetDimNum() - 1) + "], but got [" + std::to_string(split_dim) + "]."));
        return GRAPH_FAILED;
    }

    *y_shape = *x_shape;
    // dynamic shape
    if (x_shape->GetDim(split_dim) == -1) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t shapeAfter = 1;
    bool hasUnknownDim = false;
    for (int64_t j = split_dim; j < static_cast<int64_t>(x_shape->GetDimNum()); j++) {
        if (x_shape->GetDim(j) < 0) {
            hasUnknownDim = true;
            break;
        }
        shapeAfter *= x_shape->GetDim(j);
    }
    if (!hasUnknownDim && shapeAfter % SPLIT_NUM != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "SwiGlu", "x", Ops::Base::ToString(*x_shape),
            ("The shape [" + std::string(Ops::Base::ToString(*x_shape).c_str()) + "] is not divisible by 2."));
        return ge::GRAPH_FAILED;
    }

    y_shape->SetDim(split_dim, x_shape->GetDim(split_dim) / SPLIT_NUM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwiGlu(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    return ret;
}

IMPL_OP_INFERSHAPE(SwiGlu).InferShape(InferShapeForSwiGlu).InferDataType(InferDataTypeForSwiGlu);
} // namespace ops
