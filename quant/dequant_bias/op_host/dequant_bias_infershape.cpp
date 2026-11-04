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
 * \file dequant_bias.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace ops {
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t SIZE_1 = 1;
static constexpr int64_t SIZE_27 = 27;

static ge::graphStatus InferShape4DequantBias(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4DequantBias");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "dequant_bias" : (context)->GetNodeName()),
            "x"),
        return ge::GRAPH_FAILED);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    if (yShape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "dequant_bias" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object yShape is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object yShape is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum != DIM_TWO) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "dequant_bias" : (context)->GetNodeName()));
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(msgOpName.c_str(), "x", std::to_string(xDimNum),
                                                 "Dims of input x should be 2.");
        return ge::GRAPH_FAILED;
    }
    yShape->SetDimNum(DIM_TWO);
    for (size_t i = 0; i < DIM_TWO; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }

    OP_LOGI(context->GetNodeName(), "DequantBias output shape: %s.", Ops::Base::ToString(*yShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShape4DequantBias");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4DequantBias(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4DequantBias");
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "dequant_bias" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    const int64_t* output_dtype = attrs->GetAttrPointer<int64_t>(0);
    const int64_t outdtype = (output_dtype == nullptr) ? 0 : *output_dtype;
    if (outdtype == SIZE_1) {
        context->SetOutputDataType(0, DT_FLOAT16);
    } else if (outdtype == SIZE_27) {
        context->SetOutputDataType(0, DT_BF16);
    } else {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "dequant_bias" : (context)->GetNodeName()));
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(msgOpName.c_str(), "output_dtype", std::to_string(outdtype),
                                              "Attr outputDtype should be 1 or 27.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4DequantBias");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantBias).InferShape(InferShape4DequantBias).InferDataType(InferDataType4DequantBias);
} // namespace ops
