/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <string>
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_PREDICT = 0;
static constexpr int64_t IDX_LABEL = 1;
static constexpr int64_t IDX_DOUT = 2;
static constexpr int64_t IDX_OUTPUT = 0;

static std::string GetReductionAttr(gert::InferShapeContext* context)
{
    std::string reduction = "mean";
    if (context == nullptr) {
        return reduction;
    }
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        // Attr order in op_def: sigma(0), reduction(1)
        auto attr = attrs->GetStr(1);
        if (attr == nullptr) {
            // UT may provide only reduction as the sole attr at index 0
            attr = attrs->GetStr(0);
        }
        if (attr != nullptr) {
            reduction = *attr;
        }
    }
    std::transform(reduction.begin(), reduction.end(), reduction.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (reduction == "n")
        reduction = "none";
    else if (reduction == "s")
        reduction = "sum";
    else if (reduction == "m")
        reduction = "mean";
    return reduction;
}

static ge::graphStatus InferShapeSmoothL1LossGradV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSmoothL1LossGradV2");

    const gert::Shape* predictShape = context->GetInputShape(IDX_PREDICT);
    const gert::Shape* labelShape = context->GetInputShape(IDX_LABEL);
    const gert::Shape* doutShape = context->GetInputShape(IDX_DOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, doutShape);

    OP_CHECK_IF(
        predictShape->GetDimNum() != labelShape->GetDimNum(),
        OP_LOGE(
            context, "Predict and label rank mismatch: %zu vs %zu", predictShape->GetDimNum(), labelShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < predictShape->GetDimNum(); ++i) {
        OP_CHECK_IF(
            predictShape->GetDim(i) != labelShape->GetDim(i),
            OP_LOGE(
                context, "Shape mismatch at dim %zu: %ld vs %ld", i, predictShape->GetDim(i), labelShape->GetDim(i)),
            return ge::GRAPH_FAILED);
    }

    std::string reduction = GetReductionAttr(context);
    bool doutScalar = (doutShape->GetDimNum() == 0) || (doutShape->GetDimNum() == 1 && doutShape->GetDim(0) == 1);
    std::cout << "lnlninfershape";

    if (reduction == "sum" || reduction == "mean") {
        if (!doutScalar) {
            OP_LOGE(
                context, "For reduction='%s', dout must be scalar (0D or 1D length 1), but got %zuD", reduction.c_str(),
                doutShape->GetDimNum());
            return GRAPH_FAILED;
        }
    } else if (reduction == "none") {
        if (doutShape->GetDimNum() != predictShape->GetDimNum()) {
            OP_LOGE(
                context, "For reduction='none', dout rank must match predict rank: %zu vs %zu", doutShape->GetDimNum(),
                predictShape->GetDimNum());
            return GRAPH_FAILED;
        }
        for (size_t i = 0; i < predictShape->GetDimNum(); ++i) {
            if (doutShape->GetDim(i) != predictShape->GetDim(i)) {
                OP_LOGE(
                    context, "For reduction='none', dout shape mismatch at dim %zu: %ld vs %ld", i,
                    doutShape->GetDim(i), predictShape->GetDim(i));
                return GRAPH_FAILED;
            }
        }
    } else {
        OP_LOGE(context, "Invalid reduction: %s", reduction.c_str());
        return GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(IDX_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *predictShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSmoothL1LossGradV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SmoothL1LossGradV2).InferShape(InferShapeSmoothL1LossGradV2);

} // namespace ops
