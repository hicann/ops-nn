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
 * \file lp_norm_v2_infershape.cpp
 * \brief
 */

#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static constexpr size_t INDEX_AXES = 1;
static constexpr size_t INDEX_KEEPDIM = 2;

static ge::graphStatus InferShape4LpNormV2(gert::InferShapeContext *context) {
    const auto xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto axes = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(INDEX_AXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, axes);
    const bool* keepdim = attrs->GetAttrPointer<bool>(INDEX_KEEPDIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, keepdim);

    std::vector<int64_t> yVec;
    std::vector<int64_t> newAxes;
    auto dimSize = xShape->GetDimNum();
    if (axes->GetSize() == 0UL) {
        for (size_t i = 0UL; i < dimSize; i++) {
            newAxes.push_back(static_cast<int64_t>(i));
        }
    } else {
        for (size_t i = 0UL; i < axes->GetSize(); i++) {
            const int64_t tmpAxes = (axes->GetData())[i];
            const int64_t realAxes = (tmpAxes < 0) ? (tmpAxes + static_cast<int64_t>(dimSize)) : tmpAxes;
            newAxes.push_back(realAxes);
        }
    }
    for (size_t i = 0UL; i < dimSize; i++) {
        if (find(newAxes.begin(), newAxes.end(), i) != newAxes.end()) {
            if (*keepdim) {
                yVec.push_back(1);
            }
        } else {
            yVec.push_back(xShape->GetDim(i));
        }
    }

    int64_t lenRes = yVec.size();
    yShape->SetDimNum(lenRes);
    for (int64_t i = 0; i < lenRes; i++) {
        yShape->SetDim(i, yVec[i]);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4LpNormV2(gert::InferDataTypeContext *context) {
    OP_LOGD(context->GetNodeName(), "InferDataType4LpNorm start");
    auto inputXDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputXDtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4LpNorm end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LpNormV2).InferShape(InferShape4LpNormV2).InferDataType(InferDataType4LpNormV2);
} // namespace ops