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
 * \file max_pool_v3_infershape.cpp
 * \brief max_pool_v3 shape and data type inference
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "max_pool_v3_util.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMaxPoolV3(gert::InferShapeContext* context)
{
    auto inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (inShape->GetDimNum() < SHAPE_4D_SIZE) {
        outShape->SetDimNum(SHAPE_4D_SIZE);
        for (size_t i = 0; i < SHAPE_4D_SIZE; i++) {
            outShape->SetDim(i, UNKNOWN_DIM_VALUE);
        }
        return GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Lambda: fetch and validate one int-array attribute.
    // Returns nullptr if the attribute is not present (only valid for optional attrs).
    auto getArr = [&](size_t idx, const char* name) -> const int64_t* {
        auto arr = attrs->GetAttrPointer<gert::ContinuousVector>(idx);
        if (arr == nullptr) {
            return nullptr;
        }
        if (arr->GetSize() != SHAPE_4D_SIZE) {
            OP_LOGE_FOR_INVALID_LISTSIZE("MaxPoolV3", name, std::to_string(arr->GetSize()).c_str(), "4");
            return nullptr;
        }
        return static_cast<const int64_t*>(arr->GetData());
    };
    auto ksizeData = getArr(INDEX_KSIZE, "Length of ksize");
    if (!ksizeData) {
        return GRAPH_FAILED;
    }
    auto stridesData = getArr(INDEX_STRIDES, "Length of strides");
    if (!stridesData) {
        return GRAPH_FAILED;
    }
    auto padsData = getArr(INDEX_PADS, "Length of pads");
    // pads is OPTIONAL (default {0,0,0,0}); fall back to DEFAULT_PADS if not provided
    if (!padsData) {
        padsData = DEFAULT_PADS;
    }

    auto ceilModeAttr = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
    bool ceilMode = (ceilModeAttr != nullptr) ? *ceilModeAttr : false;
    *outShape = *inShape;

    size_t hDim = 2, wDim = 3;
    if (!ValidateSpatialDims(ksizeData, stridesData, hDim, wDim, "MaxPoolV3")) {
        return GRAPH_FAILED;
    }
    outShape->SetDim(hDim, CalculateUpdateDim(ksizeData[hDim], padsData[PAD_TOP], padsData[PAD_BOTTOM],
                                              stridesData[hDim], ceilMode, inShape->GetDim(hDim)));
    outShape->SetDim(wDim, CalculateUpdateDim(ksizeData[wDim], padsData[PAD_LEFT], padsData[PAD_RIGHT],
                                              stridesData[wDim], ceilMode, inShape->GetDim(wDim)));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeMaxPoolV3(gert::InferDataTypeContext* context)
{
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolV3).InferShape(InferShapeMaxPoolV3).InferDataType(InferDtypeMaxPoolV3);
} // namespace ops
