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
 * \file max_pool_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
constexpr size_t SHAPE_NHWC_SIZE = 4;
constexpr size_t INSHAPE_DIM_NUM = 4;
constexpr size_t INPUT_KSIZE_IDX = 1;
constexpr size_t INPUT_STRIDES_IDX = 2;

using InferShapePaddingFunc = ge::graphStatus (*)(gert::InferShapeContext*, size_t, size_t);

static int64_t SameUpdateDim(const int64_t ksize, const int64_t strides, int64_t dimSize)
{
    if (dimSize == UNKNOWN_DIM_VALUE_) {
        return UNKNOWN_DIM_VALUE_;
    }
    return (strides == 0) ? (dimSize - ksize + 1) : ((dimSize - ksize + strides) / strides);
}

static ge::graphStatus InferShapePaddingValid(
    gert::InferShapeContext* context, size_t hDim, size_t wDim)
{
    auto inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *inShape;
    auto ksize = context->GetInputTensor(INPUT_KSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetShapeSize() != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %ld must be 4!", ksize->GetShapeSize()), return GRAPH_FAILED);
    int64_t hksizeData = static_cast<int64_t>(ksize->GetData<int32_t>()[hDim]);
    int64_t wksizeData = static_cast<int64_t>(ksize->GetData<int32_t>()[wDim]);
    auto strides = context->GetInputTensor(INPUT_STRIDES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetShapeSize() != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %ld must be 4!", strides->GetShapeSize()), return GRAPH_FAILED);
    int64_t hstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[hDim]);
    int64_t wstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[wDim]);
    OP_CHECK_IF(
        hstridesData <= 0 || wstridesData <= 0,
        OP_LOGE(context->GetNodeName(), "MaxPoolV2: not support stride shape [%ld, %ld]", hstridesData, wstridesData),
        return GRAPH_FAILED);

    int64_t dimSize = inShape->GetDim(hDim);
    outShape->SetDim(hDim, SameUpdateDim(hksizeData, hstridesData, dimSize));
    dimSize = inShape->GetDim(wDim);
    outShape->SetDim(wDim, SameUpdateDim(wksizeData, wstridesData, dimSize));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingSame(
    gert::InferShapeContext* context, size_t hDim, size_t wDim)
{
    auto inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *inShape;
    auto strides = context->GetInputTensor(INPUT_STRIDES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        static_cast<int64_t>(strides->GetShapeSize()) != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %ld must be 4!", static_cast<int64_t>(strides->GetShapeSize())), return GRAPH_FAILED);
    int64_t hstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[hDim]);
    int64_t wstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[wDim]);
    OP_CHECK_IF(
        hstridesData <= 0 || wstridesData <= 0,
        OP_LOGE(context->GetNodeName(), "MaxPoolV2: not support stride shape [%ld, %ld]", hstridesData, wstridesData),
        return GRAPH_FAILED);
    int64_t dimSize = inShape->GetDim(hDim);
    outShape->SetDim(hDim, SameUpdateDim(1, hstridesData, dimSize));
    dimSize = inShape->GetDim(wDim);
    outShape->SetDim(wDim, SameUpdateDim(1, wstridesData, dimSize));
    return ge::GRAPH_SUCCESS;
}

static const std::vector<std::pair<std::string, InferShapePaddingFunc>> kFuncMap = {
    {"SAME", InferShapePaddingSame},
    {"VALID", InferShapePaddingValid},
};

static ge::graphStatus InferShape4MaxPoolV2(gert::InferShapeContext* context)
{
    auto inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    if (Ops::Base::IsUnknownRank(*inShape)) {
        Ops::Base::SetUnknownShape(INSHAPE_DIM_NUM, *outShape);
        return ge::GRAPH_SUCCESS;
    }

    auto srcTd = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcTd);
    auto inputFormat = srcTd->GetOriginFormat();
    size_t hDim = inputFormat == FORMAT_NHWC ? 1 : 2;
    size_t wDim = inputFormat == FORMAT_NHWC ? 2 : 3;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto paddingMode = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddingMode);
    auto it = std::find_if(
        kFuncMap.begin(), kFuncMap.end(), 
        [&paddingMode](const std::pair<std::string, InferShapePaddingFunc>& item)->bool{
          return item.first == paddingMode;
        });
    OP_CHECK_IF(
        it == kFuncMap.end(),
        OP_LOGE(context->GetNodeName(), "paddingMode %s must in (VALID, SAME).", paddingMode),
        return GRAPH_FAILED);

    // when paddingMode in (VALID, SAME)
    return it->second(context, hDim, wDim);
}

static ge::graphStatus InferDtype4MaxPoolV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "MaxPoolV2InferDtype enter");
    // Get input tout
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);

    OP_LOGD(context->GetNodeName(), "MaxPoolV2InferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolV2).InputsDataDependency({INPUT_KSIZE_IDX, INPUT_STRIDES_IDX}).InferShape(InferShape4MaxPoolV2).InferDataType(InferDtype4MaxPoolV2);
}  // namespace ops
