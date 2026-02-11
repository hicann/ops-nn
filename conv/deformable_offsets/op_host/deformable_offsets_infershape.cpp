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
 * \file deformable_offsets_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "op_common/op_host/util/opbase_export.h"
#include "register/op_impl_registry.h"
#include "runtime/storage_shape.h"

using namespace ge;
namespace ops {
constexpr size_t kDimNum = 4U;
constexpr size_t kDilationsSize = 4U;
constexpr size_t kStridesSize = 4U;
constexpr size_t kKSizeSize = 2U;
constexpr size_t kPadsSize = 4U;
constexpr size_t kNCHWN = 0U;
constexpr size_t kNCHWC = 1U;
constexpr size_t kNCHWH = 2U;
constexpr size_t kNCHWW = 3U;
constexpr size_t kNHWCN = 0U;
constexpr size_t kNHWCH = 1U;
constexpr size_t kNHWCW = 2U;
constexpr size_t kNHWCC = 3U;

static ge::graphStatus DeformableOffsetsInferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    auto stridesPtr = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesPtr);
    auto padsPtr = attrs->GetListInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    auto ksizePtr = attrs->GetListInt(2); // 2 is for ksize
    OP_CHECK_NULL_WITH_CONTEXT(context, ksizePtr);
    auto dilationsPtr = attrs->GetListInt(3); // 3 is for dilations
    OP_CHECK_NULL_WITH_CONTEXT(context, dilationsPtr);
    auto dataFormat = attrs->GetStr(4); // 4 is for dataFormat
    OP_CHECK_NULL_WITH_CONTEXT(context, dataFormat);
    auto dilations = dilationsPtr->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, dilations);
    if (dilationsPtr->GetSize() != kDilationsSize) {
        OP_LOGE(context->GetNodeName(), "dilations list size should be 4, but got %zu", dilationsPtr->GetSize());
        return ge::GRAPH_FAILED;
    }
    auto strides = stridesPtr->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    if (stridesPtr->GetSize() != kStridesSize) {
        OP_LOGE(context->GetNodeName(), "strides list size should be 4, but got %zu", stridesPtr->GetSize());
        return ge::GRAPH_FAILED;
    }
    int64_t dilationsH;
    int64_t dilationsW;
    int64_t strideH;
    int64_t strideW;
    // 这里的data_format 仅仅表示属性的format，已经不能表示input/output的format
    if (strcmp(dataFormat, "NCHW") == 0) {
        dilationsH = dilations[kNCHWH];
        dilationsW = dilations[kNCHWW];
        strideH = strides[kNCHWH];
        strideW = strides[kNCHWW];
    } else if (strcmp(dataFormat, "NHWC") == 0) {
        dilationsH = dilations[kNHWCH];
        dilationsW = dilations[kNHWCW];
        strideH = strides[kNHWCH];
        strideW = strides[kNHWCW];
    } else {
        OP_LOGE(context->GetNodeName(), "dataFormat attr only support NCHW or NHWC, but got %s", dataFormat);
        return ge::GRAPH_FAILED;
    }

    if ((strideH <= 0) || (strideW <= 0)) {
        OP_LOGE(
            context->GetNodeName(), "stride should be greater than 0, strideH [%ld], strideW [%ld]", strideH, strideW);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        (ksizePtr->GetSize() != kKSizeSize), OP_LOGE(context->GetNodeName(), "kSize list size should be 2"),
        return ge::GRAPH_FAILED);
    auto ksize = ksizePtr->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    auto ksizeH = ksize[0];
    auto ksizeW = ksize[1];

    auto dilKsizeH = (ksizeH - 1) * dilationsH + 1;
    auto dilKsizeW = (ksizeW - 1) * dilationsW + 1;

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(
        (xShape->GetDimNum() != kDimNum), OP_LOGE(context->GetNodeName(), "x rank should be 4D"),
        return ge::GRAPH_FAILED);
    const gert::Shape* offsetShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetShape);
    OP_CHECK_IF(
        (offsetShape->GetDimNum() != kDimNum), OP_LOGE(context->GetNodeName(), "offset rank should be 4D"),
        return ge::GRAPH_FAILED);
    auto posH = strchr(dataFormat, 'H') - dataFormat;
    auto posW = strchr(dataFormat, 'W') - dataFormat;

    auto xH = xShape->GetDim(posH);
    auto xW = xShape->GetDim(posW);
    auto offsetH = offsetShape->GetDim(posH);
    auto offsetW = offsetShape->GetDim(posW);
    auto pads = padsPtr->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(
        (padsPtr->GetSize() != kPadsSize), OP_LOGE(context->GetNodeName(), "pads list size should be 4"),
        return ge::GRAPH_FAILED);
    auto padU = pads[0];
    auto padD = pads[1];
    auto padL = pads[2];
    auto padR = pads[3];

    auto convOutH = (xH + padU + padD - dilKsizeH) / strideH + 1;
    auto convOutW = (xW + padL + padR - dilKsizeW) / strideW + 1;

    if ((convOutH != offsetH) || (convOutW != offsetW)) {
        OP_LOGE(
            context->GetNodeName(),
            "Input_offsets h/w should be same as h/w after convolution, but now offset: [h:%ld, w:%ld]. conv_out: "
            "[h:%ld, w:%ld].",
            offsetH, offsetW, convOutH, convOutW);
        return ge::GRAPH_FAILED;
    }

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    *outputShape = *xShape;
    outputShape->SetDim(posH, offsetH * ksizeH);
    outputShape->SetDim(posW, offsetW * ksizeW);
    OP_LOGD(
        context->GetNodeName(), "x shape is %s, offset shape is %s, output shape is %s, dataFormat is %s",
        Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*offsetShape).c_str(),
        Ops::Base::ToString(*outputShape).c_str(), dataFormat);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DeformableOffsetsInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "DeformableOffsetsInferDataType begin");
    auto inputXDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputXDtype);
    OP_LOGD(context->GetNodeName(), "DeformableOffsetsInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DeformableOffsets)
    .InferShape(DeformableOffsetsInferShape)
    .InferDataType(DeformableOffsetsInferDataType);
} // namespace ops