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
 * \file max_pool_v3_grad_infershape.cpp
 * \brief Shape inference implementation for MaxPoolV3Grad.
 */

#include <cstddef>
#include <cstdint>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "util/shape_util.h"

namespace ops {

constexpr size_t ORIG_INPUT_INDEX = 0;
constexpr size_t ORIG_OUTPUT_INDEX = 1;
constexpr size_t GRAD_INDEX = 2;
constexpr size_t OUT_GRAD_INDEX = 0;

constexpr size_t ATTR_INDEX_KSIZE = 0;
constexpr size_t ATTR_INDEX_STRIDES = 1;
constexpr size_t ATTR_INDEX_PADDING_MODE = 2;
constexpr size_t ATTR_INDEX_PADS = 3;
constexpr size_t ATTR_INDEX_DATA_FORMAT = 4;
constexpr size_t ATTR_INDEX_GLOBAL_POOLING = 5;
constexpr size_t ATTR_INDEX_CEIL_MODE = 6;

constexpr size_t EXPECTED_RANK = 4;
constexpr size_t ATTR_LIST_SIZE = 4;

static ge::graphStatus CheckAttrsValid(const gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksizeAttr = attrs->GetListInt(ATTR_INDEX_KSIZE);
    auto stridesAttr = attrs->GetListInt(ATTR_INDEX_STRIDES);
    auto padsAttr = attrs->GetListInt(ATTR_INDEX_PADS);

    OP_CHECK_NULL_WITH_CONTEXT(context, ksizeAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsAttr);

    if (ksizeAttr->GetSize() != ATTR_LIST_SIZE || stridesAttr->GetSize() != ATTR_LIST_SIZE ||
        padsAttr->GetSize() != ATTR_LIST_SIZE) {
        OP_LOGE(context->GetNodeName(), "ksize, strides and pads must contain four elements.");
        return ge::GRAPH_FAILED;
    }

    const int64_t* ksize = ksizeAttr->GetData();
    const int64_t* strides = stridesAttr->GetData();
    const int64_t* pads = padsAttr->GetData();

    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);

    const char* paddingMode = attrs->GetAttrPointer<char>(ATTR_INDEX_PADDING_MODE);
    const char* dataFormat = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    const bool* globalPooling = attrs->GetAttrPointer<bool>(ATTR_INDEX_GLOBAL_POOLING);
    const bool* ceilMode = attrs->GetAttrPointer<bool>(ATTR_INDEX_CEIL_MODE);

    OP_CHECK_NULL_WITH_CONTEXT(context, paddingMode);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataFormat);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalPooling);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceilMode);

    const std::string paddingModeString(paddingMode);
    const std::string dataFormatString(dataFormat);

    if (dataFormatString != "NCHW" && dataFormatString != "NHWC") {
        OP_LOGE(context->GetNodeName(), "data_format must be NCHW or NHWC.");
        return ge::GRAPH_FAILED;
    }

    if (paddingModeString != "CALCULATED" && paddingModeString != "SAME" && paddingModeString != "VALID") {
        OP_LOGE(context->GetNodeName(), "padding_mode must be CALCULATED, SAME or VALID.");
        return ge::GRAPH_FAILED;
    }

    const size_t channelIndex = dataFormatString == "NCHW" ? 1 : 3;
    const size_t heightIndex = dataFormatString == "NCHW" ? 2 : 1;
    const size_t widthIndex = dataFormatString == "NCHW" ? 3 : 2;

    if (strides[0] != 1 || strides[channelIndex] != 1) {
        OP_LOGE(context->GetNodeName(), "The N/C dimensions of strides must be 1.");
        return ge::GRAPH_FAILED;
    }

    if (strides[heightIndex] < 1 || strides[heightIndex] > 63 || strides[widthIndex] < 1 || strides[widthIndex] > 63) {
        OP_LOGE(context->GetNodeName(), "The H/W dimensions of strides must be in [1, 63].");
        return ge::GRAPH_FAILED;
    }

    if (!(*globalPooling)) {
        if (ksize[0] != 1 || ksize[channelIndex] != 1) {
            OP_LOGE(context->GetNodeName(), "The N/C dimensions of ksize must be 1.");
            return ge::GRAPH_FAILED;
        }

        if (ksize[heightIndex] < 1 || ksize[heightIndex] > 255 || ksize[widthIndex] < 1 || ksize[widthIndex] > 255) {
            OP_LOGE(context->GetNodeName(), "The H/W dimensions of ksize must be in [1, 255].");
            return ge::GRAPH_FAILED;
        }

        if (paddingModeString == "CALCULATED" && (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0)) {
            OP_LOGE(context->GetNodeName(), "pads must be nonnegative under CALCULATED mode.");
            return ge::GRAPH_FAILED;
        }

        // 该约束与正向 MaxPoolV3 保持一致。
        if (paddingModeString == "CALCULATED" && (pads[0] >= ksize[heightIndex] || pads[1] >= ksize[heightIndex] ||
                                                  pads[2] >= ksize[widthIndex] || pads[3] >= ksize[widthIndex])) {
            OP_LOGE(context->GetNodeName(),
                    "pads must be less than the corresponding ksize under CALCULATED mode, "
                    "ksize H=%ld, W=%ld.",
                    ksize[heightIndex], ksize[widthIndex]);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 仅比较双方均已知的维度，未知维度（-1）不参与比较，不能误判为不一致
static bool IsDimCompatible(int64_t leftDim, int64_t rightDim)
{
    return leftDim < 0 || rightDim < 0 || leftDim == rightDim;
}

// -1 表示未知维，-2 表示未知 Rank，只有小于 -2 的维度值才非法
static bool HasIllegalDimValue(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < -2) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus HandleShape(const gert::InferShapeContext* context, const std::string& dataFormat,
                                   const gert::Shape* origInputShape, const gert::Shape* origOutputShape,
                                   const gert::Shape* gradShape, gert::Shape* outGradShape)
{
    const bool origInputUnknownRank = Ops::Base::IsUnknownRank(*origInputShape);
    const bool origOutputUnknownRank = Ops::Base::IsUnknownRank(*origOutputShape);
    const bool gradUnknownRank = Ops::Base::IsUnknownRank(*gradShape);

    // 已知 rank 的输入必须为 4D；含 -1 的 rank=3/5 仍非法，只有真正的 unknown rank([-2]) 放行
    if (!origInputUnknownRank && origInputShape->GetDimNum() != EXPECTED_RANK) {
        OP_LOGE(context->GetNodeName(), "orig_input must be a 4D tensor.");
        return ge::GRAPH_FAILED;
    }
    if (!origOutputUnknownRank && origOutputShape->GetDimNum() != EXPECTED_RANK) {
        OP_LOGE(context->GetNodeName(), "orig_output must be a 4D tensor.");
        return ge::GRAPH_FAILED;
    }
    if (!gradUnknownRank && gradShape->GetDimNum() != EXPECTED_RANK) {
        OP_LOGE(context->GetNodeName(), "grad must be a 4D tensor.");
        return ge::GRAPH_FAILED;
    }

    // 必须在 unknown rank 提前返回之前拦截，避免非法动态维度被误放行
    if (HasIllegalDimValue(*origInputShape) || HasIllegalDimValue(*origOutputShape) || HasIllegalDimValue(*gradShape)) {
        OP_LOGE(context->GetNodeName(), "input shape contains an invalid dimension (less than -2).");
        return ge::GRAPH_FAILED;
    }

    // out_grad 始终由 orig_input 决定：仅 orig_input 为 unknown rank 时输出才为 unknown rank
    if (origInputUnknownRank) {
        Ops::Base::SetUnknownRank(*outGradShape);
        OP_LOGD(context->GetNodeName(), "MaxPoolV3Grad infershape handle unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    *outGradShape = *origInputShape;

    if (!origOutputUnknownRank) {
        const size_t channelIndex = dataFormat == "NCHW" ? 1 : 3;
        if (!IsDimCompatible(origInputShape->GetDim(0), origOutputShape->GetDim(0)) ||
            !IsDimCompatible(origInputShape->GetDim(channelIndex), origOutputShape->GetDim(channelIndex))) {
            OP_LOGE(context->GetNodeName(), "orig_output N/C dimensions must match orig_input.");
            return ge::GRAPH_FAILED;
        }

        if (!gradUnknownRank) {
            for (size_t i = 0; i < EXPECTED_RANK; ++i) {
                if (!IsDimCompatible(origOutputShape->GetDim(i), gradShape->GetDim(i))) {
                    OP_LOGE(context->GetNodeName(), "grad shape must match orig_output shape.");
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }

    // ho/wo 输出尺寸公式由 Tiling 统一校验，InferShape 不重复推导
    OP_LOGD(context->GetNodeName(), "MaxPoolV3Grad infershape run success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMaxPoolV3Grad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolV3Grad infershape running");

    auto origInputDesc = context->GetInputDesc(ORIG_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, origInputDesc);
    const ge::Format origInputFormat = origInputDesc->GetOriginFormat();
    if (origInputFormat != ge::FORMAT_ND && origInputFormat != ge::FORMAT_NCHW && origInputFormat != ge::FORMAT_NHWC) {
        OP_LOGE(context->GetNodeName(), "orig_input format must be ND, NCHW or NHWC.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = CheckAttrsValid(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape* origInputShape = context->GetInputShape(ORIG_INPUT_INDEX);
    const gert::Shape* origOutputShape = context->GetInputShape(ORIG_OUTPUT_INDEX);
    const gert::Shape* gradShape = context->GetInputShape(GRAD_INDEX);
    gert::Shape* outGradShape = context->GetOutputShape(OUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, origInputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, origOutputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outGradShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* dataFormatAttr = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataFormatAttr);

    return HandleShape(context, std::string(dataFormatAttr), origInputShape, origOutputShape, gradShape, outGradShape);
}

IMPL_OP_INFERSHAPE(MaxPoolV3Grad).InferShape(InferShapeMaxPoolV3Grad);

} // namespace ops
