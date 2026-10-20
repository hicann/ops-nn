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
 * \file softmax_focal_loss_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include <algorithm>
#include "platform/platform_info.h"
#include <string>
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr uint64_t INPUT_PRED_IDX = 0;
constexpr uint64_t INPUT_TARGET_IDX = 1;
constexpr uint64_t OUTPUT_Y_IDX = 0;
constexpr size_t ATTR_REDUCTION_IDX = 2; // gamma(0), alpha(1), reduction(2) —— 与 def.cpp 声明序一致
} // namespace

namespace ops {
static graphStatus InferShape4SoftmaxFocalLoss(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4SoftmaxFocalLoss.");
    const gert::Shape* predShape = context->GetInputShape(INPUT_PRED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, predShape);
    const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (!Ops::Base::IsUnknownRank(*predShape) && !Ops::Base::IsUnknownRank(*targetShape)) {
        OP_CHECK_IF(
            predShape->GetDimNum() != targetShape->GetDimNum(),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "target", std::to_string(targetShape->GetDimNum()), "must equal pred.dimNum"),
            return ge::GRAPH_FAILED);
    }

    // 输出形状由 reduction 决定: "none" 时 y 与 pred 同形; "mean"/"sum" 为标量。
    // 当前只支持 "none", 其余取值显式报错, 不静默按 "none" 推导。
    // 本文件对所有 soc 生效, 故按 soc 门控: 仅 Ascend950 读 reduction 并拒收未支持取值,
    // 其余 soc 保持原行为(y 与 pred 同形、不读该属性)。
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    const bool isAscend950 = (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                                  platformInfo, optionalInfo) == ge::GRAPH_SUCCESS) &&
                             (platformInfo.str_info.short_soc_version == "Ascend950");
    if (isAscend950) {
        const auto* attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const char* reduction = (attrs->GetAttrNum() > ATTR_REDUCTION_IDX) ?
                                    attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX) :
                                    nullptr;
        // 属性缺省时框架按原型物化为 "mean"; 取不到指针时按**声明默认值**处理, 不得回落成 "none"
        std::string red = (reduction == nullptr) ? std::string("mean") : std::string(reduction);
        std::transform(red.begin(), red.end(), red.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        OP_CHECK_IF(red != "none",
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "reduction", red.c_str(),
                        "only \"none\" is supported currently; mean/sum would reduce y to a scalar"),
                    return ge::GRAPH_FAILED);
    }
    *yShape = *predShape;
    OP_LOGD(context, "InferShape4SoftmaxFocalLoss End.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4SoftmaxFocalLoss(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDataType4SoftmaxFocalLoss Begin.");
    context->SetOutputDataType(OUTPUT_Y_IDX, context->GetInputDataType(INPUT_PRED_IDX));
    OP_LOGD(context, "InferDataType4SoftmaxFocalLoss End.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftmaxFocalLoss)
    .InferShape(InferShape4SoftmaxFocalLoss)
    .InferDataType(InferDataType4SoftmaxFocalLoss);
} // namespace ops
