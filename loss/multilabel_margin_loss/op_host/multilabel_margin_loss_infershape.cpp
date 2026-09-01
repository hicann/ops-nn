/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#include <cstring>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

namespace ops {

static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2LL;
const size_t DIM_NUM_TWO = 2;

static inline bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1 && shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

static inline void SetUnknownRank(gert::Shape* shape)
{
    shape->SetDimNum(0);
    shape->AppendDim(UNKNOWN_RANK_DIM_VALUE);
}

static ge::graphStatus InferShapeMultilabelMarginLoss(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* is_target_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, is_target_shape);

    // -2 UNKNOWN_RANK 传播:A2(ascend910b/ascend910_93)基线 infershape 无此处理,走 else=基线原样(不动 A2);
    // A5(regbase/ascend950)才补动态 rank 守护。soc 门控的 else 分支即 A2 基线行为,A2 逻辑等效不变。
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    bool isRegBase = (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                          platform_info, optional_info) == ge::GRAPH_SUCCESS &&
                      platform_info.str_info.short_soc_version == "Ascend950");
    if (isRegBase && IsUnknownRank(x1_shape)) {
        SetUnknownRank(y_shape);
        SetUnknownRank(is_target_shape);
        return GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // reduction is a String attr ("none"/"mean"/"sum"); consistent with tiling and the aclnn launcher.
    const char* reductionStr = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionStr);

    // reduction == none: per-sample loss, output 1D (N) for a 2D (N, C) input,
    // scalar for a 1D (C) input. reduction == mean/sum: scalar. Dynamic dims (-1) propagate via SetDim.
    if (strcmp(reductionStr, "none") == 0 && x1_shape->GetDimNum() >= DIM_NUM_TWO) {
        y_shape->SetDimNum(1);
        y_shape->SetDim(0, x1_shape->GetDim(0));
    } else {
        y_shape->SetDimNum(0);
    }

    *is_target_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MultilabelMarginLoss(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "InferDataType4MultilabelMarginLoss enter");
    // y 与 x 同类型
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // is_target 为 INT32，与 target 同类型（不能跟随 x）
    context->SetOutputDataType(1, context->GetInputDataType(1));
    OP_LOGD(context->GetNodeName(), "InferDataType4MultilabelMarginLoss end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MultilabelMarginLoss)
    .InferShape(InferShapeMultilabelMarginLoss)
    .InferDataType(InferDataType4MultilabelMarginLoss);
} // namespace ops
