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
 * \file in_infer_v2_infershape.cpp
 * \brief INInferV2 inferShape：y shape = x shape；batch_mean shape = mean shape；
 *        batch_variance shape = variance shape。
 *        与 canndev built-in 三条 ELMTWISE_INFER_SHAPEANDTYPE("x","y") /
 *        ("mean","batch_mean") / ("variance","batch_variance") 功能完全一致，
 *        gert infershape 2.0 写法
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace ops {

static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_MEAN_INDEX = 3;
static constexpr size_t INPUT_VAR_INDEX = 4;
static constexpr size_t OUTPUT_Y_INDEX = 0;
static constexpr size_t OUTPUT_BATCH_MEAN_INDEX = 1;
static constexpr size_t OUTPUT_BATCH_VAR_INDEX = 2;

static ge::graphStatus INInferV2InferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do INInferV2InferShape");

    // optional 输入缺席时框架会压实输入存储，必须用 def 声明序访问器（GetOptionalInputShape/
    // GetRequiredInputShape）；mean/variance 在 proto/def 层为 optional，此处 null 检查即
    // 910b TBE REQUIRED 语义的拦截点
    const gert::Shape* xShape = context->GetRequiredInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* meanShape = context->GetOptionalInputShape(INPUT_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    const gert::Shape* varShape = context->GetOptionalInputShape(INPUT_VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);

    const auto* yInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yInstanceInfo);
    OP_CHECK_IF(yInstanceInfo->GetInstanceNum() != 1,
                OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "y", yInstanceInfo->GetInstanceNum(), "1"),
                return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(yInstanceInfo->GetInstanceStart());
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto* batchMeanInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_BATCH_MEAN_INDEX);
    gert::Shape* batchMeanShape = nullptr;
    if (batchMeanInstanceInfo != nullptr) {
        OP_CHECK_IF(batchMeanInstanceInfo->GetInstanceNum() > 1,
                    OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "batch_mean",
                                                  batchMeanInstanceInfo->GetInstanceNum(), "0 or 1"),
                    return ge::GRAPH_FAILED);
    }
    if (batchMeanInstanceInfo != nullptr && batchMeanInstanceInfo->GetInstanceNum() == 1) {
        batchMeanShape = context->GetOutputShape(batchMeanInstanceInfo->GetInstanceStart());
        OP_CHECK_NULL_WITH_CONTEXT(context, batchMeanShape);
    }

    const auto* batchVarInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_BATCH_VAR_INDEX);
    gert::Shape* batchVarShape = nullptr;
    if (batchVarInstanceInfo != nullptr) {
        OP_CHECK_IF(batchVarInstanceInfo->GetInstanceNum() > 1,
                    OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "batch_variance",
                                                  batchVarInstanceInfo->GetInstanceNum(), "0 or 1"),
                    return ge::GRAPH_FAILED);
    }
    if (batchVarInstanceInfo != nullptr && batchVarInstanceInfo->GetInstanceNum() == 1) {
        batchVarShape = context->GetOutputShape(batchVarInstanceInfo->GetInstanceStart());
        OP_CHECK_NULL_WITH_CONTEXT(context, batchVarShape);
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
    } else {
        *yShape = *xShape;
    }
    if (batchMeanShape != nullptr) {
        if (Ops::Base::IsUnknownRank(*meanShape)) {
            Ops::Base::SetUnknownRank(*batchMeanShape);
        } else {
            *batchMeanShape = *meanShape;
        }
    }
    if (batchVarShape != nullptr) {
        if (Ops::Base::IsUnknownRank(*varShape)) {
            Ops::Base::SetUnknownRank(*batchVarShape);
        } else {
            *batchVarShape = *varShape;
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do INInferV2InferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(INInferV2).InferShape(INInferV2InferShape);
} // namespace ops
