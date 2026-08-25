/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kGradIndex = 0;
constexpr size_t kEpsIndex = 1;
constexpr size_t kSumGradRIndex = 0;
constexpr size_t kSumGradCIndex = 1;
constexpr size_t kSumGradRCIndex = 2;
constexpr size_t kMinGradRank = 2;
} // namespace

static ge::graphStatus CheckApplyCamePart1Inputs(const gert::InferShapeContext* context, const gert::Shape& gradShape,
                                                 const gert::Shape& epsShape)
{
    if (gradShape.GetDimNum() < kMinGradRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "grad", std::to_string(gradShape.GetDimNum()),
                                     "rank >= 2");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < gradShape.GetDimNum(); ++i) {
        if (gradShape.GetDim(i) == 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "grad",
                                                  Ops::Base::ToString(gradShape).c_str(),
                                                  "all grad dimensions must be greater than 0");
            return ge::GRAPH_FAILED;
        }
    }
    const bool epsIsOneElement = epsShape.GetDimNum() == 0 ||
                                 (epsShape.GetDimNum() == 1 &&
                                  (epsShape.GetDim(0) == 1 || epsShape.GetDim(0) == UNKNOWN_DIM));
    OP_CHECK_IF(
        !epsIsOneElement,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "eps", Ops::Base::ToString(epsShape).c_str(),
                                              "eps must be a scalar or a 1-element tensor"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyCamePart1(gert::InferShapeContext* context)
{
    const gert::Shape* gradShape = context->GetInputShape(kGradIndex);
    const gert::Shape* epsShape = context->GetInputShape(kEpsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsShape);
    if (Ops::Base::IsUnknownRank(*gradShape)) {
        Ops::Base::SetUnknownRank(*context->GetOutputShape(kSumGradRIndex));
        Ops::Base::SetUnknownRank(*context->GetOutputShape(kSumGradCIndex));
        Ops::Base::SetUnknownRank(*context->GetOutputShape(kSumGradRCIndex));
        return ge::GRAPH_SUCCESS;
    }
    if (CheckApplyCamePart1Inputs(context, *gradShape, *epsShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* sumGradR = context->GetOutputShape(kSumGradRIndex);
    gert::Shape* sumGradC = context->GetOutputShape(kSumGradCIndex);
    gert::Shape* sumGradRC = context->GetOutputShape(kSumGradRCIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumGradR);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumGradC);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumGradRC);

    const size_t gradRank = gradShape->GetDimNum();
    sumGradR->SetDimNum(gradRank - 1);
    sumGradC->SetDimNum(gradRank - 1);
    sumGradRC->SetDimNum(gradRank - 2);
    for (size_t i = 0; i + 2 < gradRank; ++i) {
        sumGradR->SetDim(i, gradShape->GetDim(i));
        sumGradC->SetDim(i, gradShape->GetDim(i));
        sumGradRC->SetDim(i, gradShape->GetDim(i));
    }
    sumGradR->SetDim(gradRank - 2, gradShape->GetDim(gradRank - 2));
    sumGradC->SetDim(gradRank - 2, gradShape->GetDim(gradRank - 1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyCamePart1).InferShape(InferShape4ApplyCamePart1);
} // namespace ops
