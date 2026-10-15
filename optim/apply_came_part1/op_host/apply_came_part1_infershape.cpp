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

static ge::graphStatus CheckEpsShape(const gert::InferShapeContext* context)
{
    const gert::Shape* epsShape = context->GetInputShape(kEpsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsShape);
    if (Ops::Base::IsUnknownRank(*epsShape) || epsShape->GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    for (size_t i = 0; i < epsShape->GetDimNum(); ++i) {
        const int64_t dim = epsShape->GetDim(i);
        if (dim < 0) {
            continue;
        }
        if (dim != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "eps", Ops::Base::ToString(*epsShape).c_str(),
                                                  "eps must contain exactly one element");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGradShape(const gert::InferShapeContext* context, const gert::Shape& gradShape)
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
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyCamePart1(gert::InferShapeContext* context)
{
    const gert::Shape* gradShape = context->GetInputShape(kGradIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    if (CheckEpsShape(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(*gradShape)) {
        Ops::Base::SetUnknownRank(*context->GetOutputShape(kSumGradRIndex));
        Ops::Base::SetUnknownRank(*context->GetOutputShape(kSumGradCIndex));
        Ops::Base::SetUnknownRank(*context->GetOutputShape(kSumGradRCIndex));
        return ge::GRAPH_SUCCESS;
    }
    if (CheckGradShape(context, *gradShape) != ge::GRAPH_SUCCESS) {
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

static ge::graphStatus InferDataType4ApplyCamePart1(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kSumGradRIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumGradCIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumGradRCIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyCamePart1).InferShape(InferShape4ApplyCamePart1).InferDataType(InferDataType4ApplyCamePart1);
} // namespace ops
