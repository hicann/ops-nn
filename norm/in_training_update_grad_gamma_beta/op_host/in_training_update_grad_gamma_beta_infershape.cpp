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
 * \file in_training_update_grad_gamma_beta_infershape.cpp
 * \brief Shape inference for INTrainingUpdateGradGammaBeta.
 */

#include <cstddef>

#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1U && shape->GetDim(0U) == ge::UNKNOWN_DIM_NUM;
}

bool IsSupportedFormat(ge::Format format)
{
    return format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC || format == ge::FORMAT_NCDHW ||
           format == ge::FORMAT_NDHWC || format == ge::FORMAT_ND;
}

bool FormatMatchesRank(ge::Format format, const gert::Shape* shape)
{
    if (shape == nullptr || IsUnknownRank(shape)) {
        return true;
    }
    const size_t rank = shape->GetDimNum();
    if (format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC) {
        return rank == 4U;
    }
    if (format == ge::FORMAT_NCDHW || format == ge::FORMAT_NDHWC) {
        return rank == 5U;
    }
    return format == ge::FORMAT_ND && (rank == 4U || rank == 5U);
}

bool ShapesCompatible(const gert::Shape* lhs, const gert::Shape* rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    if (IsUnknownRank(lhs) || IsUnknownRank(rhs)) {
        return true;
    }
    if (lhs->GetDimNum() != rhs->GetDimNum()) {
        return false;
    }
    for (size_t index = 0U; index < lhs->GetDimNum(); ++index) {
        const int64_t lhsDim = lhs->GetDim(index);
        const int64_t rhsDim = rhs->GetDim(index);
        if (lhsDim >= 0 && rhsDim >= 0 && lhsDim != rhsDim) {
            return false;
        }
    }
    return true;
}

bool IsFullyConcrete(const gert::Shape* shape)
{
    if (shape == nullptr || shape->GetDimNum() == 0U || IsUnknownRank(shape)) {
        return false;
    }
    for (size_t index = 0U; index < shape->GetDimNum(); ++index) {
        if (shape->GetDim(index) < 0) {
            return false;
        }
    }
    return true;
}

bool OutputFormatMatches(const gert::CompileTimeTensorDesc* desc, ge::Format inputFormat)
{
    if (desc == nullptr) {
        return false;
    }
    const ge::Format outputFormat = desc->GetOriginFormat();
    if (outputFormat < ge::FORMAT_NCHW || outputFormat >= ge::FORMAT_END || outputFormat == ge::FORMAT_ND) {
        return true;
    }
    return outputFormat == inputFormat;
}
} // namespace

static ge::graphStatus InferShapeForINTrainingUpdateGradGammaBeta(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* gammaShape = context->GetInputShape(0U);
    const gert::Shape* betaShape = context->GetInputShape(1U);
    gert::Shape* pdGammaShape = context->GetOutputShape(0U);
    gert::Shape* pdBetaShape = context->GetOutputShape(1U);
    const auto* gammaDesc = context->GetInputDesc(0U);
    const auto* betaDesc = context->GetInputDesc(1U);
    const auto* pdGammaDesc = context->GetOutputDesc(0U);
    const auto* pdBetaDesc = context->GetOutputDesc(1U);
    if (gammaShape == nullptr || betaShape == nullptr || pdGammaShape == nullptr || pdBetaShape == nullptr ||
        gammaDesc == nullptr || betaDesc == nullptr || pdGammaDesc == nullptr || pdBetaDesc == nullptr) {
        OP_LOGE(context, "required shape or descriptor is null");
        return ge::GRAPH_FAILED;
    }

    const ge::Format gammaFormat = gammaDesc->GetOriginFormat();
    const ge::Format betaFormat = betaDesc->GetOriginFormat();
    if (!IsSupportedFormat(gammaFormat) || !IsSupportedFormat(betaFormat) || gammaFormat != betaFormat) {
        OP_LOGE(context, "input origin formats must be identical and supported");
        return ge::GRAPH_FAILED;
    }
    if (!FormatMatchesRank(gammaFormat, gammaShape) || !FormatMatchesRank(betaFormat, betaShape)) {
        OP_LOGE(context, "input origin format does not match rank");
        return ge::GRAPH_FAILED;
    }
    if (!ShapesCompatible(gammaShape, betaShape)) {
        OP_LOGE(context, "input shapes must match");
        return ge::GRAPH_FAILED;
    }
    if (!OutputFormatMatches(pdGammaDesc, gammaFormat) || !OutputFormatMatches(pdBetaDesc, gammaFormat)) {
        OP_LOGE(context, "output origin formats must mirror the input format");
        return ge::GRAPH_FAILED;
    }

    gert::Shape expectedShape = *gammaShape;
    if (!IsUnknownRank(gammaShape)) {
        if (gammaShape->GetDimNum() == 0U) {
            OP_LOGE(context, "input rank must be 4 or 5");
            return ge::GRAPH_FAILED;
        }
        expectedShape.SetDim(0U, 1);
    }
    if (IsFullyConcrete(pdGammaShape) && !ShapesCompatible(pdGammaShape, &expectedShape)) {
        OP_LOGE(context, "declared pd_gamma shape differs from the inferred shape");
        return ge::GRAPH_FAILED;
    }
    if (IsFullyConcrete(pdBetaShape) && !ShapesCompatible(pdBetaShape, &expectedShape)) {
        OP_LOGE(context, "declared pd_beta shape differs from the inferred shape");
        return ge::GRAPH_FAILED;
    }
    *pdGammaShape = expectedShape;
    *pdBetaShape = expectedShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(INTrainingUpdateGradGammaBeta).InferShape(InferShapeForINTrainingUpdateGradGammaBeta);
} // namespace ops
