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
 * \file turbo_quant_infershape.cpp
 * \brief
 */
#include <string>

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "turbo_quant_tiling.h"

using namespace ge;
namespace ops {
constexpr size_t TQ_LATENT_DIM_NUM = 2;
constexpr size_t TQ_DIM_TOKEN = 0;
constexpr size_t TQ_DIM_HEAD = 1;
constexpr int64_t TQ_UNKNOWN_DIM = -1;

static ge::graphStatus InferShapeForTurboQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForTurboQuant");
    const gert::Shape* latentShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, latentShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* scaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    if (Ops::Base::IsUnknownRank(*latentShape)) {
        OP_LOGD(context, "latent shape is UnknownRank, set y and scale shapes to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*scaleShape);
        return ge::GRAPH_SUCCESS;
    }

    if (latentShape->GetDimNum() != TQ_LATENT_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "latent", std::to_string(latentShape->GetDimNum()).c_str(),
                                     std::to_string(TQ_LATENT_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t numTokens = latentShape->GetDim(TQ_DIM_TOKEN);
    int64_t headDim = latentShape->GetDim(TQ_DIM_HEAD);
    if (numTokens < TQ_UNKNOWN_DIM) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "numTokens", std::to_string(numTokens).c_str(),
                                  "non-negative or -1");
        return ge::GRAPH_FAILED;
    }
    if (headDim != TQ_UNKNOWN_DIM && headDim != optiling::TQ_SUPPORTED_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "headDim", std::to_string(headDim).c_str(),
                                  std::to_string(optiling::TQ_SUPPORTED_HEAD_DIM).c_str());
        return ge::GRAPH_FAILED;
    }

    yShape->SetDimNum(TQ_LATENT_DIM_NUM);
    yShape->SetDim(TQ_DIM_TOKEN, numTokens);
    scaleShape->SetDimNum(1);
    scaleShape->SetDim(0, numTokens);
    if (headDim == TQ_UNKNOWN_DIM) {
        yShape->SetDim(TQ_DIM_HEAD, TQ_UNKNOWN_DIM);
    } else {
        yShape->SetDim(TQ_DIM_HEAD, headDim / optiling::TQ_ELEMENTS_PER_BYTE);
    }

    OP_LOGD(context, "End to do InferShapeForTurboQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboQuant).InferShape(InferShapeForTurboQuant);
} // namespace ops
