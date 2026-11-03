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
 * \file relu6_grad_infershape.cpp
 * \brief
 */
#include <algorithm>
#include "op_host/infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4Relu6Grad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4Relu6Grad in ops-nn");
    constexpr size_t MAX_INPUT_DIM_NUM = 8;
    const char* inputNames[] = {"gradients", "features"};
    for (size_t input = 0; input < 2; ++input) {
        const auto* shape = context->GetInputShape(input);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        OP_CHECK_IF(shape->GetDimNum() > MAX_INPUT_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), inputNames[input],
                                                 std::to_string(shape->GetDimNum()).c_str(), "0 ~ 8 dims"),
                    return ge::GRAPH_FAILED);
    }
    const auto* gradients = context->GetInputShape(0);
    const auto* features = context->GetInputShape(1);
    const size_t commonRank = std::min(gradients->GetDimNum(), features->GetDimNum());
    for (size_t offset = 1; offset <= commonRank; ++offset) {
        const int64_t gradDim = gradients->GetDim(gradients->GetDimNum() - offset);
        const int64_t featDim = features->GetDim(features->GetDimNum() - offset);
        OP_CHECK_IF(gradDim >= 0 && featDim >= 0 && gradDim != featDim && gradDim != 1 && featDim != 1,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "gradients, features",
                        Ops::Base::ToString(std::vector<const gert::Shape*>{gradients, features}).c_str(),
                        "aligned dimensions must be equal or one of them must be 1"),
                    return ge::GRAPH_FAILED);
    }
    return Ops::Base::InferShape4Broadcast(context);
}

IMPL_OP_INFERSHAPE(Relu6Grad).InferShape(InferShape4Relu6Grad);

} // namespace ops
