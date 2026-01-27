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
 * \file gelu_grad_infershape.cpp
 * \brief
 */
#include "op_util.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
static ge::graphStatus CopyShapeInput2OutputWithIdx(
    gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForGeluGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do GeluGradInferShape");
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_CHECK_IF(
        (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
         ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeName(), "Cannot get platform info!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "soc version is %s", platform_info.str_info.short_soc_version.c_str());
    if (platform_info.str_info.short_soc_version == "Ascend910_95") {
        const size_t inputCount = 3;
        std::vector<const gert::Shape*> to_broadcast_shapes(inputCount);
        for (size_t i = 0; i < inputCount; i++) {
            auto in_shape = context->GetInputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
            to_broadcast_shapes[i] = in_shape;
        }
        auto out_shape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

        OP_CHECK_IF(
            !BroadcastShape(to_broadcast_shapes, out_shape), OP_LOGE(context->GetNodeName(), "BroadcastShape failed!"),
            return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    return CopyShapeInput2OutputWithIdx(context, 1, 0);
}

IMPL_OP_INFERSHAPE(GeluGrad).InferShape(InferShapeForGeluGrad);
} // namespace ops
