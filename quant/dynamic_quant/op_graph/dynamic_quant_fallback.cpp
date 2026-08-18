/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_SMOOTH_INDEX = 1;
constexpr size_t INPUT_GROUP_INDEX = 2;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_SCALE_INDEX = 1;

static graphStatus DynamicQuantExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("fallback_dynamic_quant", "host_api_ctx is null"),
                return GRAPH_FAILED);
    OP_LOGD(host_api_ctx->GetNodeName(), "Enter DynamicQuantExecuteFunc.");

    auto x = host_api_ctx->GetInputTensor(INPUT_X_INDEX);
    OP_CHECK_IF(x == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x is null"), return GRAPH_FAILED);

    auto y = host_api_ctx->GetOutputTensor(OUTPUT_Y_INDEX);
    OP_CHECK_IF(y == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "y is null"), return GRAPH_FAILED);

    auto scale = host_api_ctx->GetOutputTensor(OUTPUT_SCALE_INDEX);
    OP_CHECK_IF(scale == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "scale is null"), return GRAPH_FAILED);

    auto smooth_scales = host_api_ctx->GetOptionalInputTensor(INPUT_SMOOTH_INDEX);

    auto group_index = host_api_ctx->GetOptionalInputTensor(INPUT_GROUP_INDEX);

    // execute opapi
    auto api_ret = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnDynamicQuantV2, x, smooth_scales, group_index, y,
                                               scale);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "api_ret faild:%d", api_ret),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(DynamicQuant).OpExecuteFunc(DynamicQuantExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
