/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t X_INDEX = 0;
static const size_t INDEX_INDEX = 1;

static graphStatus GatherElementsHostExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "host_api_ctx is null"),
                return GRAPH_FAILED);
    OP_LOGD(host_api_ctx->GetNodeName(), "GatherElementsHostExecuteFunc in ");

    auto x = host_api_ctx->GetInputTensor(X_INDEX);
    OP_CHECK_IF(x == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "x is null"), return GRAPH_FAILED);

    auto index = host_api_ctx->GetInputTensor(INDEX_INDEX);
    OP_CHECK_IF(index == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "index is null"), return GRAPH_FAILED);

    auto y = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(y == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "y is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    const int64_t* dim = attrs->GetAttrPointer<int64_t>(0);

    OP_LOGD(host_api_ctx->GetNodeName(), "GatherElements fallback to aclnnGather begin, dim = %ld", *dim);

    // execute opapi
    auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnGather, x, *dim, index, y);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "apiRet failed:%d", apiRet),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(GatherElements).OpExecuteFunc(GatherElementsHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
