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
constexpr size_t VAR_INDEX = 0;
constexpr size_t INDICES_INDEX = 1;
constexpr size_t UPDATES_INDEX = 2;

static graphStatus ScatterNdUpdateHostExecuteFunc(OpExecuteContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE("aclnnfallback", "hostApiCtx is null"), return GRAPH_FAILED);
    OP_LOGD("aclnnfallback", "Enter ScatterNdUpdateHostExecuteFunc.");

    auto var = hostApiCtx->GetInputTensor(VAR_INDEX);
    OP_CHECK_IF(var == nullptr, OP_LOGE("aclnnfallback", "var is null"), return GRAPH_FAILED);

    auto indices = hostApiCtx->GetInputTensor(INDICES_INDEX);
    OP_CHECK_IF(indices == nullptr, OP_LOGE("aclnnfallback", "indices is null"), return GRAPH_FAILED);

    auto updates = hostApiCtx->GetInputTensor(UPDATES_INDEX);
    OP_CHECK_IF(updates == nullptr, OP_LOGE("aclnnfallback", "updates is null"), return GRAPH_FAILED);

    auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(hostApiCtx, aclnnScatterNdUpdate, var, indices, updates);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet failed:%d", apiRet), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(ScatterNdUpdate).OpExecuteFunc(ScatterNdUpdateHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
