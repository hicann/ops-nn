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
constexpr size_t QUANT_SCALES_INDEX = 3;
constexpr size_t QUANT_ZERO_POINTS_INDEX = 4;
constexpr size_t AXIS_ATTR_INDEX = 1;
constexpr size_t QUANT_AXIS_ATTR_INDEX = 2;
constexpr int64_t REDUCTION_UPDATE = 1;

static graphStatus QuantUpdateScatterHostExecuteFunc(OpExecuteContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE("aclnnfallback", "hostApiCtx is null"), return GRAPH_FAILED);
    OP_LOGD("aclnnfallback", "Enter QuantUpdateScatterHostExecuteFunc.");

    auto var = hostApiCtx->GetInputTensor(VAR_INDEX);
    OP_CHECK_IF(var == nullptr, OP_LOGE("aclnnfallback", "var is null"), return GRAPH_FAILED);

    auto indices = hostApiCtx->GetInputTensor(INDICES_INDEX);
    OP_CHECK_IF(indices == nullptr, OP_LOGE("aclnnfallback", "indices is null"), return GRAPH_FAILED);

    auto updates = hostApiCtx->GetInputTensor(UPDATES_INDEX);
    OP_CHECK_IF(updates == nullptr, OP_LOGE("aclnnfallback", "updates is null"), return GRAPH_FAILED);

    auto quantScales = hostApiCtx->GetInputTensor(QUANT_SCALES_INDEX);
    OP_CHECK_IF(quantScales == nullptr, OP_LOGE("aclnnfallback", "quant_scales is null"), return GRAPH_FAILED);

    auto quantZeroPoints = hostApiCtx->GetOptionalInputTensor(QUANT_ZERO_POINTS_INDEX);

    auto attrs = hostApiCtx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);

    const int64_t* axis = attrs->GetAttrPointer<int64_t>(AXIS_ATTR_INDEX);
    OP_CHECK_IF(axis == nullptr, OP_LOGE("aclnnfallback", "axis is null"), return GRAPH_FAILED);

    const int64_t* quantAxis = attrs->GetAttrPointer<int64_t>(QUANT_AXIS_ATTR_INDEX);
    OP_CHECK_IF(quantAxis == nullptr, OP_LOGE("aclnnfallback", "quant_axis is null"), return GRAPH_FAILED);

    auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(hostApiCtx, aclnnInplaceQuantScatter, var, indices, updates, quantScales,
                                              quantZeroPoints, *axis, *quantAxis, REDUCTION_UPDATE);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet failed:%d", apiRet), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(QuantUpdateScatter).OpExecuteFunc(QuantUpdateScatterHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
