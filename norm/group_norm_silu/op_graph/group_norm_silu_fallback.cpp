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
constexpr size_t SELF_INDEX = 0;
constexpr size_t GAMMA_INDEX = 1;
constexpr size_t BETA_INDEX = 2;
constexpr size_t OUT_INDEX = 0;
constexpr size_t MEAN_OUT_INDEX = 1;
constexpr size_t RSTD_OUT_INDEX = 2;
constexpr size_t FIRST_ATTR_INDEX = 0;
constexpr size_t SECOND_ATTR_INDEX = 1;
constexpr size_t THIRD_ATTR_INDEX = 2;

static graphStatus GroupNormSiluExecuteFunc(OpExecuteContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "hostApiCtx is null"), return GRAPH_FAILED);
    OP_LOGD(hostApiCtx->GetNodeName(), "Enter GroupNormSilu fallback.");

    auto self = hostApiCtx->GetInputTensor(SELF_INDEX);
    OP_CHECK_IF(self == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "self is null"), return GRAPH_FAILED);

    auto gamma = hostApiCtx->GetOptionalInputTensor(GAMMA_INDEX);
    auto beta = hostApiCtx->GetOptionalInputTensor(BETA_INDEX);

    auto out = hostApiCtx->GetOutputTensor(OUT_INDEX);
    OP_CHECK_IF(out == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "out is null"), return GRAPH_FAILED);

    auto meanOut = hostApiCtx->GetOutputTensor(MEAN_OUT_INDEX);
    OP_CHECK_IF(meanOut == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "meanOut is null"), return GRAPH_FAILED);

    auto rstdOut = hostApiCtx->GetOutputTensor(RSTD_OUT_INDEX);
    OP_CHECK_IF(rstdOut == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "rstdOut is null"), return GRAPH_FAILED);

    auto attrs = hostApiCtx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "attrs is null"), return GRAPH_FAILED);

    const int32_t* group = attrs->GetAttrPointer<int32_t>(FIRST_ATTR_INDEX);
    const float* epsValue = attrs->GetAttrPointer<float>(SECOND_ATTR_INDEX);
    double eps = *epsValue;
    bool activateSilu = *(attrs->GetAttrPointer<bool>(THIRD_ATTR_INDEX));

    // execute opapi
    auto host_api_ctx = hostApiCtx;
    auto apiRet = EXEC_OPAPI_CMD(aclnnGroupNormSiluV2, self, gamma, beta, *group, eps, activateSilu, out, meanOut,
                                 rstdOut);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE(hostApiCtx->GetNodeName(), "apiRet faild:%d", apiRet),
                return GRAPH_FAILED);

    OP_LOGD(hostApiCtx->GetNodeName(), "End GroupNormSilu fallback.");
    return GRAPH_SUCCESS;
}

IMPL_OP(GroupNormSilu).OpExecuteFunc(GroupNormSiluExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
