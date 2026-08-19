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
constexpr size_t X_INDEX = 0;
constexpr size_t ATTEN_MASK_INDEX = 1;
constexpr size_t RELATIVE_POS_BIAS_INDEX = 2;
constexpr size_t Y_INDEX = 0;
constexpr size_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr size_t INNER_PRECISION_MODE_ATTR_INDEX = 1;

static graphStatus MaskedSoftmaxWithRelPosBiasHostExecuteFunc(OpExecuteContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE("aclnnfallback", "hostApiCtx is null"), return GRAPH_FAILED);

    auto x = hostApiCtx->GetInputTensor(X_INDEX);
    OP_CHECK_IF(x == nullptr, OP_LOGE("aclnnfallback", "x is null"), return GRAPH_FAILED);

    auto relativePosBias = hostApiCtx->GetInputTensor(RELATIVE_POS_BIAS_INDEX);
    OP_CHECK_IF(relativePosBias == nullptr, OP_LOGE("aclnnfallback", "relative_pos_bias is null"), return GRAPH_FAILED);

    auto y = hostApiCtx->GetOutputTensor(Y_INDEX);
    OP_CHECK_IF(y == nullptr, OP_LOGE("aclnnfallback", "y is null"), return GRAPH_FAILED);

    auto attenMask = hostApiCtx->GetOptionalInputTensor(ATTEN_MASK_INDEX);

    auto attrs = hostApiCtx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);

    const float* scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    OP_CHECK_IF(scaleValue == nullptr, OP_LOGE("aclnnfallback", "scale_value is null"), return GRAPH_FAILED);

    const int32_t* innerPrecisionMode = attrs->GetAttrPointer<int32_t>(INNER_PRECISION_MODE_ATTR_INDEX);
    OP_CHECK_IF(innerPrecisionMode == nullptr, OP_LOGE("aclnnfallback", "inner_precision_mode is null"),
                return GRAPH_FAILED);

    auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(hostApiCtx, aclnnMaskedSoftmaxWithRelPosBias, x, attenMask,
                                              relativePosBias, *scaleValue, *innerPrecisionMode, y);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet failed:%d", apiRet), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MaskedSoftmaxWithRelPosBias).OpExecuteFunc(MaskedSoftmaxWithRelPosBiasHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
