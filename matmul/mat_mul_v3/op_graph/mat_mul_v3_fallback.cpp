/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_v3_fallback.cpp
 * \brief
 */

#include "op_graph/fallback_common_2stages_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t kMatmulInputSelf = 0;
constexpr size_t kMatmulInputMat2 = 1;
constexpr size_t kMatmulInputBias = 2;
constexpr size_t kMatmulOutput = 0;
constexpr float kBetaValue = 1.0f;
constexpr float kAlphaValue = 1.0f;

static graphStatus MmExecuteFunc(OpExecutePrepareContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto self_ge = host_api_ctx->GetInputTensor(kMatmulInputSelf);
    OP_CHECK_IF(self_ge == nullptr, OP_LOGE("aclnnfallback", "self_ge is null"), return GRAPH_FAILED);

    auto mat2_ge = host_api_ctx->GetInputTensor(kMatmulInputMat2);
    OP_CHECK_IF(mat2_ge == nullptr, OP_LOGE("aclnnfallback", "mat2_ge is null"), return GRAPH_FAILED);

    auto bias_ge = host_api_ctx->GetOptionalInputTensor(kMatmulInputBias);

    auto out_ge = host_api_ctx->GetOutputTensor(kMatmulOutput);
    OP_CHECK_IF(out_ge == nullptr, OP_LOGE("aclnnfallback", "out_ge is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    auto transSelfPtr = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_IF(transSelfPtr == nullptr, OP_LOGE("aclnnfallback", "transSelf attr is null"), return GRAPH_FAILED);
    auto transMat2Ptr = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_IF(transMat2Ptr == nullptr, OP_LOGE("aclnnfallback", "transMat2 attr is null"), return GRAPH_FAILED);
    auto transSelf = *transSelfPtr;
    auto transMat2 = *transMat2Ptr;
    auto self_acl = ConvertMmType(self_ge, transSelf);
    OP_CHECK_IF(self_acl == nullptr, OP_LOGE("aclnnfallback", "self_acl is null"), return GRAPH_FAILED);
    auto mat2_acl = ConvertMmType(mat2_ge, transMat2);
    OP_CHECK_IF(mat2_acl == nullptr, OP_LOGE("aclnnfallback", "mat2_acl is null"), return GRAPH_FAILED);
    auto bias_acl = ConvertType(bias_ge);

    auto cubeMathType = GetMathType(host_api_ctx);
    if (bias_acl != nullptr) {
        auto beta_acl = ConvertScalarType(kBetaValue);
        auto alpha_acl = ConvertScalarType(kAlphaValue);
        OP_CHECK_IF(beta_acl == nullptr, OP_LOGE("aclnnfallback", "beta_acl is null"), return GRAPH_FAILED);
        OP_CHECK_IF(alpha_acl == nullptr, OP_LOGE("aclnnfallback", "alpha_acl is null"), return GRAPH_FAILED);

        auto api_ret = CANN_OPS_OPB_ASYN_EXEC_ACLNN(host_api_ctx, aclnnAddmm, bias_acl, self_acl, mat2_acl, beta_acl,
                                                    alpha_acl, out_ge, cubeMathType);
        OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "aclnnAddmm api_ret failed:%d", api_ret),
                    return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    auto api_ret = CANN_OPS_OPB_ASYN_EXEC_ACLNN(host_api_ctx, aclnnMm, self_acl, mat2_acl, out_ge, cubeMathType);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "aclnnMm api_ret failed:%d", api_ret),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MatMul).Op2StageExecuteFuncs(MmExecuteFunc, ExecuteOpLaunch);
IMPL_OP(MatMulV2).Op2StageExecuteFuncs(MmExecuteFunc, ExecuteOpLaunch);
IMPL_OP(MatMulV3).Op2StageExecuteFuncs(MmExecuteFunc, ExecuteOpLaunch);

} // namespace fallback

#ifdef __cplusplus
}
#endif
