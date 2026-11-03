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
 * \file batch_mat_mul_v3_fallback.cpp
 * \brief
 */

#include <string>
#include <vector>
#include "log/log.h"
#include "op_graph/fallback_common_2stages_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t kBmmInputSelf = 0;
constexpr size_t kBmmInputMat2 = 1;
constexpr size_t kBmmInputBias = 2;
constexpr size_t kBmmOutput = 0;
constexpr size_t kBmmDimNum = 3;
constexpr size_t kMmDimNum = 2;
constexpr float kBetaValue = 1.0f;
constexpr float kAlphaValue = 1.0f;

static void ReportMissingBatchMatmulObject(const char* functionName, const char* objectName)
{
    const std::string reason = std::string(functionName) + ": Fallback prepare framework-provided " + objectName +
                               " is nullptr; check fallback context construction and operator registration";
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {"aclnnBatchMatMul", reason.c_str()};
    OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "%s", reason.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}

static graphStatus BatchMatmulExecuteFunc(OpExecutePrepareContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, ReportMissingBatchMatmulObject(__func__, "host_api_ctx"), return GRAPH_FAILED);

    auto self_ge = host_api_ctx->GetInputTensor(kBmmInputSelf);
    OP_CHECK_IF(self_ge == nullptr, ReportMissingBatchMatmulObject(__func__, "self"), return GRAPH_FAILED);
    auto mat2_ge = host_api_ctx->GetInputTensor(kBmmInputMat2);
    OP_CHECK_IF(mat2_ge == nullptr, ReportMissingBatchMatmulObject(__func__, "mat2"), return GRAPH_FAILED);
    auto bias_ge = host_api_ctx->GetOptionalInputTensor(kBmmInputBias);
    auto out_ge = host_api_ctx->GetOutputTensor(kBmmOutput);
    OP_CHECK_IF(out_ge == nullptr, ReportMissingBatchMatmulObject(__func__, "out"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportMissingBatchMatmulObject(__func__, "attrs"), return GRAPH_FAILED);
    auto transSelfPtr = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_IF(transSelfPtr == nullptr, ReportMissingBatchMatmulObject(__func__, "transSelf"), return GRAPH_FAILED);
    auto transMat2Ptr = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_IF(transMat2Ptr == nullptr, ReportMissingBatchMatmulObject(__func__, "transMat2"), return GRAPH_FAILED);
    auto transSelf = *transSelfPtr;
    auto transMat2 = *transMat2Ptr;
    auto self_acl = ConvertMmType(self_ge, transSelf);
    OP_CHECK_IF(self_acl == nullptr,
                OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback prepare: ConvertMmType(self) returned null"),
                return GRAPH_FAILED);
    auto mat2_acl = ConvertMmType(mat2_ge, transMat2);
    OP_CHECK_IF(mat2_acl == nullptr,
                OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback prepare: ConvertMmType(mat2) returned null"),
                return GRAPH_FAILED);
    auto bias_acl = ConvertType(bias_ge);

    auto cubeMathType = GetMathType(host_api_ctx);
    size_t selfDimNum = self_ge->GetStorageShape().GetDimNum();
    size_t mat2DimNum = mat2_ge->GetStorageShape().GetDimNum();
    if (bias_acl != nullptr) {
        auto beta_acl = ConvertScalarType(kBetaValue);
        auto alpha_acl = ConvertScalarType(kAlphaValue);
        OP_CHECK_IF(
            beta_acl == nullptr,
            OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback prepare: ConvertScalarType(beta) returned null"),
            return GRAPH_FAILED);
        OP_CHECK_IF(
            alpha_acl == nullptr,
            OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback prepare: ConvertScalarType(alpha) returned null"),
            return GRAPH_FAILED);
        auto api_ret = GRAPH_FAILED;
        if (selfDimNum == kMmDimNum && mat2DimNum == kMmDimNum) {
            api_ret = CANN_OPS_OPB_ASYN_EXEC_ACLNN(host_api_ctx, aclnnAddmm, bias_acl, self_acl, mat2_acl, beta_acl,
                                                   alpha_acl, out_ge, cubeMathType);
            OP_CHECK_IF(
                api_ret != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback execute: aclnnAddmm failed with rc=%u", api_ret),
                return GRAPH_FAILED);
        } else if (selfDimNum == kBmmDimNum && mat2DimNum == kBmmDimNum) {
            api_ret = CANN_OPS_OPB_ASYN_EXEC_ACLNN(host_api_ctx, aclnnBaddbmm, bias_acl, self_acl, mat2_acl, beta_acl,
                                                   alpha_acl, out_ge, cubeMathType);
            OP_CHECK_IF(
                api_ret != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback execute: aclnnBaddbmm failed with rc=%u", api_ret),
                return GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                "aclnnBatchMatMul", "self, mat2",
                "self=" + std::to_string(selfDimNum) + ", mat2=" + std::to_string(mat2DimNum),
                "With bias, self and mat2 must both have rank 2 or both have rank 3");
        }
        return api_ret;
    }
    if (selfDimNum == kBmmDimNum && mat2DimNum == kBmmDimNum) {
        auto api_ret = CANN_OPS_OPB_ASYN_EXEC_ACLNN(host_api_ctx, aclnnBatchMatMul, self_acl, mat2_acl, out_ge,
                                                    cubeMathType);
        OP_CHECK_IF(
            api_ret != GRAPH_SUCCESS,
            OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback execute: aclnnBatchMatMul failed with rc=%u", api_ret),
            return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    auto api_ret = CANN_OPS_OPB_ASYN_EXEC_ACLNN(host_api_ctx, aclnnMatmul, self_acl, mat2_acl, out_ge, cubeMathType);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("aclnnBatchMatMul", "Fallback execute: aclnnMatmul failed with rc=%u", api_ret),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(BatchMatMul).Op2StageExecuteFuncs(BatchMatmulExecuteFunc, ExecuteOpLaunch);
IMPL_OP(BatchMatMulV2).Op2StageExecuteFuncs(BatchMatmulExecuteFunc, ExecuteOpLaunch);

} // namespace fallback

#ifdef __cplusplus
}
#endif
