/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_v4.h"
#include <dlfcn.h>
#include "aclnn_quant_matmul_v3.h"
#include "aclnn_quant_matmul_weight_nz.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "quant_matmul_v3.h"
#include "matmul/common/op_host/op_api/quant_matmul_v4.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "util/math_util.h"
#include "quant_matmul_checker.h"
#include "quant_matmul_v4_common.h"

using namespace op;
using namespace quant_matmul_v4;
using Ops::Base::CeilDiv;
using Ops::NN::BoolToString;
using Ops::NN::FormatString;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;

aclnnStatus aclnnQuantMatmulV3GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                                               const aclTensor* offset, const aclTensor* bias, bool transposeX1,
                                               bool transposeX2, const aclTensor* out, uint64_t* workspaceSize,
                                               aclOpExecutor** executor)
{
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV3GetWorkspaceSize", "December 2026",
                             "aclnnQuantMatmulV5GetWorkspaceSize");
    L2_DFX_PHASE_1(aclnnQuantMatmulV3, DFX_IN(x1, x2, scale, offset, bias), DFX_OUT(out));
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto uniqueExecutor = CREATE_EXECUTOR();
    const aclTensor* tempPtr = nullptr;
    const aclTensor* tempYScalePtr = nullptr;
    const aclTensor* tempYOffsetPtr = nullptr;
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    int64_t groupSize = 0;
    auto ret = quant_matmul_v4::internal::aclnnQuantMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x1, x2, scale), std::tie(offset, tempPtr, bias, tempYScalePtr, tempYOffsetPtr, groupSize),
        std::tie(transposeX1, transposeX2), out, uniqueExecutor.get(), "aclnnQuantMatmulV3GetWorkspaceSize");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV4GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                                               const aclTensor* offset, const aclTensor* pertokenScaleOptional,
                                               const aclTensor* bias, bool transposeX1, bool transposeX2,
                                               const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV4GetWorkspaceSize", "December 2026",
                             "aclnnQuantMatmulV5GetWorkspaceSize");
    L2_DFX_PHASE_1(aclnnQuantMatmulV4, DFX_IN(x1, x2, scale, offset, pertokenScaleOptional, bias), DFX_OUT(out));
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto uniqueExecutor = CREATE_EXECUTOR();
    const aclTensor* tempYScalePtr = nullptr;
    const aclTensor* tempYOffsetPtr = nullptr;
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    int64_t groupSize = 0;
    auto ret = quant_matmul_v4::internal::aclnnQuantMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x1, x2, scale),
        std::tie(offset, pertokenScaleOptional, bias, tempYScalePtr, tempYOffsetPtr, groupSize),
        std::tie(transposeX1, transposeX2), out, uniqueExecutor.get(), "aclnnQuantMatmulV4GetWorkspaceSize");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV3", "December 2026", "aclnnQuantMatmulV5");
    L2_DFX_PHASE_2(aclnnQuantMatmulV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantMatmulV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV4", "December 2026", "aclnnQuantMatmulV5");
    L2_DFX_PHASE_2(aclnnQuantMatmulV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantMatmulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
