/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fused_matmul_silu.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedMatmulSilu);

namespace {

bool CheckAiCoreSupport(const aclTensor* x, const aclTensor* weight, const aclTensor* bias)
{
    auto arch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(arch == NpuArch::DAV_2201, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedMatmulSilu only supports Ascend910B."),
             return false);
    OP_CHECK(x->GetDataType() == DataType::DT_BF16, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be BF16."), return false);
    OP_CHECK(weight->GetDataType() == DataType::DT_BF16, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight must be BF16."),
             return false);
    OP_CHECK(bias->GetDataType() == DataType::DT_BF16, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias must be BF16."),
             return false);
    return true;
}

const aclTensor* LaunchFusedMatmulSilu(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                       const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(LaunchFusedMatmulSilu, x, weight, bias, y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedMatmulSilu, OP_INPUT(x), OP_INPUT(weight), OP_INPUT(bias),
                                           OP_OUTPUT(y));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "FusedMatmulSilu ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return y;
}

} // namespace

const aclTensor* FusedMatmulSilu(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                 aclOpExecutor* executor)
{
    OP_CHECK(CheckAiCoreSupport(x, weight, bias),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedMatmulSilu support check failed."), return nullptr);

    const auto& xShape = x->GetViewShape();
    const auto& weightShape = weight->GetViewShape();
    op::Shape yShape({xShape[0], weightShape[0]});
    const aclTensor* y = executor->AllocTensor(yShape, x->GetDataType());
    OP_CHECK(y != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed."), return nullptr);

    return LaunchFusedMatmulSilu(x, weight, bias, y, executor);
}

} // namespace l0op
