/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gelu_grad_v2.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_util.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(GeluGradV2);

static const std::initializer_list<DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline const std::initializer_list<DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
    }
}

static bool IsAiCoreSupport(const aclTensor* gradOutput, const aclTensor* self)
{
    auto checkGradOutputType = CheckType(gradOutput->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
    auto checkSelfType = CheckType(self->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
    return checkGradOutputType && checkSelfType;
}

// AiCore逻辑
static void GeluGradV2AiCore(
    const aclTensor* gradOutput, const aclTensor* self, const char* approximate, aclTensor* gradInput,
    aclOpExecutor* executor)
{
    L0_DFX(GeluGradV2AiCore, gradOutput, self, approximate, gradInput);

    auto retAicore =
        ADD_TO_LAUNCHER_LIST_AICORE(GeluGradV2, OP_INPUT(gradOutput, self), OP_OUTPUT(gradInput), OP_ATTR(approximate));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GeluGradV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
}

const aclTensor* GeluGradV2(
    const aclTensor* gradOutput, const aclTensor* self, const char* approximate, aclOpExecutor* executor)
{
    L0_DFX(GeluGradV2, gradOutput, self, approximate);
    // 判断走AiCore还是AiCPU，目前无AiCPU实现，默认走AiCore
    if (!IsAiCoreSupport(gradOutput, self)) {
        return nullptr;
    }

    Shape broadcastShape;
    if (!BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.",
            op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return nullptr;
    }

    bool isMixDataType =
        (self->GetDataType() == DataType::DT_FLOAT && gradOutput->GetDataType() == DataType::DT_FLOAT16) ||
        (self->GetDataType() == DataType::DT_FLOAT && gradOutput->GetDataType() == DataType::DT_BF16) ||
        (self->GetDataType() == DataType::DT_FLOAT16 && gradOutput->GetDataType() == DataType::DT_FLOAT) ||
        (self->GetDataType() == DataType::DT_FLOAT16 && gradOutput->GetDataType() == DataType::DT_BF16) ||
        (self->GetDataType() == DataType::DT_BF16 && gradOutput->GetDataType() == DataType::DT_FLOAT) ||
        (self->GetDataType() == DataType::DT_BF16 && gradOutput->GetDataType() == DataType::DT_FLOAT16);

    auto gradInput = isMixDataType ? executor->AllocTensor(broadcastShape, DataType::DT_FLOAT) :
                                     executor->AllocTensor(broadcastShape, self->GetDataType());
    if (gradInput == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradIntput tensor failed.");
        return nullptr;
    }

    GeluGradV2AiCore(gradOutput, self, approximate, gradInput, executor);
    return gradInput;
}
} // namespace l0op
