/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/**
 * \file threshold_v2.cpp
 * \brief ACLNN L0 API for ThresholdV2
 */
#include "threshold_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ThresholdV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "ThresholdV2 not supported on this platform: npuArch=%d. "
                     "Only DAV_3510 (Ascend950) is supported.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "ThresholdV2 supports DT_FLOAT/DT_FLOAT16/DT_BFLOAT16, got %d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    return true;
}

static const aclTensor* ThresholdV2AiCore(const aclTensor* self,
                                           float thresholdF,
                                           float valueF,
                                           const aclTensor* out,
                                           aclOpExecutor* executor)
{
    L0_DFX(ThresholdV2AiCore, self, thresholdF, valueF, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ThresholdV2,
        OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(thresholdF, valueF));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ThresholdV2AiCore failed."),
             return nullptr);
    return out;
}

const aclTensor* ThresholdV2(const aclTensor* self,
                              float thresholdF,
                              float valueF,
                              aclOpExecutor* executor)
{
    op::Shape outShape = self->GetViewShape();

    OP_CHECK(IsAiCoreSupport(self),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, self->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed."),
             return nullptr);

    return ThresholdV2AiCore(self, thresholdF, valueF, out, executor);
}

} // namespace l0op
