/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "elu.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Elu);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return CheckType(self->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor* EluAiCore(
    const aclTensor* self, float alpha, float scale, float inputScale, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(EluAiCore, self, alpha, scale, inputScale, out);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Elu算子加入任务队列
    // Elu是算子的OpType，self是算子的输入，alpha是算子的属性, out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Elu, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(alpha, scale, inputScale));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EluAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AICPU算子kernel
static inline const aclTensor* EluAiCpu(
    const aclTensor* self, float alpha, float scale, float inputScale, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(EluAiCpu, self, alpha, scale, inputScale, out);

    static internal::AicpuTaskSpace space("Elu");

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Elu算子加入任务队列
    // Elu是算子的OpType，self是算子的输入，out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Elu, OP_ATTR_NAMES({"alpha", "scale", "input_scale"}), OP_INPUT(self), OP_OUTPUT(out),
        OP_ATTR(alpha, scale, inputScale));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EluAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* Elu(const aclTensor* self, float alpha, float scale, float inputScale, aclOpExecutor* executor)
{
    // 根据输入shape申请输出tensor
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    if (IsAiCoreSupport(self)) {
        return EluAiCore(self, alpha, scale, inputScale, out, executor);
    } else {
        return EluAiCpu(self, alpha, scale, inputScale, out, executor);
    }
}
} // namespace l0op