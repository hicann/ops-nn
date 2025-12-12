/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "prelu.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(PRelu);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                         DataType::DT_FLOAT16,
                                                                                         DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *PReluAiCore(const aclTensor *self, const aclTensor *weight,
                                   const aclTensor *preluOut, aclOpExecutor *executor) {
    L0_DFX(PReluAiCore, self, weight, preluOut);
    ADD_TO_LAUNCHER_LIST_AICORE(PRelu, OP_INPUT(self, weight), OP_OUTPUT(preluOut));

    return preluOut;
}

// AICPU算子kernel
[[maybe_unused]] static const aclTensor *PReluAiCpu(const aclTensor *self, const aclTensor *weight,
                                  const aclTensor *preluOut, aclOpExecutor *executor) {
    // 使用框架宏ADD_AICPU_TO_KERNEL_OBJ_LIST，将PRelu算子加入任务队列
    L0_DFX(PReluAiCpu, self, weight, preluOut);
    static internal::AicpuTaskSpace space("PRelu");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(PRelu, OP_ATTR_NAMES(), OP_INPUT(self, weight), OP_OUTPUT(preluOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return preluOut;
}

const aclTensor *PRelu(const aclTensor *self, const aclTensor *weight, aclOpExecutor *executor) {
    auto preluOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    auto ret = INFER_SHAPE(PRelu, OP_INPUT(self, weight), OP_OUTPUT(preluOut), OP_EMPTY_ARG);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    if (IsAiCoreSupport(self)) {
      return PReluAiCore(self, weight, preluOut, executor);
    }
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "prelu is both not supported in aicore and aicpu");
    return nullptr;
}
}  // namespace l0op
