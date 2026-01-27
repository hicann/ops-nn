/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "leaky_relu.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(LeakyRelu);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST  = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST  = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型和数据类型判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICPU算子kernel
static const aclTensor *LeakyReluAICpu(const aclTensor *self, const float negativeSlope,
                                       const aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(LeakyReluAICpu, self, negativeSlope, out);
  static internal::AicpuTaskSpace space("LeakyRelu", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(LeakyRelu, OP_ATTR_NAMES({"alpha"}),
                                        OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(negativeSlope));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LeakyReluAICpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

// AICORE算子kernel
static const aclTensor *LeakyReluAICore(const aclTensor *self, const float negativeSlope,
                                        const aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(LeakyReluAICore, self, negativeSlope, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LeakyRelu, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(negativeSlope));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LeakyReluAICore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

const aclTensor *LeakyRelu(const aclTensor *self, const float negativeSlope, aclOpExecutor *executor) {
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());

  if (IsAiCoreSupport(self)) {
    return LeakyReluAICore(self, negativeSlope, out, executor);
  }
  return LeakyReluAICpu(self, negativeSlope, out, executor);
}
} // namespace l0op