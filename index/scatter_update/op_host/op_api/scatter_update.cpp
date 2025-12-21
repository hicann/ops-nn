/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file scatter_update.cpp
 * \brief
 */

#include "scatter_update.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(ScatterUpdate);

// AiCore支持的ScatterUpdate类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
  op::DataType::DT_INT64, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910_95 = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_UINT32,
  op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_UINT64};

static bool IsAiCoreSupport(const aclTensor* self) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910B);
  } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910_95);
  } else {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
  }
}

// AiCore的执行逻辑
inline void ScatterUpdateAiCore(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                                bool useLocking, aclOpExecutor* executor) {
  L0_DFX(ScatterUpdateAiCore, self, indices, updates, useLocking);
  auto retAicore =
    ADD_TO_LAUNCHER_LIST_AICORE(ScatterUpdate, OP_INPUT(self, indices, updates), OP_OUTPUT(self), OP_ATTR(useLocking));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return,
                                       "ScatterUpdate add to aicore launch list failed.");
}

// AiCPU的执行逻辑
inline const aclTensor* ScatterUpdateAiCPU(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                                     bool useLocking, aclOpExecutor* executor) {
  L0_DFX(ScatterUpdateAiCPU, self, indices, updates, useLocking);

  static internal::AicpuTaskSpace space("ScatterUpdate", ge::DEPEND_IN_SHAPE, true);
  space.SetRef(0);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ScatterUpdate, OP_ATTR_NAMES({ "Tindices", "T", "use_locking" }),
                                        OP_INPUT(self, indices, updates), OP_OUTPUT(self),
                                        OP_ATTR(indices->GetDataType(), updates->GetDataType(), useLocking));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return self;
}

const aclTensor* ScatterUpdate(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                               aclOpExecutor* executor, bool useLocking) {
  if (IsAiCoreSupport(self)) {
    ScatterUpdateAiCore(self, indices, updates, useLocking, executor);
  } else {
    return ScatterUpdateAiCPU(self, indices, updates, useLocking, executor);
  }
  return self;
}
}  // namespace l0op
