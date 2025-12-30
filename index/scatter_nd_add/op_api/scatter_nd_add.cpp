/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_nd_add.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ScatterNdAdd);

// AiCore支持的ScatterAdd类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910_95 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_BOOL};

inline static bool IsAiCoreSupport(const aclTensor* self) {
  // ScatterNdAdd只需要判断self
  if (op::GetCurrentPlatformInfo().GetSocVersion() >= op::SocVersion::ASCEND910_95) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910_95);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AiCore的执行逻辑
inline static const aclTensor* ScatterNdAddAiCore(const aclTensor* self, const aclTensor* indices,
                                                  const aclTensor* updates, bool use_locking, aclOpExecutor* executor) {
  L0_DFX(ScatterNdAddAiCore, self, indices, updates, use_locking);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ScatterNdAdd, OP_INPUT(self, indices, updates), OP_OUTPUT(self),
                                         OP_ATTR(use_locking));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);   
  return self;
}

// AiCPU的执行逻辑
inline static const aclTensor* ScatterNdAddAiCPU(const aclTensor* self, const aclTensor* indices,
                                                 const aclTensor* updates, bool use_locking, aclOpExecutor* executor) {
  L0_DFX(ScatterNdAddAiCPU, self, indices, updates, use_locking);

  static internal::AicpuTaskSpace space("ScatterNdAdd");
  space.SetRef(0);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ScatterNdAdd, OP_ATTR_NAMES({"use_locking"}), OP_INPUT(self, indices, updates),
                                        OP_OUTPUT(self), OP_ATTR(use_locking));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);       
  return self;
}

const aclTensor* ScatterNdAdd(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                              bool use_locking, aclOpExecutor* executor) {
  if (IsAiCoreSupport(self)) {
    return ScatterNdAddAiCore(self, indices, updates, use_locking, executor);
  } else {
    return ScatterNdAddAiCPU(self, indices, updates, use_locking, executor);
  }
}
}  // namespace l0op
