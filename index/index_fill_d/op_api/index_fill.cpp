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
 * \file index_fill.cpp
 * \brief
 */

#include "index_fill.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(IndexFillD);
OP_TYPE_REGISTER(IndexFill);

// AICORE
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16,  op::DataType::DT_FLOAT,
    op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> AICORE_DTYPE_910B_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16,  op::DataType::DT_FLOAT,
    op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_BF16};

// 根据dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B ||
      socVersion == SocVersion::ASCEND910_93 || Ops::NN::AclnnUtil::IsRegbase()) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_910B_SUPPORT_LIST);
  } else {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
  }
}

// AICORE算子kernel
static const aclTensor *IndexFillDAiCore(const aclTensor *self, const aclTensor *assist1, const aclTensor *assist2,
                                         int64_t dim, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(IndexFillDAiCore, self, assist1, assist2);
  // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将indexfilld算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexFillD,
                                         OP_INPUT(self, assist1, assist2),
                                         OP_OUTPUT(out),
                                         OP_ATTR(dim));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexFillDAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor *IndexFillD(const aclTensor *self, const aclTensor *assist1, const aclTensor *assist2,
                            int64_t dim, aclOpExecutor *executor) {
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  if (IsAiCoreSupport(self)) {
    return IndexFillDAiCore(self, assist1, assist2, dim, out, executor);
  } else {
    return nullptr;
  }
}

const aclTensor *IndexFill(const aclTensor *self, const aclTensor *indices, const aclTensor *value,
                            int64_t dim, aclOpExecutor *executor){
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  L0_DFX(IndexFill, self, indices, value);
  // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将indexfill算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexFill,
                                         OP_INPUT(self, indices, value),
                                         OP_OUTPUT(out),
                                         OP_ATTR(dim));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexFillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

}  // namespace l0op
