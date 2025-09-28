/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lp_loss.cpp
 * \brief
 */

#include "lp_loss.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LpLoss);

static const char* REDUCTION[] = {"none", "mean", "sum"};
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;

// AiCore支持的LpLoss类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

// 判断走AiCore还是AiCPU
static bool IsAiCoreSupport(const aclTensor *self, const aclTensor *target, int64_t reduction) {
  if (!CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
    return false;
  }
  if (target->GetDataType() != self->GetDataType()) {
    return false;
  }
  if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
    return false;
  }
  return true;
}

// AiCore的执行逻辑
static aclTensor *LpLossAiCore(const aclTensor *self,
                         const aclTensor *target,
                         int64_t p,
                         int64_t reduction,
                         aclTensor *out,
                         aclOpExecutor *executor) {
  L0_DFX(LpLossAiCore, self, target, p, reduction, out);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LpLoss,
                                         OP_INPUT(self, target),
                                         OP_OUTPUT(out),
                                         OP_ATTR(p, REDUCTION[reduction]));
  if (ret !=  ACL_SUCCESS) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LpLossAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
      return nullptr;
  }
  return out;
}

const aclTensor *LpLoss(const aclTensor *self,
                        const aclTensor *target,
                        int64_t p,
                        int64_t reduction,
                        aclOpExecutor *executor) {
  L0_DFX(LpLoss, self, target, p, reduction);

  // 目前LpLoss无AiCPU,仅支持AiCore
  if (!IsAiCoreSupport(self, target, reduction)) {
    return nullptr;
  }

  // 对self和target的shape进行broadcast
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), target->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self tensor shape:%s and target tensor shape:%s can't broadcast.",
            ToString(self->GetViewShape()).GetString(), ToString(target->GetViewShape()).GetString());
    return nullptr;
  }

  // 1、若reduction='none'，out的shape与broadcast的结果一致
  // 2、若reduction！='none'，out为0-dimensional Tensor
  aclTensor *lpLossResult = nullptr;
  if (reduction == REDUCTION_NONE_NUM) {
    lpLossResult = executor->AllocTensor(broadcastShape, self->GetDataType());
  }
  if (reduction != REDUCTION_NONE_NUM) {
    lpLossResult = executor->AllocTensor({}, self->GetDataType());
  }
  if (lpLossResult == nullptr) {
    return lpLossResult;
  }
  LpLossAiCore(self, target, p, reduction, lpLossResult, executor);
  return lpLossResult;
}
}  // namespace l0op
