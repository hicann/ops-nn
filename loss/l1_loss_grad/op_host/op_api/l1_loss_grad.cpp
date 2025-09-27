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
 * \file inplace_index_add.cpp
 * \brief
 */
#include "l1_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(L1LossGrad);

static const char* REDUCTION_STR[] = {"none", "mean", "sum"};
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;

// AiCore支持的L1LossGrad类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 判断走AiCore还是AiCPU
static bool IsAiCoreSupport(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction)
{
    // 检查self的dtype是否在dtype list内
    if (!CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    // 检查gradOutput和target的dtype是否与self一致
    if (target->GetDataType() != self->GetDataType() || gradOutput->GetDataType() != self->GetDataType()) {
        return false;
    }
    if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
        return false;
    }
    return true;
}

// AiCore的执行逻辑
static void L1LossGradAiCore(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction,
    aclTensor* gradInput, aclOpExecutor* executor)
{
    L0_DFX(L1LossGradAiCore, gradOutput, self, target, reduction, gradInput);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        L1LossGrad, OP_INPUT(gradOutput, self, target), OP_OUTPUT(gradInput), OP_ATTR(REDUCTION_STR[reduction]));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "L1LossGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
}

const aclTensor* L1LossGrad(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction,
    aclOpExecutor* executor)
{
    L0_DFX(L1LossGrad, gradOutput, self, target, reduction);

    // 目前LpLoss无AiCPU,仅支持AiCore
    if (!IsAiCoreSupport(gradOutput, self, target, reduction)) {
        return nullptr;
    }

    // 对gradOutput、self、target进行broadcast处理，得到gradInput的shape
    op::Shape broadcastShape1;
    op::Shape broadcastShape2;
    if (!BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "GradOutput tensor shape:%s and self tensor shape:%s can't broadcast.",
            ToString(gradOutput->GetViewShape()).GetString(), ToString(self->GetViewShape()).GetString());
        return nullptr;
    }
    if (!BroadcastInferShape(target->GetViewShape(), broadcastShape1, broadcastShape2)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "GradOutput tensor shape:%s self tensor shape:%s target tensor shape:%s can't broadcast.",
            op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString(),
            op::ToString(target->GetViewShape()).GetString());
        return nullptr;
    }
    auto gradInput = executor->AllocTensor(broadcastShape2, self->GetDataType());
    if (gradInput == nullptr) {
        return gradInput;
    }
    L1LossGradAiCore(gradOutput, self, target, reduction, gradInput, executor);
    return gradInput;
}
} // namespace l0op
