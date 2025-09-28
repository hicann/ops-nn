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
 * \file smooth_l1_loss_grad_v2.cpp
 * \brief
 */

#include "smooth_l1_loss_grad_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SmoothL1LossGradV2);

// SmoothL1LossGradV2 暂无AICPU分支
const aclTensor *SmoothL1LossGradV2(const aclTensor *self, const aclTensor *target, const aclTensor *gradOut,
                                    const std::string &reduction, float beta, const op::Shape outputShape,
                                    aclOpExecutor *executor)
{
    auto gradInput = executor->AllocTensor(outputShape, self->GetDataType(), op::Format::FORMAT_ND);
    L0_DFX(SmoothL1LossGradV2, self, target, gradOut, reduction, beta, gradInput);

    // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCore SmoothL1LossGradV2算子加入任务队列
    // SmoothL1LossGradV2是算子的OpType，self, target, gradOut是算子的输入，reduction 和 beta 是算子的属性，
    // gradInput是算子的输出
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(SmoothL1LossGradV2,
                                                 OP_INPUT(self, target, gradOut),
                                                 OP_OUTPUT(gradInput),
                                                 OP_ATTR(beta, reduction.c_str()));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "SmoothL1LossGradV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return gradInput;
}
} // namespace l0op
