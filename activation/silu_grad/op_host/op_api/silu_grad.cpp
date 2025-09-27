/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "silu_grad.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(SiluGrad);

static const aclTensor *SiluGradAiCore(const aclTensor *gradOutput, const aclTensor *self,
    aclTensor *gradInput, aclOpExecutor *executor) {
    L0_DFX(SiluGradAiCore, gradOutput, self, gradInput);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SiluGrad, OP_INPUT(gradOutput, self), OP_OUTPUT(gradInput));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SiluGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return gradInput;
}

const aclTensor *SiluGrad(const aclTensor *gradOutput, const aclTensor *self, aclOpExecutor *executor) {
    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, self, broadcastShape, return nullptr);
    OP_LOGD("gradOutput shape is: %s, self shape is: %s.", op::ToString(gradOutput->GetViewShape()).GetString(),
    op::ToString(self->GetViewShape()).GetString());
    OP_LOGD("broadcastShape shape is: %s.", op::ToString(broadcastShape).GetString());
    op::DataType gradInputDtype;
    if (gradOutput->GetDataType() != self->GetDataType()) {
        gradInputDtype = op::DataType::DT_FLOAT;
    } else {
        gradInputDtype = gradOutput->GetDataType();
    }
    auto gradInput = executor->AllocTensor(broadcastShape, gradInputDtype);
    return SiluGradAiCore(gradOutput, self, gradInput, executor);
}
}  // namespace l0op
