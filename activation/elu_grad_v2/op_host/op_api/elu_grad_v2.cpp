/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "elu_grad_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(EluGradV2);

const aclTensor* EluGradV2(
    const aclTensor* gradOutput, float alpha, float scale, float inputScale, bool isResult,
    const aclTensor* selfOrResult, aclOpExecutor* executor)
{
    L0_DFX(EluGradV2, gradOutput, alpha, scale, inputScale, isResult, selfOrResult);

    // 根据输入shape申请输出tensor
    auto gradIntput = executor->AllocTensor(gradOutput->GetViewShape(), gradOutput->GetDataType(), Format::FORMAT_ND);
    if (gradIntput == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradIntput tensor failed.");
        return nullptr;
    }

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore EluGradV2算子加入任务队列
    // EluGradV2是算子的OpType，gradOutput, selfOrResult是算子的输入, gradIntput是算子的输出,
    // alpha、scale、inputScale、isResult是算子的属性
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        EluGradV2, OP_INPUT(gradOutput, selfOrResult), OP_OUTPUT(gradIntput),
        OP_ATTR(alpha, scale, inputScale, isResult));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "EluGradV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return gradIntput;
}
} // namespace l0op