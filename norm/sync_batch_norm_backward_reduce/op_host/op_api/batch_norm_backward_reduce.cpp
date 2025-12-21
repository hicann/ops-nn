/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "batch_norm_backward_reduce.h"

namespace l0op {
OP_TYPE_REGISTER(SyncBatchNormBackwardReduce);

const std::tuple<aclTensor*, aclTensor*> SyncBatchNormBackwardReduce(
    const aclTensor* sumDy, const aclTensor* sumDyXmuPad, const aclTensor* mean, const aclTensor* invstd,
    aclOpExecutor* executor)
{
    L0_DFX(SyncBatchNormBackwardReduce, sumDy, sumDyXmuPad, mean, invstd);

    // 初始化输出tensor
    auto sumDyXmuOut = executor->AllocTensor(sumDy->GetViewShape(), sumDy->GetDataType(), sumDy->GetViewFormat());
    auto gradWeightOut = executor->AllocTensor(invstd->GetViewShape(), invstd->GetDataType(), invstd->GetViewFormat());

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore BatchNormBackwardReduce算子加入任务队列
    // BatchNormBackwardReduce是算子的OpType
    // sumDy, sumDyXmuPad, mean, invstd是算子的输入
    // sumDyXmuOut, gradWeightOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SyncBatchNormBackwardReduce, OP_INPUT(sumDy, sumDyXmuPad, mean, invstd), OP_OUTPUT(sumDyXmuOut, gradWeightOut));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SyncBatchNormBackwardReduceAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(sumDyXmuOut, gradWeightOut);
}
} // namespace l0op
