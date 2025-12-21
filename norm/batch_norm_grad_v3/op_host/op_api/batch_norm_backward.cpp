/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "batch_norm_backward.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BNTrainingUpdateGrad);

const std::array<aclTensor*, 2> BNTrainingUpdateGrad(
    const aclTensor* gradOut, const aclTensor* x, const aclTensor* saveMean, const aclTensor* saveInvstd, float eps,
    aclOpExecutor* executor)
{
    L0_DFX(BNTrainingUpdateGrad, gradOut, x, saveMean, saveInvstd, eps);
    auto gradWeight = executor->AllocTensor(saveMean->GetViewShape(), DataType::DT_FLOAT, saveMean->GetViewFormat());
    auto gradBias = executor->AllocTensor(saveMean->GetViewShape(), DataType::DT_FLOAT, saveMean->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BNTrainingUpdateGrad, OP_INPUT(gradOut, x, saveMean, saveInvstd), OP_OUTPUT(gradWeight, gradBias),
        OP_ATTR(eps));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BNTrainingUpdateGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {gradWeight, gradBias};
}

OP_TYPE_REGISTER(BN3DTrainingUpdateGrad);

const std::array<aclTensor*, 2> BN3DTrainingUpdateGrad(
    const aclTensor* gradOut, const aclTensor* x, const aclTensor* saveMean, const aclTensor* saveInvstd, float eps,
    aclOpExecutor* executor)
{
    L0_DFX(BN3DTrainingUpdateGrad, gradOut, x, saveMean, saveInvstd, eps);
    auto gradWeight = executor->AllocTensor(
        saveMean->GetStorageShape(), saveMean->GetOriginalShape(), DataType::DT_FLOAT, saveMean->GetStorageFormat(),
        saveMean->GetOriginalFormat());
    auto gradBias = executor->AllocTensor(
        saveMean->GetStorageShape(), saveMean->GetOriginalShape(), DataType::DT_FLOAT, saveMean->GetStorageFormat(),
        saveMean->GetOriginalFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BN3DTrainingUpdateGrad, OP_INPUT(gradOut, x, saveMean, saveInvstd), OP_OUTPUT(gradWeight, gradBias),
        OP_ATTR(eps));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BN3DTrainingUpdateGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {gradWeight, gradBias};
}

OP_TYPE_REGISTER(BNTrainingReduceGrad);

const aclTensor* BNTrainingReduceGrad(
    const aclTensor* gradOut, const aclTensor* x, const aclTensor* gradWeight, const aclTensor* gradBias,
    const aclTensor* weight, const aclTensor* saveMean, const aclTensor* saveInvstd, float eps, aclOpExecutor* executor)
{
    L0_DFX(BNTrainingReduceGrad, gradOut, x, gradWeight, gradBias, weight, saveMean, saveInvstd, eps);
    auto gradInput = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BNTrainingReduceGrad, OP_INPUT(gradOut, x, gradWeight, gradBias, weight, saveMean, saveInvstd),
        OP_OUTPUT(gradInput), OP_ATTR(eps));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BNTrainingReduceGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return gradInput;
}

OP_TYPE_REGISTER(BN3DTrainingReduceGrad);

const aclTensor* BN3DTrainingReduceGrad(
    const aclTensor* gradOut, const aclTensor* x, const aclTensor* gradWeight, const aclTensor* gradBias,
    const aclTensor* weight, const aclTensor* saveMean, const aclTensor* saveInvstd, float eps, aclOpExecutor* executor)
{
    L0_DFX(BN3DTrainingReduceGrad, gradOut, x, gradWeight, gradBias, weight, saveMean, saveInvstd, eps);
    auto gradInput = executor->AllocTensor(
        x->GetStorageShape(), x->GetOriginalShape(), x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BN3DTrainingReduceGrad, OP_INPUT(gradOut, x, gradWeight, gradBias, weight, saveMean, saveInvstd),
        OP_OUTPUT(gradInput), OP_ATTR(eps));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BN3DTrainingReduceGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return gradInput;
}

OP_TYPE_REGISTER(BNInferGrad);

const aclTensor* BNInferGrad(
    const aclTensor* gradOut, const aclTensor* weight, const aclTensor* runningVar, float eps, aclOpExecutor* executor)
{
    L0_DFX(BNInferGrad, gradOut, weight, runningVar, eps);
    auto gradInput =
        executor->AllocTensor(gradOut->GetStorageShape(), gradOut->GetDataType(), gradOut->GetStorageFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BNInferGrad, OP_INPUT(gradOut, weight, runningVar), OP_OUTPUT(gradInput), OP_ATTR(eps));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BNInferGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return gradInput;
}

OP_TYPE_REGISTER(BatchNormGradV3);

const std::array<aclTensor*, BN_GRAD_V3_OUTPUT_NUM> BatchNormGradV3(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
    aclOpExecutor* executor)
{
    L0_DFX(BatchNormGradV3, gradOut, input, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps);
    auto gradInput = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), input->GetViewFormat());
    auto gradWeight = executor->AllocTensor(saveMean->GetViewShape(), weight->GetDataType(), weight->GetViewFormat());
    auto gradBias = executor->AllocTensor(saveMean->GetViewShape(), weight->GetDataType(), weight->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchNormGradV3, OP_INPUT(gradOut, input, weight, runningMean, runningVar, saveMean, saveInvstd),
        OP_OUTPUT(gradInput, gradWeight, gradBias), OP_ATTR(training, eps));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BatchNormGradV3 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }

    return {gradInput, gradWeight, gradBias};
}

} // namespace l0op
