/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_sgd.cpp
 * \brief
 */
#include "fused_sgd.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FusedSgd);

std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*> FusedSgd(
    const aclTensorList* paramsRef,
    const aclTensorList* gradsRef,
    const aclTensorList* momentumBufferListOptionalRef,
    const aclTensor* gradScaleOptional,
    float weightDecay,
    float momentum,
    float lr,
    float dampening,
    bool nesterov,
    bool maximize,
    bool isFirstStep,
    aclOpExecutor *executor)
{
    L0_DFX(FusedSgd, paramsRef, gradsRef, momentumBufferListOptionalRef, gradScaleOptional,
           weightDecay, momentum, lr, dampening, nesterov, maximize, isFirstStep);

    const aclTensorList* momentumBufferListOptionalRefOut = nullptr;
    if (momentumBufferListOptionalRef == nullptr) {
        const op::Shape momentumBufferListOptionalRefOutShape = {1};
        const aclTensor* tmpTensor = executor->AllocTensor(momentumBufferListOptionalRefOutShape, (*paramsRef)[0]->GetDataType(), (*paramsRef)[0]->GetStorageFormat());
        op::FVector<const aclTensor*> tensorListA;
        tensorListA.emplace_back(tmpTensor);
        momentumBufferListOptionalRefOut = executor->AllocTensorList(tensorListA.data(), tensorListA.size());
        if(momentumBufferListOptionalRefOut == nullptr) {
            return std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*>(
                  nullptr, nullptr, nullptr);
        }
    } else {
        momentumBufferListOptionalRefOut = momentumBufferListOptionalRef;
    }

    const aclTensorList* momentumBufferListOptionalRefInput = momentumBufferListOptionalRef;
    if (momentumBufferListOptionalRef == nullptr) {
        const op::Shape momentumBufferListOptionalRefInputShape = {0};
        const aclTensor* tmpTensor = executor->AllocTensor(momentumBufferListOptionalRefInputShape, (*paramsRef)[0]->GetDataType(), (*paramsRef)[0]->GetStorageFormat());
        op::FVector<const aclTensor*> tensorListB;
        tensorListB.emplace_back(tmpTensor);
        momentumBufferListOptionalRefInput = executor->AllocTensorList(tensorListB.data(), tensorListB.size());
        if(momentumBufferListOptionalRefInput == nullptr) {
            return std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*>(
                  nullptr, nullptr, nullptr);
        }
    }

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(FusedSgd,
        OP_INPUT(paramsRef, gradsRef, momentumBufferListOptionalRefInput, gradScaleOptional),
        OP_OUTPUT(paramsRef, gradsRef, momentumBufferListOptionalRefOut),
        OP_ATTR(weightDecay, momentum, lr, dampening, nesterov, maximize, isFirstStep));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedSgd ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*>(nullptr, nullptr, nullptr);
    }
    if (momentumBufferListOptionalRef == nullptr) {
        return std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*>(
            paramsRef, gradsRef, nullptr);
    } else {
        return std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*>(
            paramsRef, gradsRef, momentumBufferListOptionalRefOut);
    }
}

}  // namespace l0op
