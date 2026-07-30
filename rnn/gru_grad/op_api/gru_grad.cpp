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
 * \file gru_grad.cpp
 * \brief 单层单向 GRU 反向 Level0 算子实现，匹配 GruGrad OpDef 和 kernel
 */
#include "gru_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GruGrad);

const std::array<const aclTensor*, GRU_GRAD_OUT_NUM> GruGrad(
    const aclTensor* input, const aclTensor* weightInput, const aclTensor* weightHidden, const aclTensor* initHOptional,
    const aclTensor* outputH, const aclTensor* resetGate, const aclTensor* updateGate, const aclTensor* newGate,
    const aclTensor* hn, const aclTensor* dy, const aclTensor* dh, const aclTensor* seqLengthOptional,
    const char* direction, bool hasBias, bool batchFirst, aclOpExecutor* executor)
{
    L0_DFX(GruGrad, input, weightInput, weightHidden, initHOptional, outputH, resetGate, updateGate, newGate, hn, dy,
           dh, seqLengthOptional, direction, hasBias, batchFirst);

    auto xDtype = input->GetDataType();
    auto xShape = input->GetViewShape();
    auto weightInputShape = weightInput->GetViewShape();
    auto weightHiddenShape = weightHidden->GetViewShape();
    auto initHShape = initHOptional->GetViewShape();

    const aclTensor* dx = executor->AllocTensor(xShape, xDtype, op::Format::FORMAT_ND);
    const aclTensor* dhPrev = executor->AllocTensor(initHShape, xDtype, op::Format::FORMAT_ND);
    const aclTensor* dweightInput = executor->AllocTensor(weightInputShape, xDtype, op::Format::FORMAT_ND);
    const aclTensor* dweightHidden = executor->AllocTensor(weightHiddenShape, xDtype, op::Format::FORMAT_ND);

    int64_t hiddenSize = initHShape[initHShape.GetDimNum() - 1];
    op::Shape dbBiasShape({3 * hiddenSize});
    const aclTensor* dbInput = executor->AllocTensor(dbBiasShape, xDtype, op::Format::FORMAT_ND);
    const aclTensor* dbHidden = executor->AllocTensor(dbBiasShape, xDtype, op::Format::FORMAT_ND);

    const aclTensor* seqInput = seqLengthOptional;
    if (seqInput == nullptr) {
        seqInput = executor->AllocTensor(xDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GruGrad,
                                           OP_INPUT(input, weightInput, weightHidden, initHOptional, outputH, resetGate,
                                                    updateGate, newGate, hn, dy, dh, seqInput),
                                           OP_OUTPUT(dx, dhPrev, dweightInput, dweightHidden, dbInput, dbHidden),
                                           OP_ATTR(direction, hasBias, batchFirst));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GruGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dx, dhPrev, dweightInput, dweightHidden, dbInput, dbHidden};
}

} // namespace l0op
