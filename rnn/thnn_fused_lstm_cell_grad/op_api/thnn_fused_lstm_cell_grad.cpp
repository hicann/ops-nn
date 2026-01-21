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
 * \file thnn_fused_lstm_cell_grad.cpp
 * \brief
 */
#include "thnn_fused_lstm_cell_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ThnnFusedLstmCellGrad);

// ThnnFusedLstmCellGrad AICORE算子kernel
const std::array<const aclTensor *, 3> ThnnFusedLstmCellGrad(
    const aclTensor *gradHy,
    const aclTensor *gradC,
    const aclTensor *cx,
    const aclTensor *cy,
    const aclTensor *storage,
    const bool hasBias,
    aclOpExecutor *executor) {
    L0_DFX(ThnnFusedLstmCellGrad, gradHy, gradC, cx, cy, storage, hasBias);
    auto storageShape = storage->GetViewShape();
    auto hiddenShape = cx->GetViewShape();

    const aclTensor *gradInGates = executor->AllocTensor(storageShape, gradHy->GetDataType(), op::Format::FORMAT_ND);
    const aclTensor *gradCPrev = executor->AllocTensor(hiddenShape, gradHy->GetDataType(), op::Format::FORMAT_ND);
    const aclTensor *gradB = nullptr;
    if (hasBias) {
        gert::Shape bShape;
        bShape.AppendDim(storageShape[1]);
        gradB = executor->AllocTensor(bShape, gradHy->GetDataType(), op::Format::FORMAT_ND);
    } else {
        gradB = executor->AllocTensor(gradHy->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ThnnFusedLstmCellGrad,
                                           OP_INPUT(gradHy, gradC, cx, cy, storage),
                                           OP_OUTPUT(gradInGates, gradCPrev, gradB),
                                           OP_ATTR(hasBias));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ThnnFusedLstmCellGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {gradInGates, gradCPrev, gradB};
}
}  // namespace l0op
