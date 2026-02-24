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
 * \file thnn_fused_lstm_cell.cpp
 * \brief
 */
#include "thnn_fused_lstm_cell.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ThnnFusedLstmCell);

const aclnnStatus ThnnFusedLstmCell(
    const aclTensor *inputGates,
    const aclTensor *hiddenGates,
    const aclTensor *cx,
    const aclTensor *inputBias,
    const aclTensor *hiddenBias,
    aclTensor *hy,
    aclTensor *cy,
    aclTensor *storage,
    aclOpExecutor *executor)
{
    L0_DFX(
        ThnnFusedLstmCell,
        inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage
    );

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ThnnFusedLstmCell,
        OP_INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
        OP_OUTPUT(hy, cy, storage)
    );

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ThnnFusedLstmCell ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

}  // namespace l0op
