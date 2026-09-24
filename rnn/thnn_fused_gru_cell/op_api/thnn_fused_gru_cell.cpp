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
 * \file thnn_fused_gru_cell.cpp
 * \brief
 */

#include "thnn_fused_gru_cell.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ThnnFusedGruCell);

std::array<const aclTensor*, 2> ThnnFusedGruCell(const aclTensor* inputGates, const aclTensor* hiddenGates,
                                                 const aclTensor* hx, const aclTensor* inputBias,
                                                 const aclTensor* hiddenBias, aclOpExecutor* executor)
{
    OP_CHECK_NULL(inputGates, return {});
    OP_CHECK_NULL(hiddenGates, return {});
    OP_CHECK_NULL(hx, return {});
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The executor is nullptr.");
        return {};
    }

    // 输出 shape 从 hx 推导：hy = (B, H)、storage = (B, 5H)
    op::Shape hyShape;
    op::Shape storageShape;
    if (!ThnnFusedGruCellOutShape(hx, hyShape, storageShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ThnnFusedGruCell output shape derivation failed.");
        return {};
    }

    // 内部分配连续输出，L2 侧随后用 ViewCopy 按用户布局拷回
    const aclTensor* hy = executor->AllocTensor(hyShape, hx->GetDataType());
    OP_CHECK_NULL(hy, return {});
    const aclTensor* storage = executor->AllocTensor(storageShape, hx->GetDataType());
    OP_CHECK_NULL(storage, return {});

    L0_DFX(ThnnFusedGruCell, inputGates, hiddenGates, hx, inputBias, hiddenBias, hy, storage);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ThnnFusedGruCell, OP_INPUT(inputGates, hiddenGates, hx, inputBias, hiddenBias), OP_OUTPUT(hy, storage));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return {},
                                         "ThnnFusedGruCell ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return {hy, storage};
}

} // namespace l0op
