/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "quant_batch_matmul_inplace_add.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantBatchMatmulInplaceAdd);

aclTensor* QuantBatchMatmulInplaceAdd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x2Scale, aclTensor* yRef, const aclTensor* x1ScaleOptional,  
    bool transposeX1, bool transposeX2, int64_t groupSize, aclOpExecutor* executor)
{
    L0_DFX(QuantBatchMatmulInplaceAdd, x1, x2, x2Scale, yRef, x1ScaleOptional, transposeX1, transposeX2, groupSize);

    auto ret = INFER_SHAPE(
        QuantBatchMatmulInplaceAdd, OP_INPUT(x1, x2, x2Scale, yRef, x1ScaleOptional), OP_OUTPUT(yRef),
        OP_ATTR(transposeX1, transposeX2, groupSize));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QuantBatchMatmulInplaceAdd, OP_INPUT(x1, x2, x2Scale, yRef, x1ScaleOptional), OP_OUTPUT(yRef),
        OP_ATTR(transposeX1, transposeX2, groupSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return yRef;
}

} // namespace l0op