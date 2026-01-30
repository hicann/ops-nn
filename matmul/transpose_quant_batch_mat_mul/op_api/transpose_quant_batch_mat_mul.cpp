/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_quant_batch_mat_mul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TransposeQuantBatchMatMul);

const aclTensor* TransposeQuantBatchMatMul(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const int32_t dtype, const int32_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2,
    const aclIntArray* permY, const int32_t batchSplitFactor, aclOpExecutor* executor)
{
    L0_DFX(
        TransposeQuantBatchMatMul, x1, x2, bias, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY,
        batchSplitFactor);
    auto outType = static_cast<DataType>(dtype);
    auto out = executor->AllocTensor(outType, Format::FORMAT_ND, Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        TransposeQuantBatchMatMul, OP_INPUT(x1, x2, bias, x1Scale, x2Scale), OP_OUTPUT(out),
        OP_ATTR(dtype, groupSize, permX1, permX2, permY, batchSplitFactor));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "TransposeQuantBatchMatMul InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        TransposeQuantBatchMatMul, OP_INPUT(x1, x2, bias, x1Scale, x2Scale), OP_OUTPUT(out),
        OP_ATTR(dtype, groupSize, permX1, permX2, permY, batchSplitFactor), OP_MODE(0U));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "TransposeQuantBatchMatMul ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
};
} // namespace l0op
