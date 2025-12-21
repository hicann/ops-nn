/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quantBatchmatmul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantBatchMatmul);

const aclTensor* QuantBatchMatmul(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, const bool adjX1,
    const bool adjX2, aclOpExecutor* executor)
{
    L0_DFX(QuantBatchMatmul, x1, x2, deqScale, bias, adjX1, adjX2);
    auto quant_mm_out = executor->AllocTensor(DataType::DT_FLOAT16, Format::FORMAT_ND, Format::FORMAT_ND);
    // 当前暂时不支持input size和hiddenSize两个参数的设置
    auto ret = INFER_SHAPE(
        QuantBatchMatmul, OP_INPUT(x1, x2, deqScale, bias), OP_OUTPUT(quant_mm_out), OP_ATTR(adjX1, adjX2, -1L, 0L));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "QuantBatchMatmul InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QuantBatchMatmul, OP_INPUT(x1, x2, deqScale, bias), OP_OUTPUT(quant_mm_out), OP_ATTR(adjX1, adjX2, -1L, 0L));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "QuantBatchMatmul ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return quant_mm_out;
};
} // namespace l0op
