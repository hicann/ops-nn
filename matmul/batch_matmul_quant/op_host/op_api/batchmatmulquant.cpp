/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batchmatmulquant.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BatchMatmulFixpipe);

const aclTensor* BatchMatmulQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* quantParam, const aclTensor* bias, const bool adjX1,
    const bool adjX2, aclOpExecutor* executor)
{
    L0_DFX(BatchMatmulQuant, x1, x2, quantParam, bias, adjX1, adjX2);
    // op need quantParam storage format with NC1HWC0
    const_cast<aclTensor*>(quantParam)->SetStorageFormat(Format::FORMAT_NC1HWC0);
    auto out = executor->AllocTensor(DataType::DT_INT8, Format::FORMAT_ND, Format::FORMAT_ND);
    auto ret =
        INFER_SHAPE(BatchMatmulFixpipe, OP_INPUT(x1, x2, quantParam, bias), OP_OUTPUT(out), OP_ATTR(adjX1, adjX2));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "BatchMatmulQuant InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchMatmulFixpipe, OP_INPUT(x1, x2, quantParam, bias), OP_OUTPUT(out), OP_ATTR(adjX1, adjX2));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "BatchMatmulQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
};
} // namespace l0op