/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse4to2quant_matmul.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sparse4to2QuantMatmul);

constexpr int64_t TYPE_FP16 = 1L;
constexpr int64_t TYPE_BF16 = 27L;
constexpr int64_t TYPE_INT32 = 3L;
constexpr int64_t TYPE_FP32 = 0L;

const aclTensor* Sparse4to2QuantMatmul(
    const aclTensor* x, const aclTensor* sparseWeight, const aclTensor* index, const aclTensor* xScale,
    const aclTensor* sparseWeightScale, const aclTensor* bias, int64_t dtype, aclOpExecutor* executor)
{
    L0_DFX(Sparse4to2QuantMatmul, x, sparseWeight, index, xScale, sparseWeightScale, bias, dtype);
    if (executor == nullptr) {
        return nullptr;
    }
    DataType outType = DataType::DT_BF16;
    Format format = Format::FORMAT_ND;
    auto output = executor->AllocTensor(outType, format, format);
    OP_CHECK_NULL(output, return nullptr);
    auto ret = INFER_SHAPE(
        Sparse4to2QuantMatmul, OP_INPUT(x, sparseWeight, index, xScale, sparseWeightScale, bias), OP_OUTPUT(output),
        OP_ATTR(dtype));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "Sparse4to2QuantMatmul InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Sparse4to2QuantMatmul, OP_INPUT(x, sparseWeight, index, xScale, sparseWeightScale, bias), OP_OUTPUT(output),
        OP_ATTR(dtype));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "Sparse4to2QuantMatmul ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}
} // namespace l0op