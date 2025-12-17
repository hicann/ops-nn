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
 * \file quant_matmul_reduce_sum.cpp
 * \brief
 */
#include "quant_matmul_reduce_sum.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantMatmulReduceSum);

const aclTensor* QuantMatmulReduceSum(
    const aclTensor* x1, const aclTensor* x2, const aclIntArray* dims, const aclTensor* bias,
    const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* yScale,
    const aclTensor* x1Offset, const aclTensor* x2Offset, const aclTensor* yOffset, const aclTensor* x2Table,
    int64_t dtype, bool transposeX1, bool transposeX2, uint64_t groupSize, bool keepDims,
    aclOpExecutor* executor)
{
    auto dimsTensor = executor->ConvertToTensor(dims, op::ToOpDataType(ACL_INT64));
    DataType outType = DataType::DT_BF16;
    Format format = Format::FORMAT_ND;
    auto output = executor->AllocTensor(outType, format, format);

    L0_DFX(QuantMatmulReduceSum, x1, x2, dims, bias, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, x2Table,
           dtype, transposeX1, transposeX2, groupSize, keepDims);
    OP_CHECK(
        output != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "output AllocTensor failed."),
        return nullptr);

    // 对接原型，补充compute_type为默认值-1
    int32_t compute_type = -1;
    auto ret = INFER_SHAPE(QuantMatmulReduceSum,
                            OP_INPUT(x1, x2, dimsTensor, bias, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, x2Table),
                            OP_OUTPUT(output),
                            OP_ATTR(dtype, compute_type, transposeX1, transposeX2, groupSize, keepDims));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
                            QuantMatmulReduceSum,
                            OP_INPUT(x1, x2, dimsTensor, bias, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, x2Table),
                            OP_OUTPUT(output),
                            OP_ATTR(dtype, compute_type, transposeX1, transposeX2, groupSize, keepDims));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return output;
}
} // namespace l0op
