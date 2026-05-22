/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_matmul_v4.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantBatchMatmulV4);

const aclTensor* QuantBatchMatmulV4(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                    const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* yScale,
                                    const aclTensor* x1Offset, const aclTensor* x2Offset, const aclTensor* yOffset,
                                    const aclTensor* x2Table, int64_t dtype, int32_t computeType, bool transposeX1,
                                    bool transposeX2, uint64_t groupSize, aclOpExecutor* executor) {
    L0_DFX(QuantBatchMatmulV4, x1, x2, bias, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset,
           x2Table, dtype, computeType, transposeX1, transposeX2, groupSize);
    DataType outType = static_cast<DataType>(dtype);
    Format format = Format::FORMAT_ND;
    auto output = executor->AllocTensor(outType, format, format);
    OP_CHECK(
        output != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "output AllocTensor failed."),
        return nullptr);

    auto ret = INFER_SHAPE(
        QuantBatchMatmulV4, OP_INPUT(x1, x2, bias, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, x2Table),
        OP_OUTPUT(output), OP_ATTR(dtype, computeType, transposeX1, transposeX2, groupSize));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "QuantBatchMatmulV4 InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QuantBatchMatmulV4, OP_INPUT(x1, x2, bias, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, x2Table),
        OP_OUTPUT(output), OP_ATTR(dtype, computeType, transposeX1, transposeX2, groupSize));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
            "QuantBatchMatmulV4 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}
}
