/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dual_level_quant_batch_matmul.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DualLevelQuantBatchMatmul);

constexpr int64_t TYPE_FP16 = 1;
constexpr int64_t TYPE_BF16 = 27;

const aclTensor* DualLevelQuantBatchMatmul(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Level0Scale, const aclTensor* x1Level1Scale,
    const aclTensor* x2Level0Scale, const aclTensor* x2Level1Scale, const aclTensor* bias, int dtype, bool transposeX1,
    bool transposeX2, int level0GroupSize, int level1GroupSize, aclOpExecutor* executor)
{
    L0_DFX(
        DualLevelQuantBatchMatmul, x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level0Scale, x2Level1Scale,
        bias, dtype, transposeX1, transposeX2, level0GroupSize, level1GroupSize);
    DataType outType = x1->GetDataType();
    if (dtype != -1) {
        outType = static_cast<DataType>(dtype);
    }
    auto output = executor->AllocTensor(outType, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        DualLevelQuantBatchMatmul, OP_INPUT(x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, bias),
        OP_OUTPUT(output), OP_ATTR(transposeX1, transposeX2, level0GroupSize, level1GroupSize, dtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        DualLevelQuantBatchMatmul, OP_INPUT(x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, bias),
        OP_OUTPUT(output), OP_ATTR(transposeX1, transposeX2, level0GroupSize, level1GroupSize, dtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return output;
}
} // namespace l0op