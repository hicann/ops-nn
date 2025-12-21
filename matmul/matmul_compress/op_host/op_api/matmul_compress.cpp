/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_compress.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MatmulCompress);

const aclTensor* InferShape(const aclTensor* x1, const aclTensor* bias, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(DataType::DT_FLOAT16, Format::FORMAT_FRACTAL_NZ, Format::FORMAT_ND);
    op::Shape originalShape;
    originalShape.AppendDim(x1->GetOriginalShape().GetDim(0));
    originalShape.AppendDim(bias->GetOriginalShape().GetDim(0));
    op::Shape storageShape;
    storageShape.AppendDim((bias->GetOriginalShape().GetDim(0) + 15) / 16); // 16为NZ分形大小,15是向上对齐
    storageShape.AppendDim(x1->GetStorageShape().GetDim(1)); // 1是m的维度
    storageShape.AppendDim(16);  // 16为NZ分形大小

    out->SetViewShape(originalShape);
    out->SetOriginalShape(originalShape);
    out->SetStorageShape(storageShape);
    return out;
}


const aclTensor* MatmulCompress(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* compressIndex, bool transposeX1,
    bool transposeX2, int32_t compressK, int32_t compressN, aclOpExecutor* executor)
{
    L0_DFX(MatmulCompress, x1, x2, bias, compressIndex, transposeX1, transposeX2, compressK, compressN);
    auto out = InferShape(x1, bias, executor);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MatmulCompress,
                                      OP_INPUT(x1, x2, bias, compressIndex),
                                      OP_OUTPUT(out),
                                      OP_ATTR(transposeX1, transposeX2, compressK, compressN));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                        "MatmulCompress ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op
