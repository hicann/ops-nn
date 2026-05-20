/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_compress_dequant.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MatMulV2CompressDequant);
// 用大小写区分是否可以区分两个不同的kernel
const aclTensor *MatMulCompressDequant(const aclTensor *x1, const aclTensor *x2, const aclTensor *compressIndex,
    const aclTensor *deqScale, const aclTensor *bias, const aclTensor *offsetW,
    const bool transposeX1, const bool transposeX2, const aclIntArray *compressInfo,
    const int offsetX, const std::string &algStr, aclOpExecutor *executor) {
    L0_DFX(MatMulCompressDequant, x1, x2, compressIndex, deqScale, bias, offsetW, transposeX1, transposeX2,
        compressInfo, offsetX, algStr);
    auto mmCompressDequantOut = executor->AllocTensor(DataType::DT_FLOAT16, Format::FORMAT_FRACTAL_NZ, Format::FORMAT_ND);
    // 是否可以复用tbe的infershape
    auto ret = INFER_SHAPE(MatMulV2CompressDequant, OP_INPUT(x1, x2, compressIndex, deqScale, bias, offsetW),
        OP_OUTPUT(mmCompressDequantOut), OP_ATTR(transposeX1, transposeX2, compressInfo, offsetX, algStr));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "MatMulCompressDequant InferShape failed.");
    // 使用新的MatmulV2CompressDequant内核
    ret = ADD_TO_LAUNCHER_LIST_AICORE(MatMulV2CompressDequant, OP_INPUT(x1, x2, compressIndex, deqScale, bias, offsetW),
                                    OP_OUTPUT(mmCompressDequantOut),
                                    OP_ATTR(transposeX1, transposeX2, compressInfo, offsetX, algStr));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                        "MatMulCompressDequant ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return mmCompressDequantOut;
};
}  // namespace l0op
