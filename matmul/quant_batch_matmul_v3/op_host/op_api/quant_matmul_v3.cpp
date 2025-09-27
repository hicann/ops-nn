/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_matmul_v3.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantBatchMatmulV3);

constexpr int64_t TYPE_FP16 = 1;
constexpr int64_t TYPE_BF16 = 27;
constexpr int64_t TYPE_INT32 = 3;
constexpr int64_t TYPE_HIF8 = 34;
constexpr int64_t TYPE_FP8_E4M3FN = 36;
constexpr int64_t TYPE_FP32 = 0;

const aclTensor* QuantBatchMatmulV3(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                                    const aclTensor* offset, const aclTensor* bias, const aclTensor* pertokenScale,
                                    int64_t dtype, bool transposeX1, bool transposeX2, int64_t groupSize,
                                    aclOpExecutor* executor) {
    L0_DFX(QuantBatchMatmulV3, x1, x2, scale, offset, bias, pertokenScale, transposeX1, transposeX2, groupSize);
    DataType outType = DataType::DT_INT8;
    if (dtype == TYPE_FP16) {
        outType = DataType::DT_FLOAT16;
    } else if (dtype == TYPE_BF16) {
        outType = DataType::DT_BF16;
    } else if (dtype == TYPE_INT32) {
        outType = DataType::DT_INT32;
    } else if (dtype == TYPE_HIF8) {
        outType = DataType::DT_HIFLOAT8;
    } else if (dtype == TYPE_FP8_E4M3FN) {
        outType = DataType::DT_FLOAT8_E4M3FN;
    } else if (dtype == TYPE_FP32) {
        outType = DataType::DT_FLOAT;
    }
    Format format = Format::FORMAT_ND;
    auto output = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(QuantBatchMatmulV3, OP_INPUT(x1, x2, scale, offset, bias, pertokenScale), OP_OUTPUT(output),
                            OP_ATTR(dtype, transposeX1, transposeX2, groupSize));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "QuantBatchMatmulV3 InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantBatchMatmulV3, OP_INPUT(x1, x2, scale, offset, bias, pertokenScale),
                                      OP_OUTPUT(output), OP_ATTR(dtype, transposeX1, transposeX2, groupSize));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "QuantBatchMatmulV3 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}
}