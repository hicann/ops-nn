/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t INPUT_X1_IDX = 0;
constexpr size_t INPUT_X2_IDX = 1;
constexpr size_t INPUT_BIAS_IDX = 2;
constexpr size_t INPUT_X1_SCALE_IDX = 3;
constexpr size_t INPUT_X2_SCALE_IDX = 4;
constexpr size_t INPUT_Y_SCALE_IDX = 5;
constexpr size_t INPUT_X1_OFFSET_IDX = 6;
constexpr size_t INPUT_X2_OFFSET_IDX = 7;
constexpr size_t INPUT_Y_OFFSET_IDX = 8;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr int64_t DEFAULT_GROUP_SIZE = 32;

static graphStatus QuantBatchMatmulV4ExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v4", "host_api_ctx is null"),
                return GRAPH_FAILED);
    auto x1 = host_api_ctx->GetInputTensor(INPUT_X1_IDX);
    OP_CHECK_IF(x1 == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v4", "x1 is null"), return GRAPH_FAILED);

    auto x2 = host_api_ctx->GetInputTensor(INPUT_X2_IDX);
    OP_CHECK_IF(x2 == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v4", "x2 is null"), return GRAPH_FAILED);

    auto x1Scale = host_api_ctx->GetOptionalInputTensor(INPUT_X1_SCALE_IDX);
    auto x2Scale = host_api_ctx->GetOptionalInputTensor(INPUT_X2_SCALE_IDX);
    auto bias = host_api_ctx->GetOptionalInputTensor(INPUT_BIAS_IDX);
    auto yScale = host_api_ctx->GetOptionalInputTensor(INPUT_Y_SCALE_IDX);
    auto output = host_api_ctx->GetOutputTensor(OUTPUT_Y_IDX);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v4", "output is null"),
                return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v4", "attrs is null"), return GRAPH_FAILED);
    // in QuantBatchMatmulV4 transpose attr idx is 2 and 3
    const bool* transposeX1Ptr = attrs->GetBool(2);
    const bool* transposeX2Ptr = attrs->GetBool(3);
    const bool transposeX1 = (transposeX1Ptr != nullptr ? *transposeX1Ptr : false);
    const bool transposeX2 = (transposeX2Ptr != nullptr ? *transposeX2Ptr : false);
    // in QuantBatchMatmulV4 groupSize attr idx is 4
    const int64_t* groupSizePtr = attrs->GetInt(4);
    // group size default is 32
    int64_t groupSize = (groupSizePtr != nullptr ? *groupSizePtr : DEFAULT_GROUP_SIZE);

    graphStatus apiRet = GRAPH_SUCCESS;
    const gert::Tensor* x1Offset = nullptr;
    const gert::Tensor* x2Offset = nullptr;
    const gert::Tensor* yOffset = nullptr;
    // execute opapi
    apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnQuantMatmulV5, x1, x2, x1Scale, x2Scale, yScale, x1Offset,
                                         x2Offset, yOffset, bias, transposeX1, transposeX2, groupSize, output);

    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback quant_batch_matmul_v4", "api_ret failed:%d", apiRet),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(QuantBatchMatmulV4).OpExecuteFunc(QuantBatchMatmulV4ExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
