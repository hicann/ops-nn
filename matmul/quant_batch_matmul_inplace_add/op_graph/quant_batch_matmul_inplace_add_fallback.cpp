/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_inplace_add_fallback.cpp
 * \brief
 */

#include <vector>
#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t QUANTMATMULIA_INPUTX1_INDEX = 0;
constexpr size_t QUANTMATMULIA_INPUTX2_INDEX = 1;
constexpr size_t QUANTMATMULIA_X2SCALE_INDEX = 2;
constexpr size_t QUANTMATMULIA_INDEX_INPUT_Y = 3;
constexpr size_t QUANTMATMULIA_X1SCALE_INDEX = 4;
constexpr size_t QUANTMATMULIA_TRANS_X1_INDEX = 0;
constexpr size_t QUANTMATMULIA_TRANS_X2_INDEX = 1;
constexpr size_t QUANTMATMULIA_TRANS_GROUP_SIZE_INDEX = 2;

static graphStatus QuantBatchMatmulInplaceAddExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr,
                OP_LOGE("In aclnnfallback quant_batch_matmul_inplace_add", "host_api_ctx is null"),
                return GRAPH_FAILED);
    auto x1 = host_api_ctx->GetInputTensor(QUANTMATMULIA_INPUTX1_INDEX);
    OP_CHECK_IF(x1 == nullptr, OP_LOGE("In aclnnfallback quant_batch_matmul_inplace_add", "x1 is null"),
                return GRAPH_FAILED);
    auto x2 = host_api_ctx->GetInputTensor(QUANTMATMULIA_INPUTX2_INDEX);
    OP_CHECK_IF(x2 == nullptr, OP_LOGE("In aclnnfallback quant_batch_matmul_inplace_add", "x2 is null"),
                return GRAPH_FAILED);
    auto x2Scale = host_api_ctx->GetInputTensor(QUANTMATMULIA_X2SCALE_INDEX);
    OP_CHECK_IF(x2Scale == nullptr, OP_LOGE("In aclnnfallback quant_batch_matmul_inplace_add", "x2Scale is null"),
                return GRAPH_FAILED);
    auto x1Scale = host_api_ctx->GetOptionalInputTensor(QUANTMATMULIA_X1SCALE_INDEX);
    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("In aclnnfallback quant_batch_matmul_inplace_add", "attrs is null"),
                return GRAPH_FAILED);
    const bool* transposeX1Ptr = attrs->GetBool(QUANTMATMULIA_TRANS_X1_INDEX);
    const bool* transposeX2Ptr = attrs->GetBool(QUANTMATMULIA_TRANS_X2_INDEX);
    const bool transposeX1 = (transposeX1Ptr != nullptr ? *transposeX1Ptr : false);
    const bool transposeX2 = (transposeX2Ptr != nullptr ? *transposeX2Ptr : false);
    const int64_t* groupSizePtr = attrs->GetInt(QUANTMATMULIA_TRANS_GROUP_SIZE_INDEX);
    auto yRef = host_api_ctx->GetInputTensor(QUANTMATMULIA_INDEX_INPUT_Y);
    int64_t groupSize = (groupSizePtr != nullptr ? *groupSizePtr : 0);
    // execute opapi
    auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnQuantBatchMatmulInplaceAdd, x1, x2, x1Scale, x2Scale,
                                              yRef, transposeX1, transposeX2, groupSize);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS,
                OP_LOGE("Execute aclnnfallback quant_batch_matmul_inplace_add", "api_ret failed:%d", apiRet),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(QuantBatchMatmulInplaceAdd).OpExecuteFunc(QuantBatchMatmulInplaceAddExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
