/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <algorithm>
#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t QUANTMATMULV3_INPUTX1_INDEX = 0;
constexpr size_t QUANTMATMULV3_INPUTX2_INDEX = 1;
constexpr size_t QUANTMATMULV3_SCALE_INDEX = 2;
constexpr size_t QUANTMATMULV3_OFFSET_INDEX = 3;
constexpr size_t QUANTMATMULV3_BIAS_INDEX = 4;
constexpr size_t QUANTMATMULV3_PERTOKEN_SCALE_INDEX = 5;
constexpr size_t QUANTMATMULV3_OUTPUT_INDEX = 0;

static graphStatus QuantBatchMatmulV3ExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v3", "host_api_ctx is null"),
                return GRAPH_FAILED);
    auto x1 = host_api_ctx->GetInputTensor(QUANTMATMULV3_INPUTX1_INDEX);
    OP_CHECK_IF(x1 == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v3", "x1 is null"), return GRAPH_FAILED);

    auto x2 = host_api_ctx->GetInputTensor(QUANTMATMULV3_INPUTX2_INDEX);
    OP_CHECK_IF(x2 == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v3", "x2 is null"), return GRAPH_FAILED);

    auto scale = host_api_ctx->GetInputTensor(QUANTMATMULV3_SCALE_INDEX);
    OP_CHECK_IF(scale == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v3", "scale is null"), return GRAPH_FAILED);

    auto offset = host_api_ctx->GetOptionalInputTensor(QUANTMATMULV3_OFFSET_INDEX);

    auto bias = host_api_ctx->GetOptionalInputTensor(QUANTMATMULV3_BIAS_INDEX);

    auto pertokenScale = host_api_ctx->GetOptionalInputTensor(QUANTMATMULV3_PERTOKEN_SCALE_INDEX);

    auto output = host_api_ctx->GetOutputTensor(QUANTMATMULV3_OUTPUT_INDEX);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v3", "attrs is null"), return GRAPH_FAILED);
    const bool* transposeX1Ptr = attrs->GetBool(1);
    const bool* transposeX2Ptr = attrs->GetBool(2); // in QuantBatchMatmulV3 transpose attr idx is 1 and 2
    const bool transposeX1 = (transposeX1Ptr != nullptr ? *transposeX1Ptr : false);
    const bool transposeX2 = (transposeX2Ptr != nullptr ? *transposeX2Ptr : false);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback quant_batch_matmul_v3", "output is null"),
                return GRAPH_FAILED);
    graphStatus apiRet = GRAPH_SUCCESS;
    std::vector<DataType> v5InputDtypes = {DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8, DT_FLOAT4_E2M1};
    bool isA8W8GBDtype = false;
    bool isA8W8GBDim = false;
    if (pertokenScale != nullptr && scale != nullptr) {
        bool isBiasDtype = bias == nullptr || (bias != nullptr && bias->GetDataType() == DT_FLOAT);
        isA8W8GBDtype = x1->GetDataType() == DT_INT8 && x2->GetDataType() == DT_INT8 &&
                        output->GetDataType() == DT_BF16 && pertokenScale->GetDataType() == DT_FLOAT &&
                        scale->GetDataType() == DT_FLOAT && isBiasDtype;
        isA8W8GBDim = x1->GetStorageShape().GetDimNum() == pertokenScale->GetStorageShape().GetDimNum() &&
                      x2->GetStorageShape().GetDimNum() == scale->GetStorageShape().GetDimNum();
    }

    if (std::find(v5InputDtypes.cbegin(), v5InputDtypes.cend(), x1->GetDataType()) != v5InputDtypes.cend() ||
        (isA8W8GBDtype && isA8W8GBDim)) {
        const gert::Tensor* yScale = nullptr;
        const gert::Tensor* x1Offset = nullptr;
        const gert::Tensor* yOffset = nullptr;
        const int64_t* groupSizePtr = attrs->GetInt(3); // in QuantBatchMatmulV3 greoupSize attr idx is 3
        int64_t groupSize = (groupSizePtr != nullptr ? *groupSizePtr : 0);
        // execute opapi
        apiRet = EXEC_OPAPI_CMD(aclnnQuantMatmulV5, x1, x2, pertokenScale, scale, yScale, x1Offset, offset, yOffset,
                                bias, transposeX1, transposeX2, groupSize, output);
    } else {
        // execute opapi
        apiRet = EXEC_OPAPI_CMD(aclnnQuantMatmulV4, x1, x2, scale, offset, pertokenScale, bias, transposeX1,
                                transposeX2, output);
    }
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback quant_batch_matmul_v3", "api_ret faild:%d", apiRet),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(QuantBatchMatmulV3).OpExecuteFunc(QuantBatchMatmulV3ExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
