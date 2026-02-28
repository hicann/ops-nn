/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul_simplifiedkey.h
 * \brief
 */

#ifndef __OP_HOST_TRANSPOSE_BATCH_MATMUL_SIMPILIFIEDKEY_H__
#define __OP_HOST_TRANSPOSE_BATCH_MATMUL_SIMPILIFIEDKEY_H__

#include "exe_graph/runtime/tiling_context.h"
#include "error_util.h"

namespace optiling {
namespace transpose_batch_matmul {
inline ge::graphStatus GenSimplifiedKey(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    static const size_t DEST_MAX = 100;
    static const size_t MAX_LEN_SIMPLIFIED_KEY = 256;
    static const int32_t INPUT0_INDEX = 0;
    static const int32_t INPUT1_INDEX = 1;
    static const int32_t BIAS_INDEX = 2;
    static const int32_t SCALE_INDEX = 3;
    OP_LOGI(context->GetNodeName(), "Enter genSimplifiedKey.");
    OP_TILING_CHECK(simplifiedKey == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "simplifiedKey is null"),
                    return ge::GRAPH_FAILED);

    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT0_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));

    auto input0Format = context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat();
    auto input1Format = context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat();
    auto outputFormat = context->GetOutputDesc(0)->GetStorageFormat();
    auto input0DataType = context->GetInputDesc(INPUT0_INDEX)->GetDataType();
    auto input1DataType = context->GetInputDesc(INPUT1_INDEX)->GetDataType();
    auto outputDataType = context->GetOutputDesc(0)->GetDataType();
    auto scaleDataType = ge::DT_INT64;
    auto biasDataType = input0DataType;
    if (context->GetOptionalInputDesc(BIAS_INDEX) != nullptr) {
        biasDataType = context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();
    }
    if (context->GetOptionalInputDesc(SCALE_INDEX) != nullptr) {
        scaleDataType = context->GetOptionalInputDesc(SCALE_INDEX)->GetDataType();
    }
    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    simpleKeyTemp.append(std::to_string(input0Format))
        .append("/")
        .append(std::to_string(input1Format))
        .append("/")
        .append(std::to_string(ge::FORMAT_ND))
        .append("/")
        .append(std::to_string(ge::FORMAT_ND))
        .append("/") 
        .append(std::to_string(outputFormat))
        .append("/")
        .append(std::to_string(input0DataType))
        .append("/")
        .append(std::to_string(input1DataType))
        .append("/")
        .append(std::to_string(biasDataType))
        .append("/")
        .append(std::to_string(scaleDataType))
        .append("/")
        .append(std::to_string(outputDataType));
    errno_t err = strcat_s(simplifiedKey, DEST_MAX, simpleKeyTemp.c_str());
    if (err != 0) {
        std::cerr << "Error: strcat_s failed with error code " << err << std::endl;
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "len of simplifiedKey exceeds max length."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace transpose_batch_matmul
} // namespace optiling
#endif // __OP_HOST_TRANSPOSE_BATCH_MATMUL_SIMPILIFIEDKEY_H__
