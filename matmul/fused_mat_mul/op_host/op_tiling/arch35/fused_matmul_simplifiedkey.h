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
 * \file fused_matmul_simplifiedkey.h
 * \brief
 */

#ifndef __OP_HOST_FUSED_MATMUL_SIMPILIFIEDKEY_H__
#define __OP_HOST_FUSED_MATMUL_SIMPILIFIEDKEY_H__

#include "log/log.h"
#include "error_util.h"

namespace optiling {
namespace fused_matmul {
inline ge::graphStatus GenSimplifiedKey(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    static constexpr size_t DEST_MAX = 100;
    static constexpr size_t MAX_LEN_SIMPLIFIED_KEY = 256;
    static constexpr int32_t INPUT_X1 = 0;
    static constexpr int32_t INPUT_X2 = 1;
    static constexpr int32_t BIAS_INDEX = 2;
    static constexpr int32_t INPUT_X3 = 3;
    OP_LOGI(context->GetNodeName(), "Enter genSimplifiedKey.");
    OP_TILING_CHECK(
        simplifiedKey == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "simplifiedKey is null"),
        return ge::GRAPH_FAILED);

    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT_X1));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT_X2));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));

    auto x1Formate = context->GetInputDesc(INPUT_X1)->GetStorageFormat();
    auto x2Format = context->GetInputDesc(INPUT_X2)->GetStorageFormat();
    auto yFormat = context->GetOutputDesc(0)->GetStorageFormat();
    auto x1DataType = context->GetInputDesc(INPUT_X1)->GetDataType();
    auto x2DataType = context->GetInputDesc(INPUT_X2)->GetDataType();
    auto yDataType = context->GetOutputDesc(0)->GetDataType();
    auto biasDataType = x1DataType;
    auto x3Format = yFormat;
    auto x3DataType = yDataType;
    // 二进制发布json有无bias场景合并为同一个json发布，当无法获取bias信息时，当前约定使用input0的信息代替
    if (context->GetOptionalInputDesc(BIAS_INDEX) != nullptr) {
        biasDataType = context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();
    }
    if (context->GetOptionalInputDesc(INPUT_X3) != nullptr) {
        x3DataType = context->GetOptionalInputDesc(INPUT_X3)->GetDataType();
        x3Format = context->GetOptionalInputDesc(INPUT_X3)->GetStorageFormat();
    }

    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    simpleKeyTemp.append(std::to_string(x1Formate))
        .append("/")
        .append(std::to_string(x2Format))
        .append("/")
        .append(std::to_string(ge::FORMAT_ND))
        .append("/") // bias的format均为FormatND,因此约束为仅通过FORMAT_ND参与匹配
        .append(std::to_string(x3Format))
        .append("/") // bias的format均为FormatND,因此约束为仅通过FORMAT_ND参与匹配
        .append(std::to_string(yFormat))
        .append("/")
        .append(std::to_string(x1DataType))
        .append("/")
        .append(std::to_string(x2DataType))
        .append("/")
        .append(std::to_string(biasDataType))
        .append("/")
        .append(std::to_string(x3DataType))
        .append("/")
        .append(std::to_string(yDataType));
    errno_t err = strcat_s(simplifiedKey, DEST_MAX, simpleKeyTemp.c_str());
    if (err != 0) {
        std::cerr << "Error: strcat_s failed with error code " << err << std::endl;
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(
        strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "len of simplifiedKey exceeds max length."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace fused_matmul
} // namespace optiling
#endif // __OP_HOST_FUSED_MATMUL_SIMPILIFIEDKEY_H__