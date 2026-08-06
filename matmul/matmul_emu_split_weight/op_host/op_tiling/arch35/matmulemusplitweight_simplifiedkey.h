/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "log/log.h"
#include "error_util.h"

namespace optiling {
namespace matmul_emu_split_weight {
inline ge::graphStatus GenSimplifiedKey(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    static constexpr size_t DEST_MAX = 100;
    static constexpr size_t MAX_LEN_SIMPLIFIED_KEY = 256;
    static constexpr int32_t INPUT_X = 0;
    static constexpr int32_t INPUT_W_HIGH = 1;
    static constexpr int32_t INPUT_W_LOW = 2;
    OP_LOGI(context->GetNodeName(), "Enter genSimplifiedKey.");
    OP_TILING_CHECK(simplifiedKey == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "simplifiedKey is null"),
                    return ge::GRAPH_FAILED);

    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT_X));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT_W_HIGH));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT_W_LOW));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));

    auto xFormat = context->GetInputDesc(INPUT_X)->GetStorageFormat();
    auto wHighFormat = context->GetInputDesc(INPUT_W_HIGH)->GetStorageFormat();
    auto wLowFormat = context->GetInputDesc(INPUT_W_LOW)->GetStorageFormat();
    auto yFormat = context->GetOutputDesc(0)->GetStorageFormat();
    auto xDataType = context->GetInputDesc(INPUT_X)->GetDataType();
    auto wHighDataType = context->GetInputDesc(INPUT_W_HIGH)->GetDataType();
    auto wLowDataType = context->GetInputDesc(INPUT_W_LOW)->GetDataType();
    auto yDataType = context->GetOutputDesc(0)->GetDataType();

    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    simpleKeyTemp.append(std::to_string(xFormat))
        .append("/")
        .append(std::to_string(wHighFormat))
        .append("/")
        .append(std::to_string(wLowFormat))
        .append("/")
        .append(std::to_string(yFormat))
        .append("/")
        .append(std::to_string(xDataType))
        .append("/")
        .append(std::to_string(wHighDataType))
        .append("/")
        .append(std::to_string(wLowDataType))
        .append("/")
        .append(std::to_string(yDataType));
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
} // namespace matmul_emu_split_weight
} // namespace optiling
