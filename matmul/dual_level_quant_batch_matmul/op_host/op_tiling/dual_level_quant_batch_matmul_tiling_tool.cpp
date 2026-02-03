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
 * \file dual_level_quant_batch_matmul_tiling_tool.cpp
 * \brief
 */
#include "dual_level_quant_batch_matmul_tiling_tool.h"

namespace optiling::tool {

constexpr int64_t B32_BITS = 32;
constexpr int64_t B16_BITS = 16;
constexpr int64_t B8_BITS = 8;
constexpr int64_t B4_BITS = 4;

ge::Format GetInputStorageFormat(const gert::TilingContext* context, size_t id)
{
    auto desc = context->GetInputDesc(id);
    OP_TILING_CHECK(
        desc == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING("dual_level_quant_batch_matmul_tiling", "get input[%zu] Desc is null!", id),
        return ge::FORMAT_NULL);
    return static_cast<ge::Format>(GetPrimaryFormat(desc->GetStorageFormat()));
}
} // namespace optiling::tool