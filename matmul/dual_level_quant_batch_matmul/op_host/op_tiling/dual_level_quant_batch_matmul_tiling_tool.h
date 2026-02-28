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
 * \file dual_level_quant_batch_matmul_tiling_tool.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_TOOL_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_TOOL_H

#include "log/log.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "error_util.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling::tool {

constexpr uint64_t BASIC_BLOCK = 512UL;

template <typename T1, typename T2>
T2 CalcTailSize(T1 num1, T2 num2)
{
    if (num2 == 0) {
        return 0;
    }

    T1 mod = num1 % static_cast<T1>(num2);
    return mod != 0 ? static_cast<T2>(mod) : num2;
}

template <typename T>
T GetShapeWithDataType(T size, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return size + size;
    } else {
        return size / static_cast<T>(ge::GetSizeByDataType(dtype));
    }
}

template <typename T>
T GetSizeWithDataType(T shape, ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4) {
        return (shape + 1) >> 1;
    } else {
        return shape * static_cast<T>(ge::GetSizeByDataType(dtype));
    }
}

ge::Format GetInputStorageFormat(const gert::TilingContext* context, size_t id);

} // namespace optiling::tool
#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_TOOL_H