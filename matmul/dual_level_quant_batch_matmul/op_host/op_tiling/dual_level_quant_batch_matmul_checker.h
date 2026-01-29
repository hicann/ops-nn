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
 * \file dual_level_quant_batch_matmul_checker.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_CHECKER_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_CHECKER_H
#include <cstdint>
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#ifndef __CCE_AICORE__
#include "tiling/platform/platform_ascendc.h"
#endif

namespace optiling {
enum class QuantType
{
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
};

inline constexpr const char* QuantTypeToString(QuantType quantType)
{
    switch (quantType) {
        default:
        case QuantType::NONE:
            return "NONE";
        case QuantType::PER_TENSOR:
            return "PER_TENSOR";
        case QuantType::PER_CHANNEL:
            return "PER_CHANNEL";
        case QuantType::PER_GROUP:
            return "PER_GROUP";
        case QuantType::MX:
            return "MX";
    }
}

enum class KernelTemplateType
{
    DEFAULT_CUBE_BOUND = 0,
};

enum class WeightFormat
{
    ND = 0,
    FRACTAL_NZ = 1,
};

// input index
constexpr static uint32_t X1_INDEX = 0;
constexpr static uint32_t X2_INDEX = 1;
constexpr static uint32_t X1_LEVEL0_SCALE_INDEX = 2;
constexpr static uint32_t X1_LEVEL1_SCALE_INDEX = 3;
constexpr static uint32_t X2_LEVEL0_SCALE_INDEX = 4;
constexpr static uint32_t X2_LEVEL1_SCALE_INDEX = 5;
constexpr static uint32_t BIAS_INDEX = 6;
constexpr static uint32_t OUTPUT_Y_INDEX = 0;

// attr index
constexpr static uint32_t ATTR_DTYPE_INDEX = 0;
constexpr static uint32_t ATTR_TRANSPOSE_X1_INDEX = 1;
constexpr static uint32_t ATTR_TRANSPOSE_X2_INDEX = 2;
constexpr static uint32_t ATTR_LEVEL0_GROUP_SIZE_INDEX = 3;
constexpr static uint32_t ATTR_LEVEL1_GROUP_SIZE_INDEX = 4;

struct DualLevelQuantBatchMatmulInfo {
    bool transA = false;
    bool transB = true;
    bool hasBias = false;
    uint64_t level0GroupSize = 512L;
    uint64_t level1GroupSize = 32L;
    uint64_t mSize = 0L;
    uint64_t kSize = 0L;
    uint64_t nSize = 0L;
    ge::DataType x1Dtype = ge::DT_FLOAT4_E2M1;
    ge::DataType x2Dtype = ge::DT_FLOAT4_E2M1;
    ge::DataType biasDtype = ge::DT_FLOAT;
    ge::DataType x1Level0ScaleDtype = ge::DT_FLOAT;
    ge::DataType x1Level1ScaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType x2Level0ScaleDtype = ge::DT_FLOAT;
    ge::DataType x2Level1ScaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType yDtype = ge::DT_FLOAT16;
    uint64_t libApiWorkSpaceSize = 0UL;
    QuantType level1QuantType = QuantType::MX;
    QuantType level0QuantType = QuantType::PER_GROUP;
    const char* opName = nullptr;
    ge::Format x2Format = ge::FORMAT_FRACTAL_NZ;
};

namespace checker {

ge::graphStatus CheckContext(gert::TilingContext* context, const char* opName, uint64_t tilingDataSize);

bool CheckAttrs(
    [[maybe_unused]] gert::TilingContext* context, [[maybe_unused]] NpuArch npuArch,
    const DualLevelQuantBatchMatmulInfo& inputParams);

bool CheckDtypes(
    [[maybe_unused]] gert::TilingContext* context, [[maybe_unused]] NpuArch npuArch,
    const DualLevelQuantBatchMatmulInfo& inputParams);

bool CheckInputs(
    [[maybe_unused]] gert::TilingContext* context, [[maybe_unused]] NpuArch npuArch,
    const DualLevelQuantBatchMatmulInfo& inputParams);

} // namespace checker
} // namespace optiling
#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_CHECKER_H