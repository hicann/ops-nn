/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"

namespace optiling {
namespace gemm_syrk {
namespace strategy {
constexpr int32_t SYRK_BASE = 0;

const static std::map<NpuArch, std::vector<int32_t>> GemmSyrkPrioritiesMap = {
    {NpuArch::DAV_3510, {strategy::SYRK_BASE}},
};

inline std::vector<int32_t> GetGemmSyrkPriorities(NpuArch npuArch)
{
    std::vector<int32_t> priorities = {};
    if (GemmSyrkPrioritiesMap.find(npuArch) != GemmSyrkPrioritiesMap.end()) {
        priorities = GemmSyrkPrioritiesMap.at(npuArch);
    }
    return priorities;
};
} // namespace strategy
} // namespace gemm_syrk
} // namespace optiling
