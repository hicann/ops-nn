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

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"

namespace optiling {
namespace matmul_emu_split_weight {
namespace strategy {
constexpr int32_t BASE = 999;

const static std::map<NpuArch, std::vector<int32_t>> MatmulEmuSplitWeightPrioritiesMap = {
    {NpuArch::DAV_2201, {strategy::BASE}},
};

inline std::vector<int32_t> GetMatmulEmuSplitWeightPriorities(NpuArch npuArch)
{
    std::vector<int32_t> priorities = {};
    if (MatmulEmuSplitWeightPrioritiesMap.find(npuArch) != MatmulEmuSplitWeightPrioritiesMap.end()) {
        priorities = MatmulEmuSplitWeightPrioritiesMap.at(npuArch);
    }
    return priorities;
};
} // namespace strategy
} // namespace matmul_emu_split_weight
} // namespace optiling
