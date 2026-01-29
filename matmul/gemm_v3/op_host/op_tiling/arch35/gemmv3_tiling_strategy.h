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
 * \file gemmv3_tiling_strategy.h
 * \brief
 */
#ifndef __OP_HOST_GEMMV3_TILING_STRATEGY_H__
#define __OP_HOST_GEMMV3_TILING_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace gemmv3 {
namespace strategy {
constexpr int32_t BASE = 999;

const static std::map<platform_ascendc::SocVersion, std::vector<int32_t>> Gemmv3PrioritiesMap = {
    {platform_ascendc::SocVersion::ASCEND950, {strategy::BASE}},
};

inline std::vector<int32_t> GetGemmV3Priorities(platform_ascendc::SocVersion socVersion)
{
    std::vector<int32_t> priorities = {};
    if (Gemmv3PrioritiesMap.find(socVersion) != Gemmv3PrioritiesMap.end()) {
        priorities = Gemmv3PrioritiesMap.at(socVersion);
    }
    return priorities;
};
} // namespace strategy
} // namespace gemmv3
} // namespace optiling

#endif // __OP_HOST_GEMMV3_TILING_STRATEGY_H__
