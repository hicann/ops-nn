/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file batch_matmul_v3_tiling_strategy.h
 * \brief
 */
#ifndef __OP_HOST_BATCH_MATMUL_V3_TILING_STRATEGY_H__
#define __OP_HOST_BATCH_MATMUL_V3_TILING_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
namespace strategy {
constexpr int32_t ITER_BATCH_BASICAPI = 0;
constexpr int32_t ITER_BATCH = 1;
constexpr int32_t AL1_FULL_LOAD = 2;
constexpr int32_t BL1_FULL_LOAD_BASIC = 3;
constexpr int32_t BL1_FULL_LOAD = 4;
constexpr int32_t ASW_BASIC = 5;
constexpr int32_t BASE = 999;

const static std::map<platform_ascendc::SocVersion, std::vector<int32_t>> BatchMatMulV3PrioritiesMap = {
    {platform_ascendc::SocVersion::ASCEND910_95,
     {strategy::ITER_BATCH_BASICAPI, strategy::ITER_BATCH, strategy::AL1_FULL_LOAD, strategy::BL1_FULL_LOAD_BASIC,
      strategy::BL1_FULL_LOAD, strategy::ASW_BASIC, strategy::BASE}},
};

inline std::vector<int32_t> GetBatchMatMulV3Priorities(platform_ascendc::SocVersion socVersion)
{
    std::vector<int32_t> priorities = {};
    if (BatchMatMulV3PrioritiesMap.find(socVersion) != BatchMatMulV3PrioritiesMap.end()) {
        priorities = BatchMatMulV3PrioritiesMap.at(socVersion);
    }
    return priorities;
};
}
}
}

#endif // __OP_HOST_BATCH_MATMUL_V3_STRATEGY_H__
