/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_tiling_strategy.h
 * \brief
 */
#pragma once

#include <cstdint>
#include <vector>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace strategy {
constexpr int32_t ITER_BATCH = 1L;
constexpr int32_t CUBE_ASW = 2L;
constexpr int32_t MX_BASIC_API_ASW = 3L;
constexpr int32_t CUBE_BASIC_API_ASW = 4L;
constexpr int32_t MIX_ASW = 5L;
constexpr int32_t PERBLOCK_BASIC_API_ASW = 6L;

inline const std::vector<int32_t>& GetQuantBatchMatmulV3Priorities(NpuArch npuArch)
{
    static const std::vector<int32_t> dav3510Priorities = {
        strategy::CUBE_ASW, strategy::MX_BASIC_API_ASW, strategy::CUBE_BASIC_API_ASW, strategy::MIX_ASW,
        strategy::PERBLOCK_BASIC_API_ASW};
    static const std::vector<int32_t> davResvPriorities = {strategy::ITER_BATCH, strategy::CUBE_ASW};
    static const std::vector<int32_t> emptyPriorities = {};
    switch (npuArch) {
        case NpuArch::DAV_3510:
            return dav3510Priorities;
        case NpuArch::DAV_RESV:
            return davResvPriorities;
        default:
            return emptyPriorities;
    }
}
} // namespace strategy
} // namespace optiling
