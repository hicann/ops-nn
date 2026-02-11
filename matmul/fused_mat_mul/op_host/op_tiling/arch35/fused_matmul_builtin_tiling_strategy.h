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
 * \file fused_matmul_builtin_tiling_strategy.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_MATMUL_BUILTIN_TILING_STRATEGY_H__
#define __OP_HOST_FUSED_MATMUL_BUILTIN_TILING_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"

namespace optiling {
namespace fused_matmul {
namespace strategy {
constexpr int32_t ITER_BATCH_BASICAPI = 0;
constexpr int32_t BASE = 999;

const static std::map<NpuArch, std::vector<int32_t>> FusedMatMulPrioritiesMap = {
    {NpuArch::DAV_3510,
     {strategy::ITER_BATCH_BASICAPI, matmul_v3_advanced::strategy::BASIC_STREAM_K,
      matmul_v3_advanced::strategy::BASIC_ASWT}},
    {NpuArch::DAV_RESV, {strategy::BASE}}, // supportMmadS8S4平台
};

inline std::vector<int32_t> GetFusedMatMulPriorities(NpuArch npuArch)
{   
    std::vector<int32_t> priorities = {};
    if (FusedMatMulPrioritiesMap.find(npuArch) != FusedMatMulPrioritiesMap.end()) {
        priorities = FusedMatMulPrioritiesMap.at(npuArch);
    }

    return priorities;
};
} // namespace strategy
} // namespace fused_matmul
} // namespace optiling

#endif // __OP_HOST_FUSED_MATMUL_BUILTIN_TILING_STRATEGY_H__
