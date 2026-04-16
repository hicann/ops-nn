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
#pragma once

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_strategy.h"

namespace optiling {
namespace fused_matmul {
namespace strategy {
// ***_INHERITED_FROM_***来标识该strategy继承于MMV3或BMMV3
constexpr int32_t ITER_BATCH_BASICAPI_INHERITED_FROM_BMMV3 = 0;
constexpr int32_t BASIC_STREAM_K_INHERITED_FROM_MMV3 = 1;
constexpr int32_t BASIC_ASWT_INHERITED_FROM_MMV3 = 2;
constexpr int32_t AL1_FULL_LOAD_BASIC_INHERITED_FROM_BMMV3 = 3;
constexpr int32_t BL1_FULL_LOAD_BASIC_INHERITED_FROM_BMMV3 = 4;
constexpr int32_t ASWT_BASIC_INHERITED_FROM_BMMV3 = 5;
constexpr int32_t BASE_INHERITED_FROM_BMMV3 = 6;

const static std::map<NpuArch, std::vector<int32_t>> FusedMatMulPrioritiesMap = {
    {NpuArch::DAV_3510,
     {strategy::ITER_BATCH_BASICAPI_INHERITED_FROM_BMMV3, strategy::BASIC_STREAM_K_INHERITED_FROM_MMV3,
      strategy::BASIC_ASWT_INHERITED_FROM_MMV3}},
    {NpuArch::DAV_RESV,
     {strategy::BASIC_ASWT_INHERITED_FROM_MMV3, strategy::ITER_BATCH_BASICAPI_INHERITED_FROM_BMMV3,
      strategy::AL1_FULL_LOAD_BASIC_INHERITED_FROM_BMMV3, strategy::BL1_FULL_LOAD_BASIC_INHERITED_FROM_BMMV3,
      strategy::ASWT_BASIC_INHERITED_FROM_BMMV3, strategy::BASE_INHERITED_FROM_BMMV3}}, // supportMmadS8S4平台
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

