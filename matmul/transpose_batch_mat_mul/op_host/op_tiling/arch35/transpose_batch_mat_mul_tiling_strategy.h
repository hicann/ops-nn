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
 * \file transpose_batch_mat_mul_tiling_strategy.h
 * \brief
 */
#ifndef __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_TILING_STRATEGY_H__
#define __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_TILING_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {
namespace strategy {
constexpr int32_t BASE = 999;

const static std::map<platform_ascendc::SocVersion, std::vector<int32_t>> TransposeBatchMatMulPrioritiesMap = {
    { platform_ascendc::SocVersion::ASCEND910_95,
    { strategy::BASE} },
};

inline std::vector<int32_t> GetTransposeBatchMatMulPriorities(platform_ascendc::SocVersion socVersion)
{
    std::vector<int32_t> priorities = {};
    if (TransposeBatchMatMulPrioritiesMap.find(socVersion) != TransposeBatchMatMulPrioritiesMap.end()) {
        priorities = TransposeBatchMatMulPrioritiesMap.at(socVersion);
    }
    return priorities;
};
}
}
}

#endif // __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_TILING_STRATEGY_H__
