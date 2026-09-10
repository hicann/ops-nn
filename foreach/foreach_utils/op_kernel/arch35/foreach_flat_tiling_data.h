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
 * \file foreach_flat_tiling_data.h
 * \brief Shared plain-old-data tiling schema for flat RegBase foreach kernels.
 */
#ifndef FOREACH_FLAT_TILING_DATA_H
#define FOREACH_FLAT_TILING_DATA_H

#include <cstdint>

constexpr uint16_t FOREACH_FLAT_MAX_TENSOR_COUNT = 256;
constexpr uint16_t FOREACH_FLAT_MAX_CORE_COUNT = 80;

struct ForeachFlatTilingData {
    uint32_t tileElems = 0;
    int64_t tensorDataCountList[FOREACH_FLAT_MAX_TENSOR_COUNT] = {0};
    uint16_t tensorStartList[FOREACH_FLAT_MAX_CORE_COUNT] = {0};
    uint16_t tensorEndList[FOREACH_FLAT_MAX_CORE_COUNT] = {0};
    int64_t tensorStartOffsetList[FOREACH_FLAT_MAX_CORE_COUNT] = {0};
    int64_t tensorEndOffsetList[FOREACH_FLAT_MAX_CORE_COUNT] = {0};
};

#endif // FOREACH_FLAT_TILING_DATA_H
