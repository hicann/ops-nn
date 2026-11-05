/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOREACH_ZERO_INPLACE_TILING_DATA_H
#define FOREACH_ZERO_INPLACE_TILING_DATA_H

constexpr int32_t FOREACH_ZERO_INPLACE_MAX_TENSOR_NUM = 256;
constexpr int32_t FOREACH_ZERO_INPLACE_MAX_CORE_NUM = 80;

struct ForeachZeroInplaceTilingData {
    int32_t tensorNum = 0;
    int32_t needCoreNum = 1;
    uint32_t ubChunkBytes = 0;
    int64_t tensorElements[FOREACH_ZERO_INPLACE_MAX_TENSOR_NUM] = {0};
    uint16_t tensorStart[FOREACH_ZERO_INPLACE_MAX_CORE_NUM] = {0};
    uint16_t tensorEnd[FOREACH_ZERO_INPLACE_MAX_CORE_NUM] = {0};
    int64_t tensorStartOffset[FOREACH_ZERO_INPLACE_MAX_CORE_NUM] = {0};
    int64_t tensorEndOffset[FOREACH_ZERO_INPLACE_MAX_CORE_NUM] = {0};
};

#endif // FOREACH_ZERO_INPLACE_TILING_DATA_H
