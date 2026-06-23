/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_NORM_GROUP_NORM_TILING_DATA_H_
#define EXPERIMENTAL_NORM_GROUP_NORM_TILING_DATA_H_

#include <cstdint>

struct GroupNormTilingData {
    uint32_t n;
    uint32_t c;
    uint32_t hxw;
    uint32_t numGroups;
    uint32_t channelsPerGroup;
    uint32_t elementsPerGroup;
    uint32_t groupNum;
    uint32_t groupPerCore;
    uint32_t groupTailCore;
    uint32_t tileLength;
    uint32_t hasGamma;
    uint32_t hasBeta;
    uint32_t reserved;
    float eps;
    float invElementsPerGroup;
};

#endif // EXPERIMENTAL_NORM_GROUP_NORM_TILING_DATA_H_
