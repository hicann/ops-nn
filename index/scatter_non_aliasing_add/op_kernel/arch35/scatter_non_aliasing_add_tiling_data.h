/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_NON_ALIASING_ADD_TILING_DATA_H
#define SCATTER_NON_ALIASING_ADD_TILING_DATA_H

constexpr int32_t MAX_STRIDES = 8;

struct ScatterNonAliasingAddTilingData {
    int64_t totalElements;
    int64_t perCoreElements;
    int64_t totalScatters;
    int64_t updateDataNum;
    int32_t K;
    int32_t reserved; // padding to align strides array
    int64_t strides[MAX_STRIDES];
    int64_t reservedTail; // tail padding to match opParaSize (112 bytes)
};

#endif
