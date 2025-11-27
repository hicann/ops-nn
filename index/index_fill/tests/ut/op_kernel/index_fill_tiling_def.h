/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _TEST_INDEX_FILL_TILING_H_
#define _TEST_INDEX_FILL_TILING_H_

#include <cstdint>
#include <cstring>
#include <securec.h>
#include <kernel_tiling/kernel_tiling.h>

#pragma pack(1)
struct IndexFillTilingData {
    uint32_t coreNum = 0;
    uint64_t N = 0;
    uint64_t indicesNum = 0;
    uint64_t indicesProcessMode = 0;
    uint64_t frontCoreNumTaskIndices = 0;
    uint64_t tailCoreNumTaskIndices = 0;
    uint64_t frontCoreDataTaskIndices = 0; 
    uint64_t tailCoreDataTaskIndices = 0;
    uint64_t ubSize = 0;
    uint64_t P = 0;
    uint64_t Q = 0;
};
#pragma pack()

inline void InitIndexFillTilingData(uint8_t* tiling, IndexFillTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(IndexFillTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                       \
    IndexFillTilingData tiling_data;                                           \
    InitIndexFillTilingData(tiling_arg, &tiling_data)
#define DTYPE_X float
#define DTYPE_INDICES int32_t
#endif