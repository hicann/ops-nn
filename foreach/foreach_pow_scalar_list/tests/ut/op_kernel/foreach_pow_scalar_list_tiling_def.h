/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_pow_scalar_list_tiling_def.h
 * \brief
 */

#ifndef FOREACH_ABS_INPLACE_TILING_DEF_H
#define FOREACH_ABS_INPLACE_TILING_DEF_H

#include <cstdint>

#define __aicore__

constexpr uint16_t MAX_TENSOR_CONT = 192;
constexpr uint16_t MAX_CORE_CONT = 64;

struct ForeachCommonTilingData {
    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
};

inline __aicore__ int32_t AlignDiv32(int32_t n) {
    return ((n + 31) & ~31) / 32;
}

#define COPY_ARR(arrA, arrB, count)        \
    for (uint16_t i = 0; i < count; i++) { \
        arrA[i] = arrB[i];                 \
    }

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                                        \
    ForeachCommonTilingData tilingData;                                                 \
    INIT_TILING_DATA(ForeachCommonTilingData, tilingDataPointer, tilingPointer);        \
    (tilingData).inputsTensorUbSize = tilingDataPointer->inputsTensorUbSize;                                \
    COPY_ARR((tilingData).tensorDataCountList, tilingDataPointer->tensorDataCountList, MAX_TENSOR_CONT)   \
    COPY_ARR((tilingData).tensorStartList, tilingDataPointer->tensorStartList, MAX_CORE_CONT)             \
    COPY_ARR((tilingData).tensorEndList, tilingDataPointer->tensorEndList, MAX_CORE_CONT)                 \
    COPY_ARR((tilingData).tensorStartOffsetList, tilingDataPointer->tensorStartOffsetList, MAX_CORE_CONT) \
    COPY_ARR((tilingData).tensorEndOffsetList, tilingDataPointer->tensorEndOffsetList, MAX_CORE_CONT)
#endif  // FOREACH_ABS_INPLACE_TILING_DEF_H