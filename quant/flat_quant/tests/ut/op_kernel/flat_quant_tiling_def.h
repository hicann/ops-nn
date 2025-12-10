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
 * \file flat_quant_tiling.h
 * \brief
 */

#ifndef FLAT_QUANT_TILING_DEFH
#define FLAT_QUANT_TILING_DEFH

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct FlatQuantTilingData {
    uint8_t dataType;

    int64_t K;
    int64_t M;
    int64_t N;
    float clipRatio;
    TCubeTiling matmulTilingL;
    TCubeTiling matmulTilingR;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, FlatQuantTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (auto i = 0; i < sizeof(FlatQuantTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, FlatQuantTilingData* constData)
{
    memcpy(constData, tiling, sizeof(FlatQuantTilingData));
}
#endif

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    FlatQuantTilingData tilingData;            \
    InitTilingData(tilingArg, &tilingData)

#endif