/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_QUANT_V2_TILING_H_
#define RMS_NORM_QUANT_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/arch35/rms_norm_quant_v2_tiling_data.h"
#include <climits>

// #define DT_BF16 bfloat16_t
// #define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct RmsNormQuantV2TilingData {
    uint32_t reverse;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                \
    RmsNormQuantV2TilingData tilingData;                                          \
    INIT_TILING_DATA(RmsNormQuantV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).reverse = tilingDataPointer->reverse;

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#define DTYPE_X float
#define DTYPE_SCALES1 float
#define DTYPE_ZERO_POINTS1 float
#define DTYPE_ZERO_POINTS2 float
#define DTYPE_Y1 int8_t

#define COMPUTE_MODE_FULL_LOAD 0
#define COMPUTE_MODE_RECOMPUTE 1

#endif
