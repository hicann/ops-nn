/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_MX_QUANT_TILING_DEF_H_
#define SWIGLU_MX_QUANT_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/arch35/swiglu_mx_quant_tiling_data.h"
#include "../../../op_kernel/arch35/swiglu_mx_quant_tiling_key.h"
#include <climits>

#define __CCE_UT_TEST__

#pragma pack(1)

struct SwigluMxQuantTilingDataMock {
    uint32_t reverse;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    SwigluMxQuantTilingDataMock tilingData;                                              \
    INIT_TILING_DATA(SwigluMxQuantTilingDataMock, tilingDataPointer, tilingPointer);     \
    (tilingData).reverse = tilingDataPointer->reverse;

template <typename T>
inline void InitTilingDataMock(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingDataMock<tiling_struct>(tiling_arg, &tiling_data)

#define DTYPE_X half
#define DTYPE_Y fp8_e4m3fn_t

#endif
