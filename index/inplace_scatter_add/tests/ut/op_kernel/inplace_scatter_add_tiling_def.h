/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _TEST_INPLACE_SCATTER_ADD_TILING_H_
#define _TEST_INPLACE_SCATTER_ADD_TILING_H_

#include <cstdint>
#include <cstring>
#include <securec.h>
#include <kernel_tiling/kernel_tiling.h>

#pragma pack(1)
struct InplaceScatterAddTilingData {
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t NAligned = 0;
    uint32_t frontCoreNum = 0;
    uint32_t tailCoreNum = 0;
    uint32_t frontCoreIndicesNum = 0;
    uint32_t tailCoreIndicesNum = 0;
    uint32_t ubSize = 0;
};
#pragma pack()

inline void InitInplaceScatterAddTilingData(uint8_t* tiling, InplaceScatterAddTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(InplaceScatterAddTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                       \
    InplaceScatterAddTilingData tiling_data;                                           \
    InitInplaceScatterAddTilingData(tiling_arg, &tiling_data)

#endif