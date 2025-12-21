/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _TEST_MODULATE_TILING_DEF_H_
#define _TEST_MODULATE_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

#pragma pack(1)
struct ModulateTilingData {
    int64_t inputB = 0;
    int64_t inputL = 0;
    int64_t inputD = 0;
    int64_t ubLength = 0;
    int64_t frontNum = 0;
    int64_t frontLength = 0;
    int64_t tailNum = 0;
    int64_t tailLength = 0;
    int64_t useDTiling = 0;
    int64_t parameterStatus = 0;
};
#pragma pack()

inline void InitModulateTilingData(uint8_t* tiling, ModulateTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ModulateTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                              \
    ModulateTilingData tiling_data;                                           \
    InitModulateTilingData(tiling_arg, &tiling_data)

#endif