/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __GE_GLU_GRAD_V2_TILING_DEF_H__
#define __GE_GLU_GRAD_V2_TILING_DEF_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)

struct GeGluGradV2TilingData {
    int32_t approximate = 0;
    int32_t activateLeft = 0;
    int64_t maxProcCount = 0;
    int64_t valueN = 0;
    int64_t valueM = 0;
    int64_t needCoreNum = 0;
    int64_t loopNumPerCore = 0;
    int64_t tailCoreIndex = 0;
    int64_t tailUbLoopNum = 0;
    int64_t groupNum = 0;
};

#pragma pack()

inline void InitGeGluGradV2TilingData(uint8_t* tiling, GeGluGradV2TilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(GeGluGradV2TilingData));
}

#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    GeGluGradV2TilingData tiling_data;           \
    InitGeGluGradV2TilingData(tiling_arg, &tiling_data)

#endif