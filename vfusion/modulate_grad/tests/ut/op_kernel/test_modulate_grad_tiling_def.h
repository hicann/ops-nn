/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _TEST_MODULATE_GRAD_TILING_DEF_H_
#define _TEST_MODULATE_GRAD_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

#pragma pack(1)
struct ModulateTilingData {
        uint32_t B;
        uint32_t L;
        uint32_t D;
        uint32_t dataType = 2;
        uint32_t block_dim;
        uint32_t dataTypeSize;

        uint32_t splitB;
        uint32_t coresPerB;
        uint32_t usedCores;
        uint32_t formerNum;
        uint32_t formerLength;
        uint32_t tailNum;
        uint32_t tailLength;
        uint32_t has_sacle;
        uint32_t has_shift;
};
#pragma pack()

inline void InitModulateGradTilingData(uint8_t* tiling, ModulateGradTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ModulateGradTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                              \
    ModulateGradTilingData tiling_data;                                           \
    InitModulateGradTilingData(tiling_arg, &tiling_data)

#endif
