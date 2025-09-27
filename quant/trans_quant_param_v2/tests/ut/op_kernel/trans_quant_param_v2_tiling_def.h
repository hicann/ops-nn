/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __TRANS_QUANT_PARAM_V2_TILING_DEF_H__
#define __TRANS_QUANT_PARAM_V2_TILING_DEF_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

struct TransQuantParamV2TilingData {
    uint32_t scaleLength = 2;
    uint32_t offsetLength = 1;
    uint32_t roundMode = 0;
};

inline void InitTransQuantParamV2TilingData(uint8_t* tiling, TransQuantParamV2TilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(TransQuantParamV2TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    TransQuantParamV2TilingData tiling_data;     \
    InitTransQuantParamV2TilingData(tiling_arg, &tiling_data);
#endif // __TRANS_QUANT_PARAM_V2_TILING_DEF_H__