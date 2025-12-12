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
 * \file test_expand_into_jagged_permute.h
 * \brief
 */

#ifndef _EXPAND_INTO_JAGGED_PERMUTE_TILING_H_
#define _EXPAND_INTO_JAGGED_PERMUTE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct ExpandIntoJaggedPermuteTilingDataDef {
    int64_t realCoreNum = 0;
    int64_t frontCoreNum = 0;
    int64_t blockFactor = 0;
    int64_t tailCoreBlockFactor = 0;
    int64_t lastTaskLen = 0;
    int64_t oneTaskLen = 0;
    int64_t oneTaskOffsetLen = 0;
    int64_t inputLen = 0;
    int64_t outputSize = 0;
    int64_t offsetLen = 0;
};

inline void InitExpandIntoJaggedPermuteTilingDataDef(uint8_t* tiling, ExpandIntoJaggedPermuteTilingDataDef* data)
{
    memcpy(data, tiling, sizeof(ExpandIntoJaggedPermuteTilingDataDef));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)      \
    ExpandIntoJaggedPermuteTilingDataDef tiling_data; \
    InitExpandIntoJaggedPermuteTilingDataDef(tiling_arg, &tiling_data)

#endif // _EXPAND_INTO_JAGGED_PERMUTE_TILING_H