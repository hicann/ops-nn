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
 * \file test_quantized_batch_norm.h
 * \brief
 */

#ifndef _quantized_batch_norm_TILING_H_
#define _quantized_batch_norm_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define DTYPE_X int32_t
#define DTYPE_WEIGHT float

#pragma pack(1)

struct QuantizedBatchNormWelfordTilingData {
    int64_t patternR1 = 0;
    int64_t patternR0 = 0;
    int64_t patternA = 0;
    int64_t blockFactor = 0;
    int64_t tailCoreBlockFactor = 0;
    int64_t aUbFactor = 0;
    int64_t aUbLoop = 0;
    int64_t aUbTail = 0;
    int64_t tailCoreAUbLoop = 0;
    int64_t tailCoreAUbTail = 0;
    int64_t r0UbFactor = 0;
    int64_t r0UbLoop = 0;
    int64_t r0UbTail = 0;
    int64_t procNR0 = 0;
    int64_t nR0Loop = 0;
    int64_t lastLoopNR0 = 0;
    int64_t patternR0Align = 0;
    float epsilon = 0;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, QuantizedBatchNormWelfordTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(QuantizedBatchNormWelfordTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, QuantizedBatchNormWelfordTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(QuantizedBatchNormWelfordTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif