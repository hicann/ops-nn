/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef MATMUL_CELOSS_PART2_TILING_DEF_H
#define MATMUL_CELOSS_PART2_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

#define DTYPE_VOCAB_PARALLEL_LOGITS float

struct FusedCrossEntropyLossWithMaxSumTilingData {
    uint32_t formerbtCountLen;
    uint32_t latterbtCountLen;
    uint32_t formerbtCountTime;
    uint32_t latterbtCountTime;
    uint32_t formerCoreNum;
    uint32_t formerCoreReservedbtNum;
    uint32_t latterCoreReservedbtNum;
    uint32_t singleCalculationQuantity;
    uint32_t singleCalculationReservedQuantity;
    uint32_t elementsNumber;
    uint32_t vLen;
};


inline void InitFusedCrossEntropyLossWithMaxSumTilingData(uint8_t* tiling, FusedCrossEntropyLossWithMaxSumTilingData* data)
{
    memcpy(data, tiling, sizeof(FusedCrossEntropyLossWithMaxSumTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                               \
    FusedCrossEntropyLossWithMaxSumTilingData tiling_data;                                \
    InitFusedCrossEntropyLossWithMaxSumTilingData(tiling_arg, &tiling_data)

#endif // MATMUL_CELOSS_PART2_TILING_DEF_H