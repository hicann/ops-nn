/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DEEP_NORM_GRAD_TILING_H_
#define DEEP_NORM_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct DeepNormGradTilingData {
    uint32_t useCoreNum = 0;
    uint32_t nDimNum = 0;
    uint32_t dDimNum = 0;
    uint32_t nDealPerCore = 0;
    uint32_t nDealLastCore = 0;
    uint32_t mergeNCount = 0;
    uint32_t cutDTime = 0;
    uint32_t cutDPerTime = 0;
    uint32_t cutDLastTime = 0;
    uint32_t alpha = 0;
    int32_t fixedOutputFlag = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                              \
    DeepNormGradTilingData tilingData;                                          \
    INIT_TILING_DATA(DeepNormGradTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).useCoreNum = tilingDataPointer->useCoreNum;                    \
    (tilingData).nDimNum = tilingDataPointer->nDimNum;                          \
    (tilingData).dDimNum = tilingDataPointer->dDimNum;                          \
    (tilingData).nDealPerCore = tilingDataPointer->nDealPerCore;                \
    (tilingData).nDealLastCore = tilingDataPointer->nDealLastCore;              \
    (tilingData).mergeNCount = tilingDataPointer->mergeNCount;                  \
    (tilingData).cutDTime = tilingDataPointer->cutDTime;                        \
    (tilingData).cutDPerTime = tilingDataPointer->cutDPerTime;                  \
    (tilingData).cutDLastTime = tilingDataPointer->cutDLastTime;                \
    (tilingData).alpha = tilingDataPointer->alpha;                              \
    (tilingData).fixedOutputFlag = tilingDataPointer->fixedOutputFlag;
#endif
