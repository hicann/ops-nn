/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_update_scatter_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_H
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_X bfloat16_t
#define DTYPE_VAR int4b_t

#define __CCE_UT_TEST__

#pragma pack(1)

struct DynamicQuantUpdateScatterV2TilingData {
    uint32_t coreNum;
    uint32_t rowLen;
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t multiRowNumHeadCore;
    uint32_t multiRowNumTailCore;
    uint32_t batchSize;
    uint32_t dstSeqLen;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                             \
    DynamicQuantUpdateScatterV2TilingData tilingData;                                          \
    INIT_TILING_DATA(DynamicQuantUpdateScatterV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).coreNum = tilingDataPointer->coreNum;                                         \
    (tilingData).rowLen = tilingDataPointer->rowLen;                                           \
    (tilingData).headCoreNum = tilingDataPointer->headCoreNum;                                 \
    (tilingData).rowPerHeadCore = tilingDataPointer->rowPerHeadCore;                           \
    (tilingData).rowPerTailCore = tilingDataPointer->rowPerTailCore;                           \
    (tilingData).multiRowNumHeadCore = tilingDataPointer->multiRowNumHeadCore;                 \
    (tilingData).multiRowNumTailCore = tilingDataPointer->multiRowNumTailCore;                 \
    (tilingData).dstSeqLen = tilingDataPointer->dstSeqLen;                                     \
    (tilingData).batchSize = tilingDataPointer->batchSize;
#endif