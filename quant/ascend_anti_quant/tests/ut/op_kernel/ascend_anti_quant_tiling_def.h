/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ASCEND_ANTI_QUANT_TILING_DEF_H_
#define _ASCEND_ANTI_QUANT_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#define __CCE_UT_TEST__

// Suppress kernel-side include of arch35/ascend_anti_quant_tilingdata.h to avoid struct redefinition.
#define _ASCEND_ANTI_QUANT_TILINGDATA_

#include <cstdint>

struct AscendAntiQuantBaseTilingData {
    int64_t  dim0;
    int32_t  coreNum;
    int32_t  ubFormer;
    int64_t  blockFormer;
    int64_t  blockNum;
    int64_t  ubLoopOfFormerBlock;
    int64_t  ubLoopOfTailBlock;
    int64_t  ubTailOfFormerBlock;
    int64_t  ubTailOfTailBlock;
    int64_t  elemNum;
    uint64_t scheMode;
};

struct AscendAntiQuantTilingData {
    AscendAntiQuantBaseTilingData baseTiling;
    float    scale;
    float    offset;
    int32_t  sqrtMode;
    int32_t  reserved;
};

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                                  \
    AscendAntiQuantTilingData tilingData;                                                           \
    INIT_TILING_DATA(AscendAntiQuantTilingData, tilingDataPointer, tilingPointer);                  \
    (tilingData).baseTiling.dim0 = tilingDataPointer->baseTiling.dim0;                              \
    (tilingData).baseTiling.coreNum = tilingDataPointer->baseTiling.coreNum;                        \
    (tilingData).baseTiling.ubFormer = tilingDataPointer->baseTiling.ubFormer;                      \
    (tilingData).baseTiling.blockFormer = tilingDataPointer->baseTiling.blockFormer;                \
    (tilingData).baseTiling.blockNum = tilingDataPointer->baseTiling.blockNum;                      \
    (tilingData).baseTiling.ubLoopOfFormerBlock = tilingDataPointer->baseTiling.ubLoopOfFormerBlock;\
    (tilingData).baseTiling.ubLoopOfTailBlock = tilingDataPointer->baseTiling.ubLoopOfTailBlock;    \
    (tilingData).baseTiling.ubTailOfFormerBlock = tilingDataPointer->baseTiling.ubTailOfFormerBlock;\
    (tilingData).baseTiling.ubTailOfTailBlock = tilingDataPointer->baseTiling.ubTailOfTailBlock;    \
    (tilingData).baseTiling.elemNum = tilingDataPointer->baseTiling.elemNum;                        \
    (tilingData).baseTiling.scheMode = tilingDataPointer->baseTiling.scheMode;                      \
    (tilingData).scale = tilingDataPointer->scale;                                                  \
    (tilingData).offset = tilingDataPointer->offset;                                                \
    (tilingData).sqrtMode = tilingDataPointer->sqrtMode;                                            \
    (tilingData).reserved = tilingDataPointer->reserved;

#endif
