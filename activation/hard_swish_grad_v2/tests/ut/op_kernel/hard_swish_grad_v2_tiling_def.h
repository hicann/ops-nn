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
 * \file hard_swish_grad_v2_tiling_def.h
 * \brief hard_swish_grad_v2 kernel-UT tiling data definition
 */

#ifndef _HARD_SWISH_GRAD_V2_TILING_DEF_H_
#define _HARD_SWISH_GRAD_V2_TILING_DEF_H_

#include <cstdint>

#ifndef __aicore__
#define __aicore__
#endif

struct HardSwishGradV2TilingData {
    int64_t elementNum = 0;
    uint32_t needCoreNum = 0;
    uint64_t ubSize = 0;
    int64_t elementNumEachCore = 0;
};

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)              \
    __ubuf__ tilingStruct* tilingDataPointer = reinterpret_cast<__ubuf__ tilingStruct*>( \
        (__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                 \
    HardSwishGradV2TilingData tilingData;                                          \
    INIT_TILING_DATA(HardSwishGradV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).elementNum = tilingDataPointer->elementNum;                       \
    (tilingData).needCoreNum = tilingDataPointer->needCoreNum;                     \
    (tilingData).ubSize = tilingDataPointer->ubSize;                               \
    (tilingData).elementNumEachCore = tilingDataPointer->elementNumEachCore;

#endif
