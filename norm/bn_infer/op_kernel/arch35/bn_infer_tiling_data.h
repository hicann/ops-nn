/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_INFER_TILING_DATA_H
#define BN_INFER_TILING_DATA_H

#include <cstdint>

struct BNInferTilingData {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t totalALen;
    int64_t totalB1Len;
    int64_t b0Outer;
    int64_t aOuter;
    int64_t b1Outer;
    int64_t tileBlockB0Len;
    int64_t tileBlockB0Tail;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockB1Len;
    int64_t tileBlockB1Tail;
    int64_t tileBlockAPaddingNum;
    float epsilon;
};

struct BNInferLastChannelTilingData {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t totalALen;
    int64_t aOuter;
    int64_t bOuter;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockAPaddingNum;
    int64_t tileBlockBLen;
    int64_t tileBlockBTail;
    float epsilon;
};

#endif // BN_INFER_TILING_DATA_H
