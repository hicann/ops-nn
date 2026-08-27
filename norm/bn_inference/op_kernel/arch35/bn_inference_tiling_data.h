/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_INFERENCE_TILING_DATA_H
#define BN_INFERENCE_TILING_DATA_H

#include <cstddef>
#include <cstdint>

struct BNInferenceTilingData {
    int64_t baseTilesPerCore;
    int64_t extraCoreCount;
    int64_t n;
    int64_t c;
    int64_t inner;
    int64_t tileElements;
    int64_t tileRows;
    int64_t paramTileLen;
    int64_t paramCacheLen;
    int64_t innerTileCount;
    float epsilon;
    // Explicit ABI padding: keep this device-side mirror at 88 bytes.
    uint32_t reserved;
};

constexpr size_t BN_INFERENCE_TILING_DATA_EXPECTED_BYTES = 88U;
static_assert(sizeof(BNInferenceTilingData) == BN_INFERENCE_TILING_DATA_EXPECTED_BYTES,
              "BNInference tiling ABI size changed unexpectedly");

#endif // BN_INFERENCE_TILING_DATA_H
