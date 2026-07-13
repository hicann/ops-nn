/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_v3_tiling_data.h
 * \brief tiling data struct shared between host and device
 */

#ifndef __MAX_POOL_V3_TILING_DATA_H__
#define __MAX_POOL_V3_TILING_DATA_H__

#include <cstdint>

// Double buffer count shared between kernel and tiling host.
// Must stay in sync with the kernel's inQueueX/outQueueY QUE buffer count.
constexpr uint32_t MAX_POOL_V3_BUFFER_NUM = 2;

struct MaxPoolV3TilingData {
    // Per-core output element distribution (swish-style core load balancing)
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t tileDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;

    // Shape information (needed for index calculation on device)
    uint32_t n;
    uint32_t c;
    uint32_t hIn;
    uint32_t wIn;
    uint32_t hOut;
    uint32_t wOut;

    // Pooling kernel parameters
    uint32_t kH;
    uint32_t kW;
    uint32_t sH;
    uint32_t sW;
    uint32_t padT;
    uint32_t padL;

    // Derived values for convenience
    uint32_t inHW;  // hIn * wIn
    uint32_t outHW; // hOut * wOut
};

#endif
