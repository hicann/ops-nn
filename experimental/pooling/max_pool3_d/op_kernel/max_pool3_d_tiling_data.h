/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL3_D_TILING_DATA_H_
#define MAX_POOL3_D_TILING_DATA_H_

#include <cstdint>

struct MaxPool3DTilingData {
    uint64_t totalOut = 0;
    uint64_t normalCoreOut = 0;
    uint64_t splitOut = 0;
    uint64_t splitQuantum = 1;
    int64_t n = 0;
    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t c = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t kD = 1;
    int64_t kH = 1;
    int64_t kW = 1;
    int64_t sD = 1;
    int64_t sH = 1;
    int64_t sW = 1;
    int64_t padFront = 0;
    int64_t padTop = 0;
    int64_t padLeft = 0;
    int64_t dilationD = 1;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
    uint32_t dataFormat = 0;
    uint32_t outputLayout = 0;
    int64_t outputD = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int64_t outputC1 = 1;
    int64_t outputC0 = 0;
    int64_t outputC0Block = 0;
    uint32_t inputLayout = 0;
    int64_t inputC1 = 1;
    int64_t inputC0 = 0;
    int64_t inputC0Block = 0;
    uint32_t blockDim = 1;
    uint32_t balancedSplit = 0;
};

#endif // MAX_POOL3_D_TILING_DATA_H_
