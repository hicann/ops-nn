/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _AVERAGE_POOLING_GRAD_TILING_DATA_H_
#define _AVERAGE_POOLING_GRAD_TILING_DATA_H_

struct AveragePoolingGradTilingData {
    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t hIn = 0;
    uint32_t wIn = 0;
    uint32_t hOut = 0;
    uint32_t wOut = 0;
    uint32_t kernelH = 0;
    uint32_t kernelW = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t padTop = 0;
    uint32_t padLeft = 0;
    uint32_t countIncludePad = 1;
    int32_t divisorOverride = 0;
    float invDivisor = 1.0f;
    uint32_t totalElements = 0;
    uint32_t normalCoreProcessNum = 0;
    uint32_t tailCoreProcessNum = 0;
    uint32_t usedCoreNum = 0;
    uint32_t tileH = 1;
    uint32_t tileW = 1;
    uint32_t tileTaskNum = 0;
    uint32_t gradOutCacheDataNum = 0;
};

#endif
