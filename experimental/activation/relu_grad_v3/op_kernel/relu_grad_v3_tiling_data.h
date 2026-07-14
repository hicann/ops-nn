/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu_grad_v3_tiling_data.h
 * \brief tiling data struct for multi-core ReLU gradien
 */

#ifndef _RELU_GRAD_V3_TILING_DATA_H_
#define _RELU_GRAD_V3_TILING_DATA_H_

struct ReluGradV3TilingData {
    uint64_t totalLength = 0;
    uint64_t smallCoreDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t ubPartDataNum = 0;
    uint64_t smallCoreTailDataNum = 0;
    uint64_t bigCoreTailDataNum = 0;
    uint64_t smallCoreLoopNum = 0;
    uint64_t bigCoreLoopNum = 0;
    uint64_t tailBlockNum = 0;
    uint64_t broadcastMode = 0;
    uint64_t dimNum = 0;
    uint64_t xElementNum = 0;
    uint64_t yElementNum = 0;
    uint64_t outShape[8] = {0};
    uint64_t xStrides[8] = {0};
    uint64_t yStrides[8] = {0};
};
#endif
