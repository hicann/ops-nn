/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GAUSSIAN_NLL_LOSS_GRAD_TILING_DATA_H_
#define GAUSSIAN_NLL_LOSS_GRAD_TILING_DATA_H_

#include <cstdint>

struct GaussianNllLossGradTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
    uint32_t totalDataNum;
    uint32_t targetDataNum;
    uint32_t varDataNum;
    uint32_t targetBroadcastAxisSize;
    uint32_t targetInnerStride;
    uint32_t targetBroadcastMode;
    uint32_t varBroadcastMode;
    uint32_t varReduceSize;
    uint32_t reduction;
    float eps;
    float meanScale;
};

#endif
