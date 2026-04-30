/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_TILING_DATA_H
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_TILING_DATA_H

#include <cstdint>

struct SigmoidCrossEntropyWithLogitsGradV2TilingData {
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t finalBigTileNum;
    uint64_t finalSmallTileNum;
    uint64_t tileDataNum;
    uint64_t smallTailDataNum;
    uint64_t bigTailDataNum;
    uint64_t tailBlockNum;
    float globalScale;
    uint32_t has_weight;
    uint32_t has_pos_weight;
    uint32_t fp32_lite_mode;
};

#endif // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_TILING_DATA_H
