/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

struct SigmoidFocalLossGradTilingData {
    int64_t dim0;                // Elements: flattened N * C.
    int32_t coreNum;             // Cores: target AIV core count.
    int64_t blockFormer;         // Elements: work assigned to a former block.
    int64_t blockNum;            // Blocks: effective launched block count.
    int64_t ubFormer;            // Elements: work processed by a former UB tile.
    int64_t ubLoopOfFormerBlock; // Loops: UB tile loops of a former block.
    int64_t ubTailOfFormerBlock; // Elements: valid tail of a former block.
    int64_t ubLoopOfTailBlock;   // Loops: UB tile loops of the tail block.
    int64_t ubTailOfTailBlock;   // Elements: valid tail of the tail block.
    float alpha;                 // Runtime focal-loss positive-class weight.
    float gamma;                 // Runtime focal-loss focusing exponent.
    float reduceMeanCoef;        // Runtime reduction coefficient.
    int32_t weightDtype;         // 0: float16; 1: float32; ignored when weight is absent.
};
