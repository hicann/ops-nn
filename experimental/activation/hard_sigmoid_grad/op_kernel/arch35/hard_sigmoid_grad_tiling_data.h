/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef _HARD_SIGMOID_GRAD_TILING_DATA_H_
#define _HARD_SIGMOID_GRAD_TILING_DATA_H_

struct HardSigmoidGradTilingData {
    int64_t totalLength = 0;    // total number of elements
    int64_t blockFactor = 0;    // elements per core
    int64_t ubFactor = 0;       // elements per UB tile
    float alpha = 0.16666666f;  // alpha parameter
    float beta = 0.5f;          // beta parameter
};

#endif
