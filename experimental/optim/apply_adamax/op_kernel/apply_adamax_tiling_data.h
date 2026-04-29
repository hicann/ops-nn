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
/*!
 * \file apply_adamax_tiling_data.h
 * \brief ApplyAdamax TilingData structure
 */

#ifndef APPLY_ADAMAX_TILING_DATA_H_
#define APPLY_ADAMAX_TILING_DATA_H_

struct ApplyAdamaxTilingData {
    int64_t totalNum = 0;        // Total number of elements in var
    int64_t blockFactor = 0;     // Per-core element count (aligned to 32B/sizeof(T))
    int64_t ubFactor = 0;        // Per UB-loop element count (tileLen)
    float   beta1Power = 0.0f;   // beta1^t (from aclScalar)
    float   lr = 0.0f;           // Learning rate
    float   beta1 = 0.0f;        // First moment decay
    float   beta2 = 0.0f;        // Second moment decay
    float   epsilon = 0.0f;      // Numerical stability constant
    float   oneMinusBeta1 = 0.0f; // 1 - beta1, pre-computed
    float   lrAdj = 0.0f;        // lr / (1 - beta1Power), pre-computed
};

#endif  // APPLY_ADAMAX_TILING_DATA_H_
