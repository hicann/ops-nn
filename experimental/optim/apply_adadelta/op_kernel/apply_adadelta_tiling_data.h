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
 * \file apply_adadelta_tiling_data.h
 * \brief ApplyAdadelta TilingData structure
 */

#ifndef _APPLY_ADADELTA_TILING_DATA_H_
#define _APPLY_ADADELTA_TILING_DATA_H_

struct ApplyAdadeltaTilingData {
    int64_t totalNum = 0;       // Total number of elements
    int64_t blockFactor = 0;    // Number of elements per core (aligned to UB block)
    int64_t ubFactor = 0;       // Number of elements per UB loop iteration (aligned to UB block)
    float   lr = 0.0f;          // Learning rate (from aclScalar, stored as fp32)
    float   rho = 0.0f;         // Decay coefficient
    float   epsilon = 0.0f;     // Numerical stability constant
    float   oneMinusRho = 0.0f; // 1 - rho, pre-computed to avoid Kernel redundant computation
};

#endif
