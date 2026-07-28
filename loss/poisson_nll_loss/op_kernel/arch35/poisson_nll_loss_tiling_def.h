/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file poisson_nll_loss_tiling_def.h
 * \brief tiling data struct (hand-written kernel, reduction=none stage-1; pure POD, no atvoss deps)
 */

#ifndef __POISSON_NLL_LOSS_TILING_DATA_H__
#define __POISSON_NLL_LOSS_TILING_DATA_H__

#include <cstdint>

struct PoissonNllLossTilingData {
    int64_t totalNum = 0;    // total element count (input == target; == output when reduction=none)
    int64_t blockFactor = 0; // elements per core
    int64_t ubFactor = 0;    // elements per UB tile
    float eps = 1e-8f;       // eps for log_input=False
    float meanCof = 1.0f;    // 1/totalNum for reduction=mean (else unused)
    uint32_t logInput = 1;   // runtime: 1=log_input true, 0=false
    uint32_t full = 0;       // runtime: 1=full true, 0=false
    uint32_t reduction = 0;  // runtime: 0=none, 1=sum, 2=mean
};

#endif // __POISSON_NLL_LOSS_TILING_DATA_H__
