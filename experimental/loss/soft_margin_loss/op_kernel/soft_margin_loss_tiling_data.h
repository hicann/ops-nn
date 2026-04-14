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

/**
 * \file soft_margin_loss_tiling_data.h
 * \brief SoftMarginLoss TilingData structure definition
 */

#ifndef _SOFT_MARGIN_LOSS_TILING_DATA_H_
#define _SOFT_MARGIN_LOSS_TILING_DATA_H_

struct SoftMarginLossTilingData {
    int64_t totalNum = 0;       // Total number of elements
    int64_t blockFactor = 0;    // Number of elements per AI Core
    int64_t ubFactor = 0;       // Number of elements per UB loop iteration (in float32 units)
    int32_t reductionMode = 0;  // Reduction mode: 0=none, 1=mean, 2=sum
    float invNumel = 0.0f;      // 1.0f / totalNum (only used in mean mode)
    int64_t usedCoreNum = 0;    // Actual number of cores used (for cross-core reduction)
};

#endif
