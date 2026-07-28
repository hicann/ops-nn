/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mse_loss_v2_tiling_data.h
 * \brief MSELossV2 arch35 tiling data struct (plain struct, shared by host tiling and kernel)
 */

#ifndef MSE_LOSS_V2_ARCH35_TILING_DATA_H
#define MSE_LOSS_V2_ARCH35_TILING_DATA_H

#include <cstdint>

struct MSELossV2Arch35TilingData {
    int64_t totalNum = 0;    // total number of elements (input flattened to 1D)
    int64_t blockFactor = 0; // number of elements per AI Core
    int64_t ubFactor = 0;    // number of elements per UB iteration
    float meanCof = 1.0f;    // 1 / totalNum, applied for reduction=mean (unused for none/sum)
    uint32_t reduction = 0;  // runtime reduction: 0=none, 1=sum, 2=mean
};

#endif // MSE_LOSS_V2_ARCH35_TILING_DATA_H
