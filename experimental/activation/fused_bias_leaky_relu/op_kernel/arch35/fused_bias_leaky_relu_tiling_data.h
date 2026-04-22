/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_bias_leaky_relu_tiling_data.h
 * \brief FusedBiasLeakyRelu TilingData structure
 */

#ifndef _FUSED_BIAS_LEAKY_RELU_TILING_DATA_H_
#define _FUSED_BIAS_LEAKY_RELU_TILING_DATA_H_

struct FusedBiasLeakyReluTilingData {
    int64_t totalNum = 0;                // Total number of elements
    int64_t blockFactor = 0;             // Number of elements per core
    int64_t ubFactor = 0;                // Number of elements per UB loop iteration
    float negativeSlope = 0.2f;          // LeakyReLU negative slope (float32, default 0.2)
    float scale = 1.414213562373f;       // Scale factor (float32, default sqrt(2))
};

#endif
