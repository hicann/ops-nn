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
 * \file in_training_update_grad_gamma_beta_tiling_data.h
 * \brief Tiling data shared by the host and Ascend 950 kernel.
 */

#ifndef IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_DATA_H_
#define IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_DATA_H_

#include <cstdint>

// A compressed FP32 expansion covers exponents -149 through 127 with twelve
// components. Reserve one extra slot while inserting the next input value;
// compress after every insertion, before this scratch slot is needed again.
constexpr uint32_t IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_MAX_PARTIALS = 13U;

struct INTrainingUpdateGradGammaBetaTilingData {
    int64_t reduceCount;
    int64_t outputElements;
    int64_t baseBlocksPerCore;
    uint32_t tileElements;
    uint32_t reduceRowsPerTile;
    uint32_t extraBlockCoreCount;
    uint32_t blockElements;
    uint32_t usedCoreNum;
    float safeMagnitude;
    float inputScale;
    float outputScale;
};

#endif // IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_DATA_H_
