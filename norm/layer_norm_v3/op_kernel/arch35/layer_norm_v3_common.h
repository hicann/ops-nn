/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_v3_common.h
 * \brief
 */

#ifndef LAYER_NORM_V3_COMMON_H
#define LAYER_NORM_V3_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormV3 {
using namespace AscendC;

constexpr static int64_t BLOCK_SIZE = 32;
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static int64_t MAX_STRIDE = 65535;
constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int64_t LOOP_HIGH_SHIFT = 21;
constexpr static int64_t LOOP_STRIDE_HIGH_SHIFT = 40;
constexpr static uint32_t NUM_TWO = 2;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static float zero = 0.0f;
constexpr static uint32_t VL_B32 = GetVecLen() / sizeof(float);
constexpr static uint32_t VL_B16 = GetVecLen() / sizeof(int16_t);
constexpr static uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

} // namespace LayerNormV3

#endif // LAYER_NORM_V3_COMMON_H
