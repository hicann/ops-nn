// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#ifndef MAX_POOL2D_WITH_MASK_BACKWARD_TILING_KEY_H
#define MAX_POOL2D_WITH_MASK_BACKWARD_TILING_KEY_H

#include <cstdint>

namespace MaxPool2dWithMaskBackwardTilingKey {
// Normal path (no overlap / overlap)
constexpr uint64_t NORMAL = 0;
constexpr uint64_t NORMAL_OVERLAP = 100;
// Scatter path (unique target / overlap)
constexpr uint64_t SCATTER = 2;
constexpr uint64_t SCATTER_OVERLAP = 102;
// CutK path (no cut / cut do / cut do,kd / ... / cut nc variants)
constexpr uint64_t CUTK = 1;
constexpr uint64_t CUTK_DO = 21;
constexpr uint64_t CUTK_DO_KD_HO = 31;
constexpr uint64_t CUTK_DO_KD_HO_KH_WO = 41;
constexpr uint64_t CUTK_DO_KD = 51;
constexpr uint64_t CUTK_DO_KD_HO_KH = 61;
constexpr uint64_t CUTK_DO_KD_HO_KH_WO_KW = 71;
constexpr uint64_t CUTK_NC = 101;
constexpr uint64_t CUTK_NC_DO = 121;
constexpr uint64_t CUTK_NC_DO_KD_HO = 131;
constexpr uint64_t CUTK_NC_DO_KD_HO_KH_WO = 141;
constexpr uint64_t CUTK_NC_KD = 151;
constexpr uint64_t CUTK_NC_KD_HO_KH = 161;
constexpr uint64_t CUTK_NC_KD_HO_KH_WO_KW = 171;
// Atlas 300V Pro (arch20) keys are emitted by *_310p_tiling.cpp (20020..20073).
} // namespace MaxPool2dWithMaskBackwardTilingKey

#endif // MAX_POOL2D_WITH_MASK_BACKWARD_TILING_KEY_H
