/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_tiling.cpp
 * \brief Architecture-independent attribute validation.
 */

#include "dynamic_augru_tiling.h"

namespace optiling {
// Return the unsupported attribute name, or nullptr for a supported configuration.
// Exact comparisons also reject NaN/Inf for the two fixed floating-point attributes.
const char* InvalidDynamicAUGRUAttribute(const DynamicAUGRUAttributes& attrs)
{
    if (attrs.direction != "UNIDIRECTIONAL") {
        return "direction";
    }
    if (attrs.cellDepth != 1) {
        return "cell_depth";
    }
    if (attrs.keepProb != 1.0F) {
        return "keep_prob";
    }
    if (attrs.cellClip != -1.0F) {
        return "cell_clip";
    }
    if (attrs.numProj != 0) {
        return "num_proj";
    }
    if (!attrs.timeMajor) {
        return "time_major";
    }
    if (attrs.activation != "tanh") {
        return "activation";
    }
    if (attrs.gateOrder != "zrh" && attrs.gateOrder != "rzh") {
        return "gate_order";
    }
    if (!attrs.resetAfter) {
        return "reset_after";
    }
    // Both bool values of is_training are legal. Both compute all seven outputs.
    return nullptr;
}

} // namespace optiling
