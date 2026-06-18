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
 * \file fake_quant_with_min_max_vars_gradient_tilingdata.h
 * \brief TilingData structure for fake_quant_with_min_max_vars_gradient (arch35)
 *
 * Key difference from args version: no nudgedMin/nudgedMax fields.
 * min/max are tensor inputs read on Device side; nudged values computed in Kernel.
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_TILINGDATA_H_
#define FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_TILINGDATA_H_
#include <cstdint>

class FakeQuantWithMinMaxVarsGradientTilingData {
public:
    // ---- shape/tiling parameters ----
    int64_t totalLen;          // total element count (gradients/x)
    int64_t numCore;           // actual core count used
    int64_t blockFactor;       // elements per normal core (aligned)
    int64_t blockTailFactor;   // elements for tail core
    int64_t baseLen;           // single Compute tile size (UB unit)
    // ---- attributes (from Tiling, NOT tensor values) ----
    int64_t numBits;           // quantization bit width [2, 16]
    bool    narrowRange;       // narrow range flag
};

#endif // FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_TILINGDATA_H_
