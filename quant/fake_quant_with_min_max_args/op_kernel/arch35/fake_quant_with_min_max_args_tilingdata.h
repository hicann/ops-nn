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
 * \file fake_quant_with_min_max_args_tilingdata.h
 * \brief
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_ARGS_TILINGDATA_H_
#define FAKE_QUANT_WITH_MIN_MAX_ARGS_TILINGDATA_H_
#include <cstdint>

class FakeQuantWithMinMaxArgsTilingData {
public:
    // ---- elewise tiling ----
    int64_t totalLen;          // 总元素数
    int64_t numCore;           // 实际使用核数
    int64_t blockFactor;       // 普通核每核元素数
    int64_t blockTailFactor;   // 尾核元素数
    int64_t baseLen;           // 单次 Compute 元素数（基本单元）
    // ---- nudge constants (TF Eigen) ----
    float   nudgedMin;         // 浮点下界
    float   nudgedMax;         // 浮点上界
    float   scale;             // 反量化比例 = (max-min)/(qMax-qMin)
    float   scaleInv;          // 量化比例（必须 re-divide，不能用 1/scale）
    float   quantZero;         // floor(-nudgedMin*scaleInv + 0.5)，作为浮点存放
};

#endif // FAKE_QUANT_WITH_MIN_MAX_ARGS_TILINGDATA_H_
