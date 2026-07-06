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
 * \file fake_quant_with_min_max_vars_per_channel_tiling_data.h
 * \brief FakeQuantWithMinMaxVarsPerChannel TilingData (arch35)
 *
 * 字段集 100% 对齐 ascend-quant-ops-design skill `QuantizeTilingData` 通用结构：
 *   - skill 通用：numCore / blockAxis / blockFactor / blockTailFactor / blockUnion /
 *                 ubAxis / baseN / baseLen / axis / dim0 / dim1 / dim2 / hasZeroPoint
 *   - Native PC 专用扩展：headNum / tailDim
 *   - 本算子专用：quantMin / quantMax / numBits / narrowRange
 *
 * DESIGN.md v2.1 §3.2
 */
#ifndef _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_TILING_DATA_H_
#define _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_TILING_DATA_H_

#include <cstdint>

struct FakeQuantWithMinMaxVarsPerChannelTilingData {
    // === 核切分（skill §3 通用）===
    int64_t numCore = 0;         // 实际活跃核数
    int64_t blockAxis = 0;       // Native PC: 0 (切 dim0=M) 或 1 (切 dim1=N)
    int64_t blockFactor = 0;     // 非尾核块大小
    int64_t blockTailFactor = 0; // 尾核块大小
    int64_t blockUnion = 1;      // Native PC 置 1（per-head 才用）

    // === UB 切分（skill §3 通用）===
    int64_t ubAxis = 0;  // Native PC: 0 (多行) 或 1 (单行)
    int64_t baseN = 1;   // UB 块行数
    int64_t baseLen = 0; // UB 块列宽（VL=64 对齐）

    // === 合轴后形状（skill §3 通用）===
    int64_t axis = -1; // 占位（本算子尾轴）
    int64_t dim0 = 0;  // M = headNum
    int64_t dim1 = 0;  // N = tailDim
    int64_t dim2 = 0;  // 占位 0（PH 才用）

    // === 选项（skill §3 通用）===
    int64_t hasZeroPoint = 0; // 本算子恒 0

    // === Native PC 专用扩展 ===
    int64_t headNum = 0; // 同 dim0
    int64_t tailDim = 0; // 同 dim1

    // === 本算子专用字段 ===
    float quantMin = 0.0f;
    float quantMax = 0.0f;
    int32_t numBits = 8;
    int32_t narrowRange = 0;
};

#endif // _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_TILING_DATA_H_
