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
 * \file non_zero_with_value_tiling_arch35.h
 * \brief NonZeroWithValue arch35 tiling —— 2D/fp32/transpose=true(坐标主序)专用。
 */

#ifndef CANN_OPS_INDEX_NON_ZERO_WITH_VALUE_OP_HOST_NON_ZERO_WITH_VALUE_TILING_ARCH35_H_
#define CANN_OPS_INDEX_NON_ZERO_WITH_VALUE_OP_HOST_NON_ZERO_WITH_VALUE_TILING_ARCH35_H_

#include "register/tilingdata_base.h"

namespace optiling {

struct NonZeroWithValueCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
    int64_t vRegSize;
};

BEGIN_TILING_DATA_DEF(NonZeroWithValueTilingData)
// 2D 形状(严格 2D):x = [row, col],numel = row*col
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, numel);

// 核间切分(按展平线性 numel 切,对齐 A2/NonZero)
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);

// UB 切分:计数遍(找非零 + 归约计数)
TILING_DATA_FIELD_DEF(int64_t, ubFactorNum);      // 单核一次能处理的最大元素数(VF 对齐)
TILING_DATA_FIELD_DEF(int64_t, loopNumPerCore);   // 主核 UB 循环次数
TILING_DATA_FIELD_DEF(int64_t, loopTailPerCore);  // 主核尾块元素数
TILING_DATA_FIELD_DEF(int64_t, loopNumTailCore);  // 尾核 UB 循环次数
TILING_DATA_FIELD_DEF(int64_t, loopTailTailCore); // 尾核尾块元素数

// UB 切分:输出遍(压缩坐标 + gather value + 写出)
TILING_DATA_FIELD_DEF(int64_t, loopNumO);
TILING_DATA_FIELD_DEF(int64_t, beforeNumO);
TILING_DATA_FIELD_DEF(int64_t, loopTailO);
TILING_DATA_FIELD_DEF(int64_t, loopNumTo);
TILING_DATA_FIELD_DEF(int64_t, loopTailTo);

// buffer 尺寸 / scratch 偏移
TILING_DATA_FIELD_DEF(int64_t, xInputSize);   // x 输入 buffer 元素数
TILING_DATA_FIELD_DEF(int64_t, valueBufSize); // value gather buffer 尺寸(本算子新增通道)

// 线性 idx → (row, col) 的 col 快速除法常量:row = idx/col, col = idx%col
TILING_DATA_FIELD_DEF(int64_t, quickDivColK);
TILING_DATA_FIELD_DEF(int64_t, quickDivColM);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonZeroWithValue, NonZeroWithValueTilingData)
} // namespace optiling
#endif // CANN_OPS_INDEX_NON_ZERO_WITH_VALUE_OP_HOST_NON_ZERO_WITH_VALUE_TILING_ARCH35_H_
