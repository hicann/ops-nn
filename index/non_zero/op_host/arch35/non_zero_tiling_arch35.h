/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file non_zero_tiling_arch35.h
 * \brief
 */

#ifndef CANN_OPS_INDEX_NON_ZERO_OP_HOST_NON_ZERO_TILING_ARCH35_H_
#define CANN_OPS_INDEX_NON_ZERO_OP_HOST_NON_ZERO_TILING_ARCH35_H_

#include "register/tilingdata_base.h"

namespace optiling {
constexpr int64_t MAX_DIM_NUM = 7;

struct NonZeroCompileInfo {
    int64_t block_dim;
    int64_t coreNum;
    int64_t ubSize;
    int64_t vRegSize;
};

BEGIN_TILING_DATA_DEF(NonZeroTilingData)
TILING_DATA_FIELD_DEF(int64_t, inputDims);        // 输入tensor维度
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);      // 实际使用的核数
TILING_DATA_FIELD_DEF(int64_t, numPerCore);       // 每个核的数据量
TILING_DATA_FIELD_DEF(int64_t, numTailCore);      // 分核余数
TILING_DATA_FIELD_DEF(int64_t, ubFactorNum);      // 单核UB的切分因子，一次计算能处理的最大数据量
TILING_DATA_FIELD_DEF(int64_t, loopNumPerCore);   // 每个核ub处理次数
TILING_DATA_FIELD_DEF(int64_t, loopTailPerCore);  // 每个核循环最后一次数据量
TILING_DATA_FIELD_DEF(int64_t, loopNumTailCore);  // 尾核ub循环次数
TILING_DATA_FIELD_DEF(int64_t, loopTailTailCore); // 尾核循环最后一次的数据量

TILING_DATA_FIELD_DEF(int64_t, needTranspose); // 是否转置，默认为false
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, offsetInt32Trans);
TILING_DATA_FIELD_DEF(int64_t, offsetInt64);
TILING_DATA_FIELD_DEF(int64_t, maskLoopNum);

TILING_DATA_FIELD_DEF(int64_t, loopNumO); // 前面核输出循环次数
TILING_DATA_FIELD_DEF(int64_t, beforeNumO);
TILING_DATA_FIELD_DEF(int64_t, loopTailO);
TILING_DATA_FIELD_DEF(int64_t, loopNumTo); // 尾核输出循环次数
TILING_DATA_FIELD_DEF(int64_t, loopTailTo);
TILING_DATA_FIELD_DEF(int64_t, maskLoopNumO);

TILING_DATA_FIELD_DEF(int64_t, xInputSize);
TILING_DATA_FIELD_DEF(int64_t, maskSize);

TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, mulInDimRList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, quickDivRKList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, quickDivRMList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonZero, NonZeroTilingData)
} // namespace optiling
#endif // CANN_OPS_INDEX_NON_ZERO_OP_HOST_NON_ZERO_TILING_ARCH35_H_
