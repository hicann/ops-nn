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
 * \file rms_norm_quant_common_tiling.h
 * \brief
 */

#ifndef ASCEND_OPS_RMS_NORM_ATB_COMMON_H
#define ASCEND_OPS_RMS_NORM_ATB_COMMON_H

#include <register/tilingdata_base.h>

namespace optiling {

enum class NormInputIndex : int32_t
{
    X = 0,
    GAMMA = 1,
    BETA = 2,
    RES_IN = 3
};

enum class NormOutputIndex : int32_t
{
    Y = 0,
    RES_OUT
};

enum class NormAttrIndex : int32_t
{
    EPSILON = 0,
    HIGH_PRECISION_MODE = 1,
    GEMMA_MODE = 2,
    // SCALE = 3,
    // OFFSET = 4
};

// compile info defination
struct NormCommonCompileInfo {
    uint32_t coreNumAiv = 0;
    uint32_t ubSize = 0;
    uint32_t sysWorkspaceSize = 0;
};

// tiling data defination
BEGIN_TILING_DATA_DEF(NormCommonTilingData1)
TILING_DATA_FIELD_DEF(uint32_t, numCore);       // 实际使用核数
TILING_DATA_FIELD_DEF(uint32_t, numCol);        // 最后一维大小
TILING_DATA_FIELD_DEF(uint32_t, numRow);        // batch数量，xShape = (numRow..., numCol)
TILING_DATA_FIELD_DEF(float, avgFactor);        // numCol的倒数
TILING_DATA_FIELD_DEF(float, epsilon);          // norm参数
TILING_DATA_FIELD_DEF(uint32_t, sliceSize);     // 每核计算batch数
TILING_DATA_FIELD_DEF(uint32_t, sliceNum);      // 核数？
TILING_DATA_FIELD_DEF(uint32_t, tailSliceSize); // 尾块？sliceSize*sliceNum+tailSliceSize = numRow?
TILING_DATA_FIELD_DEF(int32_t, quantMin);       // 对称量化范围到-127
TILING_DATA_FIELD_DEF(float, scale);            // 量化缩放
TILING_DATA_FIELD_DEF(int32_t, offset);         // 量化偏移
TILING_DATA_FIELD_DEF(bool, highPrecisionMode); // 高精度模式，兼容未模板化
TILING_DATA_FIELD_DEF(bool, gemmaMode);         // 在gamma上+1，兼容未模板化
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormQuant, NormCommonTilingData1)

} // namespace optiling

#endif