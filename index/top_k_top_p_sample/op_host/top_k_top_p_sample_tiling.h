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
 * \file top_k_top_p_sample_tiling.h
 * \brief
 */
 
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_TOP_K_TOP_P_SAMPLE_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_TOP_K_TOP_P_SAMPLE_TILING_H

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "util/math_util.h"

 namespace optiling {

BEGIN_TILING_DATA_DEF(TopKTopPSampleTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, rowNum);
TILING_DATA_FIELD_DEF(uint32_t, rowLen);
TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, perHeadCoreRowNum);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreRowNum);
TILING_DATA_FIELD_DEF(uint32_t, perHeadCorePartNum);
TILING_DATA_FIELD_DEF(uint32_t, tailCorePartNum);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopEle);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopTime);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopEleTail);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopEleTailPad);
TILING_DATA_FIELD_DEF(uint32_t, softmaxLoopTime);
TILING_DATA_FIELD_DEF(uint32_t, softmaxLoopEleTail);
TILING_DATA_FIELD_DEF(uint32_t, softmaxLoopEleTailPad);
TILING_DATA_FIELD_DEF(uint32_t, eightKPartNum);
TILING_DATA_FIELD_DEF(uint32_t, eightKPartTail);
TILING_DATA_FIELD_DEF(uint32_t, eightKPartTailPad);
TILING_DATA_FIELD_DEF(uint32_t, mrgMode);
TILING_DATA_FIELD_DEF(uint32_t, headOffset);
TILING_DATA_FIELD_DEF(uint32_t, isNeedLogits);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, topKGuess);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TopKTopPSample, TopKTopPSampleTilingData)

struct TopKTopPSampleCompileInfo {
    uint32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
    uint32_t sysWorkspaceSize = 0;
};
}  // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_TOP_K_TOP_P_SAMPLE_TILING_H