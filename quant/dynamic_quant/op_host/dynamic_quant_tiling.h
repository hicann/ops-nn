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
 * \file dynamic_quant_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_TILING_H
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum); // numCore
TILING_DATA_FIELD_DEF(uint32_t, rowLen);
TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowPerHeadCore);
TILING_DATA_FIELD_DEF(uint32_t, rowPerTailCore);
TILING_DATA_FIELD_DEF(uint32_t, multiRowNumHeadCore);
TILING_DATA_FIELD_DEF(uint32_t, multiRowNumTailCore);
TILING_DATA_FIELD_DEF(uint32_t, innerLoopEle);   // innerLoopNumCol
TILING_DATA_FIELD_DEF(uint32_t, innerLoopTimes); // innerLoopTimes
TILING_DATA_FIELD_DEF(uint32_t, innerLoopTail);  // innerLoopTailNumCol
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, alignGroupNum);
TILING_DATA_FIELD_DEF(uint32_t, hasSmooth);
TILING_DATA_FIELD_DEF(uint32_t, unused);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
// add for 310P

TILING_DATA_FIELD_DEF(uint32_t, sizeH);
TILING_DATA_FIELD_DEF(uint32_t, sizeX);
TILING_DATA_FIELD_DEF(uint32_t, sizeZOut);
TILING_DATA_FIELD_DEF(uint32_t, sizeCopyRow);
TILING_DATA_FIELD_DEF(uint32_t, numCopyRow);
TILING_DATA_FIELD_DEF(uint32_t, numHeadCore);
TILING_DATA_FIELD_DEF(uint32_t, numTailCore);
TILING_DATA_FIELD_DEF(uint32_t, numHeadTimes);
TILING_DATA_FIELD_DEF(uint32_t, numTailTimes);
TILING_DATA_FIELD_DEF(uint32_t, numLastTailRow);
TILING_DATA_FIELD_DEF(uint32_t, alignType);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicQuant, DynamicQuantTilingData)
REGISTER_TILING_DATA_CLASS(DynamicQuantV2, DynamicQuantTilingData)
struct DynamicQuantCompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_TILING_H
