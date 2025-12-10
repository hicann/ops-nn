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
 * \file dynamic_quant_update_scatter_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_H
#include <cstdint>
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicQuantUpdateScatterV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowLen);
TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowPerHeadCore);
TILING_DATA_FIELD_DEF(uint32_t, rowPerTailCore);
TILING_DATA_FIELD_DEF(uint32_t, multiRowNumHeadCore);
TILING_DATA_FIELD_DEF(uint32_t, multiRowNumTailCore);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, dstSeqLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicQuantUpdateScatterV2, DynamicQuantUpdateScatterV2TilingData)
struct DynamicQuantUpdateScatterV2CompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_H