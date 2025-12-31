/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_block_quant_i8_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_QUANT_I8_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_QUANT_I8_H
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "dynamic_block_quant_tiling.h"

namespace optiling {
namespace dynamic_block_quant_i8 {

class DynamicBlockQuantI8 {
public:
    explicit DynamicBlockQuantI8(gert::TilingContext* ctx) : context(ctx){};
    ge::graphStatus DoTiling(DynamicBlockQuantTilingParam& tilingParam);
    ge::graphStatus CheckParams(DynamicBlockQuantTilingParam& tilingParam);
    void SetTilingData(DynamicBlockQuantTilingData& tilingData, const DynamicBlockQuantTilingParam& tilingParam);

private:
    ge::graphStatus CheckDtype();
    ge::graphStatus GetAttr(DynamicBlockQuantTilingParam& tilingParam);
    ge::graphStatus GetAttrBlock(DynamicBlockQuantTilingParam& tilingParam);
    ge::graphStatus CheckShape(DynamicBlockQuantTilingParam& tilingParam);
    void CalcTilingKey(ge::DataType inputType, DynamicBlockQuantTilingParam& tilingParam);
    void CalcTilingData(DynamicBlockQuantTilingParam& tilingParam, const gert::Shape& xShape);

private:
    gert::TilingContext* context = nullptr;
    int64_t perCoreRowNum = 0;
    uint32_t tailRowList[MAX_CORE_COUNT] = {0};
    uint32_t tailColBlockStartList[MAX_CORE_COUNT] = {0};
    uint32_t tailColBlockEndList[MAX_CORE_COUNT] = {0};
};
} // namespace dynamic_block_quant_i8
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_QUANT_I8_H