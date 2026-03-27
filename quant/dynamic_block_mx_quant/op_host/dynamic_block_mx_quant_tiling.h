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
 * \file dynamic_block_mx_quant_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_MX_QUANT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_MX_QUANT_TILING_H
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/arch35/dynamic_block_mx_quant_tilingdata.h"

using namespace Ops::NN::Optiling;

namespace optiling {
struct DynamicBlockMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicBlockMxQuantTilingParam {
    int64_t tilingKey = 0;
    int64_t totalCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t ubSize = 0;
    int64_t roundMode = 0;
    int64_t dstType = 0;
    int64_t scaleAlg = 0;
    float dstTypeMax = 0;
    int64_t blockSizeRow = 0;
    int64_t blockSizeCol = 0;
    int64_t batchNum = 0;
    int64_t rowNum = 0;
    int64_t colNum = 0;
    int64_t singleBatchRowBlockLoopNum = 0;
    int64_t rowBlockLoopNum = 0;
    int64_t colBlockLoopNum = 0;
    int64_t rowUbBlockLoopNum = 0;
    int64_t colUbBlockLoopNum = 0;
    int64_t rowUbFactor = 0;
    int64_t colUbFactor = 0;
    int64_t rowTileNum = 0;
    int64_t colTileNum = 0;
    int64_t normalCoreRowTileNum = 0;
    int64_t normalCoreColTileNum = 0;
    int64_t tailCoreRowTileNum = 0;
    int64_t tailCoreColTileNum = 0;
    int64_t rowNormalCoreNum = 0;
    int64_t colNormalCoreNum = 0;
    int64_t rowTailCoreNum = 0;
    int64_t colTailCoreNum = 0;
    int64_t blockH = 0;
    int64_t blockW = 0;
    int64_t rowScaleNum = 0;
    int64_t colScaleNum = 0;
    int64_t workspaceSize = 0;
};

enum class RoundModeList
{
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

class DynamicBlockMxQuantTiling {
public:
    explicit DynamicBlockMxQuantTiling(gert::TilingContext* context) : context_(context)
    {}
    ~DynamicBlockMxQuantTiling()
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckShape() const;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus SetTilingParams();
    ge::graphStatus AutoTiling();
    ge::graphStatus CalcAxisSize();

    void SetTilingKey();
    void SetTilingData();
    void PrintTilingData();
    void SplitCore();

    RoundModeList GetRoundMode(const std::string& roundMode);

private:
    uint64_t roundMode_ = 0;
    uint64_t scaleAlg_ = 0;
    gert::TilingContext* context_ = nullptr;
    DynamicBlockMxQuantTilingData tilingData;
    DynamicBlockMxQuantTilingParam tilingParams;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_BLOCK_MX_QUANT_TILING_H
