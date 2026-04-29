/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add_with_sorted_simd_tiling.h
 * \brief scatter_add_with_sorted_simd_tiling
 */
#ifndef SCATTER_ADD_WITH_SORTED_SIMD_TILING_H
#define SCATTER_ADD_WITH_SORTED_SIMD_TILING_H

#include "scatter_add_with_sorted_tiling_base.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {

class ScatterAddWithSortedSimdTiling : public ScatterAddWithSortedBaseTiling 
{
public:
    explicit ScatterAddWithSortedSimdTiling(gert::TilingContext* context) : ScatterAddWithSortedBaseTiling(context)
    {}
    ~ScatterAddWithSortedSimdTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData() override;
    void DoBlockTiling();
    void DoUBTiling();
    void AutoTilingRowCol(
        int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum, int64_t rowTotalNum, int64_t colTotalNum);
    void DeterminTemplateUbTiling();

private:
    ScatterAddWithSortedSimdTilingData* tilingData_;
    int64_t normalCoreColNum_ = 0;
    int64_t coreNumInCol_ = 0;
    int64_t tailCoreColNum_ = 0;
    int64_t coreNumInRow_ = 0;
    int64_t normalCoreRowNum_ = 0;
    int64_t tailCoreRowNum_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t availableUbsize = 0;
    int64_t updatesBufferSize_ = 0;
    int64_t indicesBufferSize_ = 0;
    int64_t FrontAndBackIndex = 0;
    int64_t posBufferSize_ = 0;
    int64_t outBufferSize_ = 0;
    int64_t normalCoreRowUbLoop_ = 0;
    int64_t normalCoreNormalLoopRows_ = 0;
    int64_t normalCoreTailLoopRows_ = 0;
    int64_t tailCoreRowUbLoop_ = 0;
    int64_t tailCoreNormalLoopRows_ = 0;
    int64_t tailCoreTailLoopRows_ = 0;
    int64_t normalCoreColUbLoop_ = 0;
    int64_t normalCoreTailLoopCols_ = 0;
    int64_t normalCoreNormalLoopCols_ = 0;
    int64_t tailCoreColUbLoop_ = 0;
    int64_t tailCoreNormalLoopCols_ = 0;
    int64_t tailCoreTailLoopCols_ = 0;
    int64_t vecAlignSize_ = 0;
    int64_t colNumInUbDeterm = 0;

    int64_t resUb = 0;
    int64_t usedCoreNumForDetermin = 0;
    int64_t normalCoreDeterminCols_ = 0;
    int64_t tailCoreDeterminCols_ = 0;
    int64_t normalCoreDeterminColsUbLoop_ = 0;
    int64_t updatesDeterminBufferSize_ = 0;
    int64_t outBufferDeterminSize_ = 0;
    int64_t indicesBufferDeterminSize_ = 0;

    int64_t normalCoreDeterminNormalLoopCols_ = 0;
    int64_t normalCoreDeterminTailLoopCols_ = 0;
    int64_t tailCoreDeterminColsUbLoop_ = 0;
    int64_t tailCoreDeterminNormalLoopCols_ = 0;
    int64_t tailCoreDeterminTailLoopCols_ = 0;
    int64_t indicesWorkspaceBufferSize_ = 0;
    int64_t normalCoreColUbDetermLoop_ = 0;
    int64_t normalCoreNormalLoopDetermCols_ = 0;
    int64_t normalCoreTailLoopDetermCols_ = 0;
    int64_t normalCoreColDetermNum_ = 0;
    int64_t coreNumInColDeterm_ = 0;
    int64_t tailCoreColNumDeterm_ = 0;
    int64_t tailCoreColUbDetermLoop_ = 0;
    int64_t tailCoreNormalLoopDetermCols_ = 0;
    int64_t tailCoreTailLoopDetermCols_ = 0;
    int64_t ubBlock_ = 0;
};
} // namespace optiling
#endif // SCATTER_ADD_WITH_SORTED_SIMD_TILING_H