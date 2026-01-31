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
 * \file segment_sum_simd_tiling.h
 * \brief segment_sum_simd_tiling
 */

#ifndef SEGMENT_SUM_SIMD_TILING_H
#define SEGMENT_SUM_SIMD_TILING_H

#include "segment_sum_tiling_base.h"
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {

class SegmentSumSimdTiling : public SegmentSumBaseTiling
{
public:
    explicit SegmentSumSimdTiling(gert::TilingContext* context) : SegmentSumBaseTiling(context)
    {}
    ~SegmentSumSimdTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData() override;
    bool IsAtomicSupport();
    void DoBlockTiling();
    void DoUBTiling();
    void DoSplitColUBTiling(int64_t availableUbsize);
    void AutoTilingRowCol(int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum, int64_t rowTotalNum, int64_t colTotalNum);
    void DoMultCoreAddTiling();

private:
    SegmentSumSimdTilingData* tilingData_;
    int64_t needCoreNum_ = 0;
    int64_t xBufferSize_ = 0;
    int64_t segmentIdBufferSize_ = 0;
    int64_t yBufferSize_ = 0;

    int64_t blockNumInRow_ = 0;
    int64_t blockNumInCol_ = 0;
    int64_t normalCoreInnerNum_ = 0;
    int64_t normalCoreOutterNum_ = 0;
    int64_t tailCoreInnerNum_ = 0;
    int64_t tailCoreOutterNum_ = 0;

    int64_t normalCoreRowUbLoop_ = 0;
    int64_t normalCoreNormalLoopOutters_ = 0;
    int64_t normalCoreTailLoopOutters_ = 0;
    int64_t tailCoreRowUbLoop_ = 0;
    int64_t tailCoreNormalLoopOutters_ = 0;
    int64_t tailCoreTailLoopOutters_ = 0;

    int64_t normalCoreColUbLoop_ = 1;
    int64_t normalCoreNormalLoopInners_ = 0;
    int64_t normalCoreTailLoopInners_ = 0;
    int64_t tailCoreColUbLoop_ = 1;
    int64_t tailCoreNormalLoopInners_ = 0;
    int64_t tailCoreTailLoopInners_ = 0;

    int64_t usedCoreNumForClear_ = 0;
    int64_t normalCoreClearNum_ = 0;
    int64_t tailCoreClearNum_ = 0;

    int64_t usedCoreNumForMultAdd_ = 0;
    int64_t normalCoreMultAddInners_ = 0;
    int64_t tailCoreMultAddInners_ = 0;

    int64_t normalCoreMultAddInnerLoop_ = 0;
    int64_t normalCoreMultAddNormalLoopInners_ = 0;
    int64_t normalCoreMultAddTailLoopInners_ = 0;
    int64_t tailCoreMultAddInnerLoop_ = 0;
    int64_t tailCoreMultAddNormalLoopInners_ = 0;
    int64_t tailCoreMultAddTailLoopInners_ = 0;

    int64_t multAddXBufferSize_ = 0;
    int64_t multAddYBufferSize_ = 0;

    bool isDeterministic_ = false;
    bool isAtomicSupport_ = false;
    bool isSplitCol_ = false;
};
} // namespace optiling
#endif // SEGMENT_SUM_SIMD_TILING_H