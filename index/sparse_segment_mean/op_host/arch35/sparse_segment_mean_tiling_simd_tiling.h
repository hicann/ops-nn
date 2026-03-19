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
 * \file sparse_segment_mean_tiling_simd_tiling.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_SIMD_TILING_H_
#define SPARSE_SEGMENT_MEAN_SIMD_TILING_H_

#include "sparse_segment_mean_tiling_base.h"

namespace optiling
{
struct SparseSegmentMeanSplitInfo {
    // DoBlockTiling
    int64_t normalCoreInnerNum{0};
    int64_t tailCoreInnerNum{0};
    int64_t innerOuter{0};

    int64_t normalCoreIndicesNum{0};
    int64_t tailCoreIndicesNum{0};
    int64_t indicesOuter{0};

    int64_t usedCoreNum{0};

    // DoUBTiling
    int64_t xBufferSize{0};
    int64_t yBufferSize{0};
    int64_t sharedTmpBufferSize{0};
    int64_t workspaceBufferSize{0};

    // clear
    int64_t normalCoreProcessNumForClear{0};
    int64_t tailCoreProcessNumForClear{0};
    int64_t usedCoreNumForClear{0};

    // mul core
    int64_t perCoreInnerElements{0};
    int64_t tailCoreInnerElements{0};
    int64_t usedCoreNumForMulCore{0};
    int64_t inBufferSize{0};
    int64_t outBufferSize{0};
};

class SparseSegmentMeanSimdTiling : public SparseSegmentMeanBaseTiling
{
public:
    explicit SparseSegmentMeanSimdTiling(gert::TilingContext* context)
        : SparseSegmentMeanBaseTiling(context)
    {
    }

    ~SparseSegmentMeanSimdTiling() override
    {
    }

private:
    void DoBlockTiling();
    void DoUBTiling();
    void SetTilingData();
    void PrintSplitData() const;
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    SparseSegmentMeanSplitInfo splitData;
};

}  // namespace optiling

#endif