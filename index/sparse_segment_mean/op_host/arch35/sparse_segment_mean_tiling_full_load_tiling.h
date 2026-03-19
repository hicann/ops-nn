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
 * \file sparse_segment_mean_tiling_full_load_tiling.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_FULL_LOAD_TILING_H_
#define SPARSE_SEGMENT_MEAN_FULL_LOAD_TILING_H_

#include "sparse_segment_mean_tiling_base.h"

namespace optiling {
struct SparseSegmentMeanFullLoadInfo {
    int64_t usedCoreNum{0};
    int64_t normalCoreIndicesNum{0};
    int64_t tailCoreIndicesNum{0};

    int64_t xBufferSize{0};
    int64_t indicesBufferSize{0};

    int64_t threadNumX{0};
    int64_t threadNumY{0};
    int64_t useSimtMode{0};

    int64_t perCoreSegmentNum{0};
    int64_t resSegmentNum{0};
};

class SparseSegmentMeanFullLoadTiling : public SparseSegmentMeanBaseTiling
{
public:
    explicit SparseSegmentMeanFullLoadTiling(gert::TilingContext* context) : SparseSegmentMeanBaseTiling(context)
    {}

    ~SparseSegmentMeanFullLoadTiling() override = default;

private:
    void DoBlockTiling();
    void SetTilingData();
    void ThreadTiling();
    int64_t GetUpPow2(int64_t n);
    void PrinttilingData() const;
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    SparseSegmentMeanFullLoadInfo fullLoadData;
    int64_t needCoreNum_ = 0;
    int64_t normalCoreSegmentNum_ = 0; // 正常核处理的SegmentNum数
    int64_t secondToLastCoreSegmentNum_ = 0;     // 倒数第二个核处理的SegmentNum数
    int64_t lastCoreSegmentNum_ = 0; // 尾核处理的SegmentNum数
    bool specialBlockTiling_ = false;
};

} // namespace optiling

#endif