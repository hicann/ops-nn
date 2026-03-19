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
 * \file sparse_segment_mean_tiling_simt_tiling.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_SIMT_TILING_H_
#define SPARSE_SEGMENT_MEAN_SIMT_TILING_H_

#include "sparse_segment_mean_tiling_base.h"

namespace optiling
{

class SparseSegmentMeanSimtTiling : public SparseSegmentMeanBaseTiling
{
public:
    explicit SparseSegmentMeanSimtTiling(gert::TilingContext* context)
        : SparseSegmentMeanBaseTiling(context)
    {
    }

    ~SparseSegmentMeanSimtTiling() override
    {
    }

private:
    uint64_t GetTilingKey() const override;
    void PrintTilingData() const;
    bool IsCapable() override;
    int64_t GetUpPow2(int64_t n);
    void CalcThreadTiling();
    void CalcBlockTiling();
    
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

    int64_t needCoreNum_ = 0;
    int64_t threadNum_ = 2048;
    int64_t threadNumX_ = 0;
    int64_t threadNumY_ = 0;
    int64_t perCoreSegmentNum_ = 0;
    int64_t resSegmentNum_ = 0;
    int64_t normalCoreSegmentNum_ = 0; // 正常核处理的SegmentNum数
    int64_t secondToLastCoreSegmentNum_ = 0;     // 倒数第二个核处理的SegmentNum数
    int64_t lastCoreSegmentNum_ = 0; // 尾核处理的SegmentNum数
    bool specialBlockTiling_ = false;
    bool isSmallInner_ = false;
    bool isSimtLoop_ = false;
};

}  // namespace optiling

#endif