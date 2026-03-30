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
 * \file sorted_sparse_segment_mean_grad_tiling_simt_tiling.h
 * \brief
 */
 
#ifndef SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_TILING_H_
#define SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_TILING_H_

#include "sorted_sparse_segment_mean_grad_tiling_base.h"
#include "index/sorted_sparse_segment_mean_grad/op_kernel/arch35/sorted_sparse_segment_mean_grad_struct.h" 

namespace optiling
{

class SortedSparseSegmentMeanGradSimtTiling : public SortedSparseSegmentMeanGradBaseTiling
{
public:
    explicit SortedSparseSegmentMeanGradSimtTiling(gert::TilingContext* context)
        : SortedSparseSegmentMeanGradBaseTiling(context)
    {
    }

    ~SortedSparseSegmentMeanGradSimtTiling() override = default;

private:
    uint64_t GetTilingKey() const override;
    void PrintTilingData() const;
    bool IsCapable() override;
    int64_t GetUpPow2(int64_t n);
    void CalcThreadTiling();
    void CalcBlockTiling();
    void CalcSegIndexThreadTiling(int32_t& threadNumX, int32_t& threadNumY, int64_t num);
    
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

    int64_t needCoreNum_ = 0;
    int64_t threadNum_ = 512;
    int64_t threadNumX_ = 0;
    int64_t threadNumY_ = 0;
    int32_t segThreadNumX_ = 0;
    int32_t segThreadNumY_ = 0;
    int32_t indexThreadNumX_ = 0;
    int32_t indexThreadNumY_ = 0;
    int64_t perCoreIndicesNum_ = 0;
    int64_t resIndicesNum_ = 0;
    int64_t innerCore_ = 1;
    int64_t perCoreInnerSize_ = 0;
    int64_t resCoreInnerSize_ = 0;
    bool isSmallInner_ = false;
};

}  // namespace optiling

#endif