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
 * \file unsorted_segment_sum_simd_spilt_col_tiling.h
 * \brief unsorted_segment_sum_simd_spilt_col_tiling
 */

#ifndef UNSORTED_SEGMENT_SUM_SIMD_SPLIT_COL_TILING_H
#define UNSORTED_SEGMENT_SUM_SIMD_SPLIT_COL_TILING_H

#include "unsorted_segment_sum_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UnsortedSegmentSumSimdSplitColTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputOuterDim);  // totalSampleNum_
TILING_DATA_FIELD_DEF(uint64_t, outputOuterDim); // segmentNum_
TILING_DATA_FIELD_DEF(uint64_t, innerDim);
TILING_DATA_FIELD_DEF(uint64_t, normBlockData);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockData);
TILING_DATA_FIELD_DEF(uint64_t, baseS);
TILING_DATA_FIELD_DEF(uint64_t, baseA);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_5000, UnsortedSegmentSumSimdSplitColTilingData);

class UnsortedSegmentSumSimdSplitColTiling : public UnsortedSegmentSumBaseTiling
{
public:
    explicit UnsortedSegmentSumSimdSplitColTiling(gert::TilingContext* context) : UnsortedSegmentSumBaseTiling(context)
    {}
    ~UnsortedSegmentSumSimdSplitColTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData();
    bool IsFullLoad();

    uint64_t normBlockData_ = 0;
    uint64_t tailBlockData_ = 0;
    uint64_t baseS_ = 1;
    uint64_t baseA_ = 1;
    uint64_t outUbsize_ = 0;
    UnsortedSegmentSumSimdSplitColTilingData tilingData_;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SUM_SIMD_SPLIT_COL_TILING_H
