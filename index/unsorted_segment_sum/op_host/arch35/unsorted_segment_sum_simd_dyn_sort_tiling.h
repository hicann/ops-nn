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
 * \file unsorted_segment_sum_simd_dyn_sort_tiling.h
 * \brief unsorted_segment_sum_simd_dyn_sort_tiling
 */

#ifndef UNSORTED_SEGMENT_SUM_SIMD_DYN_SORT_TILING_H
#define UNSORTED_SEGMENT_SUM_SIMD_DYN_SORT_TILING_H

#include "unsorted_segment_sum_tiling.h"


namespace optiling {

BEGIN_TILING_DATA_DEF(UnsortedSegmentSumSimdDynSortTilingData)
TILING_DATA_FIELD_DEF(uint64_t, outputOuterDim); // segmentNum_
TILING_DATA_FIELD_DEF(uint64_t, innerDim);
TILING_DATA_FIELD_DEF(uint64_t, sTileNum);
TILING_DATA_FIELD_DEF(uint64_t, aTileNum);
TILING_DATA_FIELD_DEF(uint64_t, normBlockS);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockS);
TILING_DATA_FIELD_DEF(uint64_t, normBlockA);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockA);
TILING_DATA_FIELD_DEF(uint64_t, baseS);
TILING_DATA_FIELD_DEF(uint64_t, baseA);
TILING_DATA_FIELD_DEF(uint64_t, sortBaseS);
TILING_DATA_FIELD_DEF(uint64_t, sortBaseA);
TILING_DATA_FIELD_DEF(uint64_t, sortSharedBufSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_7000, UnsortedSegmentSumSimdDynSortTilingData);

class UnsortedSegmentSumSimdDynSortTiling : public UnsortedSegmentSumBaseTiling
{
public:
    explicit UnsortedSegmentSumSimdDynSortTiling(gert::TilingContext* context) : UnsortedSegmentSumBaseTiling(context)
    {}
    ~UnsortedSegmentSumSimdDynSortTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData();
    void DoBlockTiling();
    uint64_t CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd);

    uint64_t sTileNum_ = 0;
    uint64_t aTileNum_ = 0;
    uint64_t normBlockS_ = 0;
    uint64_t tailBlockS_ = 0;
    uint64_t normBlockA_ = 0;
    uint64_t tailBlockA_ = 0;
    uint64_t baseS_ = 1;
    uint64_t baseA_ = 1;
    uint64_t sortBaseS_ = 1;
    uint64_t sortBaseA_ = 1;
    uint32_t sortSharedBufSize_ = 0;
    UnsortedSegmentSumSimdDynSortTilingData tilingData_;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SUM_SIMD_DYN_SORT_TILING_H
