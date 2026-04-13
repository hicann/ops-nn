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
 * \file unsorted_segment_sum_deterministic_big_innerdim_tiling.h
 * \brief unsorted_segment_sum_deterministic_big_innerdim_tiling
 */

#ifndef UNSORTED_SEGMENT_SUM_DETERMINISTIC_BIG_INNERDIM_TILING_H
#define UNSORTED_SEGMENT_SUM_DETERMINISTIC_BIG_INNERDIM_TILING_H

#include "unsorted_segment_sum_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnsortedSegmentSumDeterministicBigInnerDimTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputOuterDim);  
TILING_DATA_FIELD_DEF(uint64_t, outputOuterDim); 
TILING_DATA_FIELD_DEF(uint64_t, innerDim);
TILING_DATA_FIELD_DEF(uint64_t, normalCoreProcessCols);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreProcessCols);
TILING_DATA_FIELD_DEF(uint64_t, baseS);
TILING_DATA_FIELD_DEF(uint64_t, baseA);
TILING_DATA_FIELD_DEF(uint32_t, sortSharedBufSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_8000, UnsortedSegmentSumDeterministicBigInnerDimTilingData);

class UnsortedSegmentSumDeterministicBigInnerDimTiling : public UnsortedSegmentSumBaseTiling
{
public:
    explicit UnsortedSegmentSumDeterministicBigInnerDimTiling(gert::TilingContext* context) : UnsortedSegmentSumBaseTiling(context)
    {}
    ~UnsortedSegmentSumDeterministicBigInnerDimTiling() override
    {}
private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData();
    int64_t FindMaxColsInUb();
    int64_t GetSortTmpSize(int64_t rowsNum);

    uint64_t normalCoreProcessCols_ = 0;
    uint64_t tailCoreProcessCols_ = 0;
    uint64_t baseS_ = 1;
    uint64_t baseA_ = 1;
    uint32_t sortSharedBufSize_ = 0;
    uint32_t col32BAlign_ = 0;

    UnsortedSegmentSumDeterministicBigInnerDimTilingData tilingData_;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SUM_DETERMINISTIC_BIG_INNERDIM_TILING_H
