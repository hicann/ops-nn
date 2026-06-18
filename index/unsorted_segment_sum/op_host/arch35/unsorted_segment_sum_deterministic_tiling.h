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
 * \file unsorted_segment_sum_deterministic_tiling.h
 * \brief unsorted_segment_sum_deterministic_tiling
 */

#ifndef UNSORTED_SEGMENT_SUM_DETERMINISTIC_TILING_H
#define UNSORTED_SEGMENT_SUM_DETERMINISTIC_TILING_H

#include "unsorted_segment_sum_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UnsortedSegmentSumDetermTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputOuterDim);
TILING_DATA_FIELD_DEF(uint64_t, outputOuterDim);
TILING_DATA_FIELD_DEF(uint64_t, innerDim);
TILING_DATA_FIELD_DEF(uint32_t, tmpBufferSize);
TILING_DATA_FIELD_DEF(uint32_t, rowsNumInUB);
TILING_DATA_FIELD_DEF(uint32_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_3100, UnsortedSegmentSumDetermTilingData);
REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_3101, UnsortedSegmentSumDetermTilingData);
REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_3102, UnsortedSegmentSumDetermTilingData);
REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_3200, UnsortedSegmentSumDetermTilingData);
REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_3201, UnsortedSegmentSumDetermTilingData);
REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_3202, UnsortedSegmentSumDetermTilingData);

class UnsortedSegmentSumDetermTiling : public UnsortedSegmentSumBaseTiling
{
public:
    explicit UnsortedSegmentSumDetermTiling(gert::TilingContext* context) : UnsortedSegmentSumBaseTiling(context)
    {}
    ~UnsortedSegmentSumDetermTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData();
    int64_t getRestAvailableSize(uint64_t sampleNum, int32_t originalSize);
    int64_t FindMaxRowsInUb();
    int64_t GetSortTmpSize(int64_t rowsNum);

    uint32_t sortSharedBufSize_;
    uint32_t rowsNumInUB_;
    uint32_t tailSampleNum_;
    uint32_t loopIncore_;
    uint32_t usedCoreNum_;
    uint32_t row32BAlign_;
    uint64_t normalCoreProcessNum_;
    uint64_t tailCoreProcessNum_;
    UnsortedSegmentSumDetermTilingData tilingData_;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SUM_DETERMINISTIC_TILING_H
