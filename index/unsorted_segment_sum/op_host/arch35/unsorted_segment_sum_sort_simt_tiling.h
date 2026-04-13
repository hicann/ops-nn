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
 * \file unsorted_segment_sum_sort_simt_tiling.h
 * \brief unsorted_segment_sum_sort_simt_tiling
 */
#ifndef UNSORTED_SEGMENT_SUM_SORT_SIMT_TILING_H
#define UNSORTED_SEGMENT_SUM_SORT_SIMT_TILING_H

#include "unsorted_segment_sum_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UnsortedSegmentSumSortSimtTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputOuterDim);
TILING_DATA_FIELD_DEF(uint64_t, outputOuterDim);
TILING_DATA_FIELD_DEF(uint64_t, innerDim);
TILING_DATA_FIELD_DEF(uint64_t, maxIndexNum);
TILING_DATA_FIELD_DEF(uint64_t, oneCoreUbLoopTimes);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreUbLoopTimes);
TILING_DATA_FIELD_DEF(uint64_t, maxThread);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, sortTmpSize);
TILING_DATA_FIELD_DEF(uint64_t, tailIndexNum);
TILING_DATA_FIELD_DEF(uint64_t, indicesCastMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_4100, UnsortedSegmentSumSortSimtTilingData);

class UnsortedSegmentSumSortSimtTiling : public UnsortedSegmentSumBaseTiling
{
public:
    explicit UnsortedSegmentSumSortSimtTiling(gert::TilingContext* context) : UnsortedSegmentSumBaseTiling(context)
    {}
    ~UnsortedSegmentSumSortSimtTiling() override
    {}
private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    ge::graphStatus CalcTiling();
    int64_t GetSortBufferSize(ge::DataType dataType, int64_t indexSize);
private:
    UnsortedSegmentSumSortSimtTilingData tilingData_;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SUM_SORT_SIMT_TILING_H