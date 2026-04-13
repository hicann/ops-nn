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
 * \file unsorted_segment_sum_simt_tiling.h
 * \brief unsorted_segment_sum_simt_tiling
 */

#ifndef UNSORTED_SEGMENT_SUM_SIMT_TILING_H
#define UNSORTED_SEGMENT_SUM_SIMT_TILING_H

#include "unsorted_segment_sum_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UnsortedSegmentSumSimtTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputOuterDim);  // totalSampleNum_
TILING_DATA_FIELD_DEF(uint64_t, outputOuterDim); // segmentNum_
TILING_DATA_FIELD_DEF(uint64_t, innerDim);
TILING_DATA_FIELD_DEF(uint64_t, maxThread);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnsortedSegmentSum_1000, UnsortedSegmentSumSimtTilingData);

class UnsortedSegmentSumSimtTiling : public UnsortedSegmentSumBaseTiling
{
public:
    explicit UnsortedSegmentSumSimtTiling(gert::TilingContext* context) : UnsortedSegmentSumBaseTiling(context)
    {}
    ~UnsortedSegmentSumSimtTiling() override
    {}

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData();

    UnsortedSegmentSumSimtTilingData tilingData_;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SUM_SIMT_TILING_H
