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
 * \file unsorted_segment_simd_non_sort_tiling.h
 * \brief unsorted_segment_simd_non_sort_tiling
 */

#ifndef UNSORTED_SEGMENT_SIMD_NON_SORT_TILING_H
#define UNSORTED_SEGMENT_SIMD_NON_SORT_TILING_H

#include "unsorted_segment_tiling_common.h"

namespace optiling {

class UnsortedSegmentSimdNonSortTiling : public UnsortedSegmentBaseTiling
{
public:
    explicit UnsortedSegmentSimdNonSortTiling(gert::TilingContext* context) : UnsortedSegmentBaseTiling(context)
    {}
    ~UnsortedSegmentSimdNonSortTiling() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData();

    uint64_t sTileNum_ = 0;
    uint64_t aTileNum_ = 0;
    uint64_t normBlockS_ = 0;
    uint64_t tailBlockS_ = 0;
    uint64_t normBlockA_ = 0;
    uint64_t tailBlockA_ = 0;
    uint64_t baseS_ = 1;
    uint64_t baseA_ = 1;
    uint64_t initUsedCore_ = 0;
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SIMD_NON_SORT_TILING_H
