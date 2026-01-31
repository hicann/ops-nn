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
 * \file unsorted_segment_min_tiling.h
 * \brief unsorted_segment_min_tiling
 */

#ifndef UNSORTED_SEGMENT_MIN_TILING_H
#define UNSORTED_SEGMENT_MIN_TILING_H

#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_tiling_common.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_sort_simt_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simt_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simd_spilt_col_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simd_non_sort_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simd_dyn_sort_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_output_fullload_tiling.h"

namespace optiling {

// ---------------------------UnsortedSegmentMin Simt Tiling---------------------------
class UnsortedSegmentMinSimtTiling : public UnsortedSegmentSimtTiling
{
public:
    explicit UnsortedSegmentMinSimtTiling(gert::TilingContext* context) : UnsortedSegmentSimtTiling(context)
    {}
    ~UnsortedSegmentMinSimtTiling() override
    {}
};

// ---------------------------UnsortedSegmentMin Simt Sort Tiling---------------------------
class UnsortedSegmentMinSortSimtTiling : public UnsortedSegmentSortSimtTiling
{
public:
    explicit UnsortedSegmentMinSortSimtTiling(gert::TilingContext* context) : UnsortedSegmentSortSimtTiling(context)
    {}
    ~UnsortedSegmentMinSortSimtTiling() override
    {}
};

// ---------------------------UnsortedSegmentMin Simd Split Column Tiling---------------------------
class UnsortedSegmentMinSimdSplitColTiling : public UnsortedSegmentSimdSplitColTiling
{
public:
    explicit UnsortedSegmentMinSimdSplitColTiling(gert::TilingContext* context) : UnsortedSegmentSimdSplitColTiling(context)
    {}
    ~UnsortedSegmentMinSimdSplitColTiling() override
    {}
};

// ---------------------------UnsortedSegmentMin Simd Non Sort Tiling---------------------------
class UnsortedSegmentMinSimdNonSortTiling : public UnsortedSegmentSimdNonSortTiling
{
public:
    explicit UnsortedSegmentMinSimdNonSortTiling(gert::TilingContext* context) : UnsortedSegmentSimdNonSortTiling(context)
    {}
    ~UnsortedSegmentMinSimdNonSortTiling() override
    {}
};

// ---------------------------UnsortedSegmentMin Simd Dynamic Sort Tiling---------------------------
class UnsortedSegmentMinSimdDynSortTiling : public UnsortedSegmentSimdDynSortTiling
{
public:
    explicit UnsortedSegmentMinSimdDynSortTiling(gert::TilingContext* context) : UnsortedSegmentSimdDynSortTiling(context)
    {}
    ~UnsortedSegmentMinSimdDynSortTiling() override
    {}
};

// ---------------------------UnsortedSegmentMin Out Full Load Tiling---------------------------
class UnsortedSegmentMinOutFlTiling : public UnsortedSegmentOutFlTiling
{
public:
    explicit UnsortedSegmentMinOutFlTiling(gert::TilingContext* context) : UnsortedSegmentOutFlTiling(context)
    {}
    ~UnsortedSegmentMinOutFlTiling() override
    {}
};

} // namespace optiling
#endif // UNSORTED_SEGMENT_SORT_SIMT_TILING_H
