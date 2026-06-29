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
 * \file unsorted_segment_max_tiling.h
 * \brief unsorted_segment_max_tiling
 */

#ifndef UNSORTED_SEGMENT_MAX_TILING_H
#define UNSORTED_SEGMENT_MAX_TILING_H

#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_tiling_common.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_sort_simt_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simt_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simd_spilt_col_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simd_non_sort_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_simd_dyn_sort_tiling.h"
#include "index/unsorted_segment_common/op_host/arch35/unsorted_segment_output_fullload_tiling.h"

namespace optiling {

// ---------------------------UnsortedSegmentMax Simt Tiling---------------------------
class UnsortedSegmentMaxSimtTiling : public UnsortedSegmentSimtTiling
{
public:
    explicit UnsortedSegmentMaxSimtTiling(gert::TilingContext* context) : UnsortedSegmentSimtTiling(context)
    {}
    ~UnsortedSegmentMaxSimtTiling() override
    {}
};

// ---------------------------UnsortedSegmentMax Simt Sort Tiling---------------------------
class UnsortedSegmentMaxSortSimtTiling : public UnsortedSegmentSortSimtTiling
{
public:
    explicit UnsortedSegmentMaxSortSimtTiling(gert::TilingContext* context) : UnsortedSegmentSortSimtTiling(context)
    {}
    ~UnsortedSegmentMaxSortSimtTiling() override
    {}
};

// ---------------------------UnsortedSegmentMax Simd Split Column Tiling---------------------------
class UnsortedSegmentMaxSimdSplitColTiling : public UnsortedSegmentSimdSplitColTiling
{
public:
    explicit UnsortedSegmentMaxSimdSplitColTiling(gert::TilingContext* context) : UnsortedSegmentSimdSplitColTiling(context)
    {}
    ~UnsortedSegmentMaxSimdSplitColTiling() override
    {}
};

// ---------------------------UnsortedSegmentMax Simd Non Sort Tiling---------------------------
class UnsortedSegmentMaxSimdNonSortTiling : public UnsortedSegmentSimdNonSortTiling
{
public:
    explicit UnsortedSegmentMaxSimdNonSortTiling(gert::TilingContext* context) : UnsortedSegmentSimdNonSortTiling(context)
    {}
    ~UnsortedSegmentMaxSimdNonSortTiling() override
    {}
};

// ---------------------------UnsortedSegmentMax Simd Dynamic Sort Tiling---------------------------
class UnsortedSegmentMaxSimdDynSortTiling : public UnsortedSegmentSimdDynSortTiling
{
public:
    explicit UnsortedSegmentMaxSimdDynSortTiling(gert::TilingContext* context) : UnsortedSegmentSimdDynSortTiling(context)
    {}
    ~UnsortedSegmentMaxSimdDynSortTiling() override
    {}
};

// ---------------------------UnsortedSegmentMax Out Full Load Tiling---------------------------
class UnsortedSegmentMaxOutFlTiling : public UnsortedSegmentOutFlTiling
{
public:
    explicit UnsortedSegmentMaxOutFlTiling(gert::TilingContext* context) : UnsortedSegmentOutFlTiling(context)
    {}
    ~UnsortedSegmentMaxOutFlTiling() override
    {}
};

} // namespace optiling
#endif // UNSORTED_SEGMENT_MAX_TILING_H
