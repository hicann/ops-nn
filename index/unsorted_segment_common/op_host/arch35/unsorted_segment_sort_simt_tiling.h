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
 * \file unsorted_segment_sort_simt_tiling.h
 * \brief unsorted_segment_sort_simt_tiling
 */
#ifndef UNSORTED_SEGMENT_SORT_SIMT_TILING_H
#define UNSORTED_SEGMENT_SORT_SIMT_TILING_H

#include "unsorted_segment_tiling_common.h"

namespace optiling {

class UnsortedSegmentSortSimtTiling : public UnsortedSegmentBaseTiling
{
public:
    explicit UnsortedSegmentSortSimtTiling(gert::TilingContext* context) : UnsortedSegmentBaseTiling(context)
    {}
    ~UnsortedSegmentSortSimtTiling() override
    {}
protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    ge::graphStatus CalcTiling();
};
} // namespace optiling
#endif // UNSORTED_SEGMENT_SORT_SIMT_TILING_H
