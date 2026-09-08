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
 * \file unsorted_segment_prod_input_partition_tiling.h
 * \brief unsorted_segment_prod_input_partition_tiling
 */

#ifndef UNSORTED_SEGMENT_PROD_INPUT_PARTITION_TILING_H
#define UNSORTED_SEGMENT_PROD_INPUT_PARTITION_TILING_H

#include "unsorted_segment_prod_tiling.h"

namespace optiling {

class UnsortedSegmentProdInputPartTiling : public UnsortedSegmentBaseTiling {
public:
    explicit UnsortedSegmentProdInputPartTiling(gert::TilingContext* context) : UnsortedSegmentBaseTiling(context) {}
    ~UnsortedSegmentProdInputPartTiling() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    void SetTilingData();

    uint64_t innerAlign_ = 0;
    uint64_t ySize_ = 0;
    uint64_t normRowNum_ = 0;
    uint64_t baseS_ = 1;
    uint64_t partCoreNum_ = 0;
    uint64_t mergeNormNum_ = 0;
    uint64_t mergeChunk_ = 0;
};

} // namespace optiling
#endif // UNSORTED_SEGMENT_PROD_INPUT_PARTITION_TILING_H
