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
 * \file unsorted_segment_prod_output_fullload_ws_merge_tiling.h
 * \brief unsorted_segment_prod_output_fullload_ws_merge_tiling
 */

#ifndef UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_WS_MERGE_TILING_H
#define UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_WS_MERGE_TILING_H

#include "unsorted_segment_prod_tiling.h"

namespace optiling {

class UnsortedSegmentProdOutFlWsMergeTiling : public UnsortedSegmentOutFlTiling {
public:
    explicit UnsortedSegmentProdOutFlWsMergeTiling(gert::TilingContext* context) : UnsortedSegmentOutFlTiling(context)
    {}
    ~UnsortedSegmentProdOutFlWsMergeTiling() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;

    ge::graphStatus UbAddBranchFixedP();
    uint64_t wsStride_ = 0;
};

} // namespace optiling
#endif // UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_WS_MERGE_TILING_H
