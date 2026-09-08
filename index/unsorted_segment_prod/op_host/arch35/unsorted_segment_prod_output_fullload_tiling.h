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
 * \file unsorted_segment_prod_output_fullload_tiling.h
 * \brief unsorted_segment_prod_output_fullload_tiling
 */

#ifndef UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_TILING_H
#define UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_TILING_H

#include "unsorted_segment_prod_tiling.h"

namespace optiling {

class UnsortedSegmentProdOutFlTiling : public UnsortedSegmentOutFlTiling {
public:
    explicit UnsortedSegmentProdOutFlTiling(gert::TilingContext* context) : UnsortedSegmentOutFlTiling(context) {}
    ~UnsortedSegmentProdOutFlTiling() override {}

protected:
    bool IsCapable() override;
};

} // namespace optiling
#endif // UNSORTED_SEGMENT_PROD_OUTPUT_FULLLOAD_TILING_H
