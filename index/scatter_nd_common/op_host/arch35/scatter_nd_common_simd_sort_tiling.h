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
 * \file scatter_nd_common_simd_sort_tiling.h
 * \brief scatter_nd_common_simd_sort_tiling
 */

#ifndef SCATTER_ND_COMMON_SIMD_SORT_TILING_H
#define SCATTER_ND_COMMON_SIMD_SORT_TILING_H

#include "scatter_nd_common_base_tiling.h"

namespace optiling {

class ScatterNdCommonSimdSortTiling : public ScatterNdCommonBaseTiling {
public:
    explicit ScatterNdCommonSimdSortTiling(gert::TilingContext* context) : ScatterNdCommonBaseTiling(context) {}
    ~ScatterNdCommonSimdSortTiling() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    std::string TilingDataToString();
    void SetTilingData();
    void DoOpTilingForSimdSort();
    void DoOpTilingSplitIndicesSingleCol();
    void DoOpTilingSplitAfter();
    void DoOpTilingSimdSplitIndices();

    int64_t eachCoreIndexCount_ = 0;
    int64_t usedCoreNumBefore_ = 0;
    int64_t tailCoreIndexCount_ = 0;
    int64_t indicesFactor_ = 0;
    int64_t afterAxisFactor_ = 0;
    int64_t updateLoopSize_ = 0;
    int64_t updateTailNum_ = 0;
    int64_t singleCol_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t tailCoreAfterAxisCount_ = 0;
    int64_t indicesLoopSize_ = 0;
    int64_t indiceTailNum_ = 0;
    int64_t tailUpdateLoopSize_ = 0;
    int64_t tailUpdateTailNum_ = 0;
    int64_t isSplitAfterAxis_ = 0;
};
} // namespace optiling
#endif // SCATTER_ND_COMMON_SIMD_SORT_TILING_H
