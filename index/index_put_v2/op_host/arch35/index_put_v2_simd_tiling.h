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
 * \file index_put_v2_simd_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_V2_SIMD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_V2_SIMD_TILING_H

#include <cstddef>
#include <cstdint>
#include "../../../index/op_host/arch35/index_tiling.h"
#include "../../op_kernel/arch35/index_put_v2_tiling_data.h"

namespace optiling {
constexpr size_t MAX_DIM_NUM = 8;

class IndexPutV2SimdTiling : public IndexTilingCommon {
public:
    explicit IndexPutV2SimdTiling(gert::TilingContext* context) : IndexTilingCommon(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus SetTilingData();
    bool CheckInputDtype();
    void DoUBTiling();
    bool IsContinuous();
    void AutoTilingRowCol(int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum, int64_t rowTotalNum,
                          int64_t colTotalNum);

private:
    uint64_t valueTypeSize = 0;
    uint64_t indicesTypeSize = 0;
    uint64_t needCoreNum_ = 0;
    int64_t inputLength_ = 0;
    int64_t valueLength_ = 0;
    int64_t inputDimNum_ = 0;
    int64_t indexedDimNum_ = 0;
    int64_t nonIndexedDimNum_ = 0;
    int64_t indexedSizesNum_ = 0;
    int64_t accumulateMode_ = 0;
    int64_t indexedLength_ = 1;
    int64_t nonIndexedLength_ = 1;
    int64_t inputShapes_[MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexedStrides_[MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexedSizes_[MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t normalCoreRowsNum_ = 0;
    int64_t normalCoreColsNum_ = 0;
    int64_t tailCoreRowsNum_ = 0;
    int64_t tailCoreColsNum_ = 0;
    int64_t blockNumInCol_ = 0;
    int64_t blockNumInRow_ = 0;
    int64_t colsFactor_ = 0;
    int64_t rowsFactor_ = 0;

    IndexPutV2::IndexPutV2SimdTilingData* tilingData_;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_V2_SIMD_TILING_H
