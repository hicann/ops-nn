/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_tiling_simd.h
 * \brief SIMD tiling class for Index operator
 */

#ifndef INDEX_TILING_SIMD_H
#define INDEX_TILING_SIMD_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/index_tiling_data.h"
#include "index_tiling.h"

namespace optiling {
using namespace Index;

constexpr int64_t MAX_DIM_NUM = 8;

class IndexTilingSimd : public IndexTilingCommon {
public:
    explicit IndexTilingSimd(gert::TilingContext* context) : IndexTilingCommon(context)
    {}

private:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    // customized functions
    ge::graphStatus CheckShapeInfo();
    ge::graphStatus GetShapeDtypeSize();
    void CalcSimdTiling();
    bool IsIndexContinue();
    bool MargeInputAxis();
    ge::graphStatus MargeOutputAxis();

private:
    gert::Shape inputShapes_;
    gert::Shape maskShape_;
    const gert::Tensor* maskTensor_;
    uint64_t mergeInputShape_[MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t mergeInputIndexed_[MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t mergeInputShapeDim_{0};
    uint64_t mergeOutputShape_[MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t mergeOutToInput_[MAX_DIM_NUM] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int32_t indicesToInput_[MAX_DIM_NUM] = {-1, -1, -1, -1, -1, -1, -1, -1};
    uint32_t mergeOutputShapeDim_{0};
    uint64_t indexSize_{0};
    uint32_t inputDtypeSize_{0};
    uint64_t outputLength_{0};
    uint32_t indexedDimNum_{0};
    int64_t needCoreNum_{0};
    int32_t isZeroOneZero_{0};
    IndexSimdTilingData* simdTilingData_;
};
} // namespace optiling
#endif // INDEX_TILING_SIMD_H
