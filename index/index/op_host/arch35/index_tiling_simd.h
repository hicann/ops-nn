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
 * \brief ac index tiling h
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_SIMD_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_SIMD_H_
#pragma once

#include "index_tiling_common.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr int64_t MAX_DIM_NUM = 8;

/////////////////////////////////////
// tilingdata define
/////////////////////////////////////
BEGIN_TILING_DATA_DEF(IndexSimdTilingData)
TILING_DATA_FIELD_DEF(uint32_t, indexedDimNum);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM_NUM, mergeInputShape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_DIM_NUM, mergeInputIndexed);
TILING_DATA_FIELD_DEF(uint32_t, mergeInputShapeDim);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM_NUM, mergeOutputShape);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_DIM_NUM, mergeOutToInput);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_DIM_NUM, indicesToInput);
TILING_DATA_FIELD_DEF(uint32_t, mergeOutputShapeDim);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, maxElement);
TILING_DATA_FIELD_DEF(int64_t, indiceUbSize);
TILING_DATA_FIELD_DEF(int64_t, inputDtypeSize);
TILING_DATA_FIELD_DEF(uint64_t, indexSize);
TILING_DATA_FIELD_DEF(int32_t, isZeroOneZero);
END_TILING_DATA_DEF;

// simt template ascendc tools
REGISTER_TILING_DATA_CLASS(Index_3001, IndexSimdTilingData);

class IndexTilingSimd : public IndexTilingCommon {
public:
    explicit IndexTilingSimd(gert::TilingContext* context) : IndexTilingCommon(context)
    {}

private:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    // customized functions
    ge::graphStatus CheckShapeInfo();
    ge::graphStatus GetShapeDtypeSize();
    ge::graphStatus GetParamsShapeInfo();
    bool IsSimd();
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
    uint64_t inputLength_{0};
    uint64_t outputLength_{0};
    uint32_t indexedDimNum_{0};
    int64_t needCoreNum_{0};
    uint32_t tilingKey_{0};
    bool isSimdTemplate_{false};
    int32_t isZeroOneZero_{0};
    IndexSimdTilingData simdTilingData_;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_SIMD_H_
