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
 * \file index_put_with_sort_v2_tiling_arch35.h
 * \brief IndexPutWithSortV2 tiling file
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_WITH_SORT_V2_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_WITH_SORT_V2_ARCH35_H

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"

namespace optiling {
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t CACHELINE_SIZE = 128;
constexpr int64_t B8_SIZE = 8;
constexpr int64_t B4_SIZE = 4;
constexpr int64_t B2_SIZE = 2;
constexpr int64_t B1_SIZE = 1;

class IndexPutWithSortV2Tiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit IndexPutWithSortV2Tiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::DataType xDataType_{ge::DT_INT32};
    int64_t indicesTypeSize_{0};
    int64_t indexed0_ = {0};

    uint32_t selfDimNum_{1};
    uint32_t indexedDimNum_{0};
    uint32_t nonIndexedDimNum_{0};
    int64_t indexedDimSize_{1};
    int64_t nonIndexedDimSize_{1};

    int64_t selfDims_[MAX_DIM_NUM] = {0};
    int64_t indexedSizes_[MAX_DIM_NUM] = {0};  // 看需不需要类成员
    int64_t nonIdxedDims_[MAX_DIM_NUM] = {0};

    int64_t nonIdxedStride_[MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t nonIdxedSelfStride_[MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t nonIdxedValueStride_[MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    bool isContinous_{false};
    bool accumulate_{false};
    bool indexedBlockMode_{false};
    int64_t idxedValueStride_{0};
    int64_t indexedThreadNum_{0};
    int64_t nonIndexedThreadNum_{0};
    int64_t aivCoreNum_{0};
    uint64_t maxUbSize_{0};
    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckInputsDtypeAndFormat();
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    bool IsIndexedContinous(const int64_t* arr, int64_t size);
    void CalcSelfAndValueStride(int64_t* selfStride, int64_t* valueStride);
    void CalcNonIndexedStride(int64_t* selfStride, int64_t* valueStride);
    void CalcThreadNum();
    void SetTilingData();
    void LogTilingResult();
};
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_WITH_SORT_V2_H
