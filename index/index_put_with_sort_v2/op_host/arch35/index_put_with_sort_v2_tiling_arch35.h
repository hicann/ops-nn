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

#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr size_t MAX_DIM_NUM = 8;

BEGIN_TILING_DATA_DEF(IndexPutWithSortV2TilingData)
  TILING_DATA_FIELD_DEF(int64_t, nonIndexedDimNum);
  TILING_DATA_FIELD_DEF(int64_t, indexedDimSize);
  TILING_DATA_FIELD_DEF(int64_t, nonIndexedDimSize);
  TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, nonIdxedStride);
  TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, nonIdxedSelfStride);
  TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, nonIdxedValueStride);
  TILING_DATA_FIELD_DEF(int64_t, idxedValueStride);
  TILING_DATA_FIELD_DEF(int64_t, indexedThreadNum);
  TILING_DATA_FIELD_DEF(int64_t, nonIndexedThreadNum);
  

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexPutWithSortV2, IndexPutWithSortV2TilingData);

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

private:
    IndexPutWithSortV2TilingData tilingData_;
    uint32_t selfDimNum_;
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
    bool isContinous_;
    bool accumulate_;
    bool indexedBlockMode_;
    int64_t idxedValueStride_;
    int64_t indexedThreadNum_;
    int64_t nonIndexedThreadNum_;
    int64_t aivCoreNum_;
    uint64_t maxUbSize_;
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
