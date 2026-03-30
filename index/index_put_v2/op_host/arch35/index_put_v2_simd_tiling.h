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

#include <array>
#include <cstdint>
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "util/platform_util.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
constexpr size_t MAX_DIM_NUM = 8;

struct IndexPutV2SimdCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

BEGIN_TILING_DATA_DEF(IndexPutV2SimdTilingData)
TILING_DATA_FIELD_DEF(int64_t, inputLength);
TILING_DATA_FIELD_DEF(int64_t, valueLength);
TILING_DATA_FIELD_DEF(int64_t, inputDimNum);
TILING_DATA_FIELD_DEF(int64_t, indexedSizesNum);
TILING_DATA_FIELD_DEF(int64_t, indexedDimNum);
TILING_DATA_FIELD_DEF(int64_t, nonIndexedDimNum);
TILING_DATA_FIELD_DEF(int64_t, accumulateMode);
TILING_DATA_FIELD_DEF(int64_t, indexedLength);
TILING_DATA_FIELD_DEF(int64_t, nonIndexedLength);
TILING_DATA_FIELD_DEF(int64_t, normalCoreRowsNum);
TILING_DATA_FIELD_DEF(int64_t, normalCoreColsNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreRowsNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreColsNum);
TILING_DATA_FIELD_DEF(int64_t, blockNumInRow);
TILING_DATA_FIELD_DEF(int64_t, blockNumInCol);
TILING_DATA_FIELD_DEF(int64_t, rowsFactor);
TILING_DATA_FIELD_DEF(int64_t, colsFactor);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, inputShapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, indexedStrides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexPutV2_3000, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3001, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3002, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3003, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3004, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3005, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3006, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3007, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3100, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3101, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3102, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3103, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3104, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3105, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3106, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_3107, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4001, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4002, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4003, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4004, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4005, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4101, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4102, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4103, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4104, IndexPutV2SimdTilingData);
REGISTER_TILING_DATA_CLASS(IndexPutV2_4105, IndexPutV2SimdTilingData);

class IndexPutV2SimdTiling : public TilingBaseClass {
public:
    explicit IndexPutV2SimdTiling(gert::TilingContext* context)
        : TilingBaseClass(context), opName_(context->GetNodeName())
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus SetTilingData();
    bool CheckInputDtype();
    void DoUBTiling();
    bool IsContinuous();
    void AutoTilingRowCol(int64_t& blockNumInRow, int64_t& blockNumInCol, int64_t usedCoreNum, int64_t rowTotalNum, int64_t colTotalNum);
    void GenIndexSimdTilingKey();

private:
    const std::string opName_ = "IndexPutV2";
    ge::DataType valueType;
    ge::DataType indicesType;
    uint64_t valueTypeSize = 0;
    uint64_t indicesTypeSize = 0;
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t needCoreNum_ = 0;
    uint64_t tmp_buf = 0;
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
    int64_t nonIndexDims_[MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t normalCoreRowsNum_ = 0;
    int64_t normalCoreColsNum_ = 0;
    int64_t tailCoreRowsNum_ = 0;
    int64_t tailCoreColsNum_ = 0;
    int64_t blockNumInCol_ = 0;
    int64_t blockNumInRow_ = 0;
    int64_t colsFactor_ = 0;
    int64_t rowsFactor_ = 0;
    uint32_t tilingkey_ = 0;

    IndexPutV2SimdTilingData tilingData_; 
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_PUT_V2_SIMD_TILING_H