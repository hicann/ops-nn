/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_elements_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_V2_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_V2_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling
{
using Ops::NN::Optiling::TilingBaseClass;
constexpr int16_t TILING_ARRAY_LEN = 7;

// ///////////////////////////////////
// tilingdata define
// ///////////////////////////////////
BEGIN_TILING_DATA_DEF(ScatterElementsV2AscTilingData)
TILING_DATA_FIELD_DEF_ARR(uint64_t, TILING_ARRAY_LEN, dataStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TILING_ARRAY_LEN, indicesStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, TILING_ARRAY_LEN, updatesStride);
TILING_DATA_FIELD_DEF(int64_t, loopLength);
TILING_DATA_FIELD_DEF(int64_t, allAxis);
TILING_DATA_FIELD_DEF(int64_t, dataAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesAxis);
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, midAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, indicesUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, indicesNormBlockData);
TILING_DATA_FIELD_DEF(int64_t, indicesTailBlockData);
TILING_DATA_FIELD_DEF(int64_t, baseS);
TILING_DATA_FIELD_DEF(int64_t, baseA);
TILING_DATA_FIELD_DEF(int64_t, isDeterministic);
TILING_DATA_FIELD_DEF(int16_t, rank);
TILING_DATA_FIELD_DEF(int16_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, sortSharedBufSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000001, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000002, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000004, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000008, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000100, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000101, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000102, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000103, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000104, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000106, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000127, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000109, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000112, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000200, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000201, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000202, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000203, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000204, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000206, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000227, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1000209, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001001, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001002, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001004, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001008, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001100, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001101, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001102, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001103, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001104, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001106, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001127, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001109, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001112, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001200, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001201, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001202, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001203, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001204, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001206, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001227, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1001209, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010001, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010002, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010004, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010008, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010100, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010101, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010102, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010103, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010104, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010106, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010127, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010109, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010112, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010200, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010201, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010202, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010203, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010204, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010206, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010227, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1010209, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011001, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011002, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011004, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011008, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011100, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011101, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011102, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011103, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011104, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011106, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011127, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011109, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011112, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011200, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011201, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011202, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011203, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011204, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011206, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011227, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElementsV2_1011209, ScatterElementsV2AscTilingData)

class ScatterElementsV2AscTiling : public TilingBaseClass
{
public:
    explicit ScatterElementsV2AscTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckXDtype(const ge::DataType dtype);
    bool CompareShape(const gert::Shape& shape1, const gert::Shape& shape2, int16_t dim = -1);
    void ComputeShape(const gert::Shape& dataShape, const gert::Shape& indicesShape, const gert::Shape& updatesShape);
    uint64_t GetStride(const std::vector<uint64_t>& shapeList, int16_t start);
    void ComputeStride();
    void GetCastTypeSize();
    void CombineIndicesAxis();
    uint32_t GetMaxSortTmpBuf(int64_t sortDim);
    int64_t CalBestBaseSize(int64_t baseXoStart, int64_t baseXoEnd);

private:
    int16_t dim_ = 0;
    int16_t rank_ = 1;
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t loopLength_ = 0;
    int64_t allAxis_ = 1;
    int64_t dataAxis_ = 1;
    int64_t updatesAxis_ = 1;
    int64_t castTypeSize_ = 0;
    int64_t isDeterministic_ = 0;
    int64_t preAxis_ = 1;
    int64_t midAxis_ = 1;
    int64_t afterAxis_ = 1;
    int64_t indicesUsedCoreNum_ = 0;
    int64_t indicesNormBlockData_ = 0;
    int64_t indicesTailBlockData_ = 0;
    int64_t baseS_ = 1;
    int64_t baseA_ = 1;
    int64_t indicesTypeSize_ = 0;
    int64_t ubBlockSize_ = 0;
    uint64_t reduction_ = 0;
    uint64_t typeSize_ = 0;
    uint64_t tilingKey_ = 0;
    uint32_t sortSharedBufSize_ = 0;
    std::vector<uint64_t> dataCurSize_;
    std::vector<uint64_t> indicesCurSize_;
    std::vector<uint64_t> updatesCurSize_;
    uint64_t dataStride_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1};
    uint64_t indicesStride_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1};
    uint64_t updatesStride_[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1};

    ge::DataType dtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    ScatterElementsV2AscTilingData tilingData_;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_ELEMENTS_V2_TILING_H_