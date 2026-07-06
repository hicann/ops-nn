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
 * \file index_tiling_no_continuous.h
 * \brief Non-continuous tiling class for Index/IndexPutV2 operators
 */
#ifndef INDEX_TILING_NO_CONTINUOUS_H
#define INDEX_TILING_NO_CONTINUOUS_H

#include <string>
#include <vector>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/index_tiling_data.h"
#include "index_tiling.h"

namespace optiling {
using namespace Index;

constexpr int64_t ARRAY_LEN_FOUR = 4;
constexpr int64_t ARRAY_LEN_EIGHT = 8;
class IndexNonContinuousTiling : public IndexTilingCommon {
public:
    explicit IndexNonContinuousTiling(gert::TilingContext* context) : IndexTilingCommon(context) {}
    bool isIndexPut_{false};

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    struct TensorMeta {
        std::vector<int64_t> shape;
        std::vector<int64_t> stride;
    };

private:
    ge::graphStatus GetContinuousTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx, bool isOut);
    ge::graphStatus GetTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx, bool isOut = false);
    void GetIndexStrideInfo(gert::Shape& shape, gert::Stride& stride, size_t idx, int64_t i);
    void GetContinuousStrideInfo(gert::Shape& shape, gert::Stride& stride);
    bool IsContinuous(const gert::Shape& xShape, const gert::Stride& xStride);
    bool ParamTypeIsInvalid(ge::DataType& x);
    bool isDimCanKeep(int64_t dim_idx, const TensorMeta& tensor);
    bool IsAllIndexStrideEqual();
    void InitVector(std::vector<int64_t>& tempIndexShape, std::vector<int64_t>& tempIndexStride);
    void mergeIndexAxis(std::vector<int64_t>& tempIndexShape, std::vector<int64_t>& tempIndexStride);
    void UpdateResultFromVector(std::vector<int64_t>& tempIndexShape, std::vector<int64_t>& tempIndexStride);
    void CoalesceIndex();
    void SetTilingData();
    void PrintTilingData();

private:
    bool isCoalesced_ = false;
    uint64_t inputLength_ = 0;
    uint64_t outputLength_ = 0;
    int64_t tensorNum_ = 0;
    int64_t indexedDimNum_ = 0;
    int64_t indexedSizesNum_ = 0;
    int64_t valueDimNum_ = 0;
    uint32_t inputDimNum_ = 0;
    int64_t paramIndexedSizesIdx_ = 0;
    int64_t paramIndicesIdx_ = 0;
    uint32_t isAccumulate = 0;
    gert::Shape xShape_;
    gert::Shape indexShape_;
    gert::Shape valueShape_;
    gert::Shape yShape_;

    gert::Stride xStride_;
    gert::Stride indexStride1_;
    gert::Stride indexStride2_;
    gert::Stride indexStride3_;
    gert::Stride indexStride4_;
    gert::Stride valueStride_;
    gert::Stride yStride_;
    ge::DataType xDtype_ = ge::DT_FLOAT;
    std::vector<gert::Stride> indexstrideList = {};
    IndexNonContinuousTilingData* m_tilingData_;
};
} // namespace optiling
#endif // INDEX_TILING_NO_CONTINUOUS_H
