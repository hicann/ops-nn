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
 * \file index_elements_no_contiguous_tiling.h
 * \brief
 */
#ifndef INDEX_ELEMENTS_NO_CONTIGUOUS_TILING_H
#define INDEX_ELEMENTS_NO_CONTIGUOUS_TILING_H

#include <iostream>
#include <string>
#include <vector>
#include "index_tiling_common.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr int64_t ARRAY_LEN_FOUR = 4;
BEGIN_TILING_DATA_DEF(IndexNonContinuousTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputLength);
TILING_DATA_FIELD_DEF(uint32_t, inputDimNum);
TILING_DATA_FIELD_DEF(uint64_t, outputLength);
TILING_DATA_FIELD_DEF(int64_t, indexSize);
TILING_DATA_FIELD_DEF(int64_t, indexedDimNum);
TILING_DATA_FIELD_DEF(uint32_t, indexedSizesNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, xShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, indexShape);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, xStride);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, indexStride1);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, indexStride2);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, indexStride3);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, indexStride4);
TILING_DATA_FIELD_DEF_ARR(int64_t, ARRAY_LEN_FOUR, yStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Index_20001, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20002, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20004, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20008, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20016, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20101, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20102, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20104, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20108, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_20116, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30001, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30002, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30004, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30008, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30016, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30101, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30102, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30104, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30108, IndexNonContinuousTilingData);
REGISTER_TILING_DATA_CLASS(Index_30116, IndexNonContinuousTilingData);

class IndexNonContinuousTiling : public IndexTilingCommon {
public:
    explicit IndexNonContinuousTiling(gert::TilingContext* context) : IndexTilingCommon(context) {
    }
protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    struct TensorMeta {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    };

private:
    ge::graphStatus GetContinuousTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut);
    ge::graphStatus GetTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut=false); 
    void GetIndexStrideInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, int64_t i);
    void GenNonContinuousTilingKey();
    void GetContinuousStrideInfo(gert::Shape &shape, gert::Stride &stride);
    uint64_t GetDataTypeInByte(gert::TilingContext *context); 
    bool IsContinuous(const gert::Shape &xShape, const gert::Stride &xStride);
    bool ParamTypeIsInvalid(ge::DataType &x);
    bool isDimCanKeep(int64_t dim_idx, const TensorMeta& tensor);
    bool IsAllIndexStrideEqual();
    void InitVector(std::vector<int64_t> &tempIndexShape, std::vector<int64_t> &tempIndexStride);
    void mergeIndexAxis(std::vector<int64_t> &tempIndexShape, std::vector<int64_t> &tempIndexStride);
    void UpdateResultFromVector(std::vector<int64_t> &tempIndexShape, std::vector<int64_t> &tempIndexStride);
    void CoalesceIndex();
    void SetTilingData();
    void PrintTilingData();

private:
    bool isNonContinuous_ = false;
    bool isCoalesced_ = false;
    uint64_t inputLength_ = 0;
    uint64_t outputLength_= 0;
    int64_t tensorNum_ = 0;
    int64_t indexedDimNum_ = 0;
    int64_t indexedSizesNum_= 0;
    uint32_t inputDimNum_= 0;
    uint32_t tilingKey_= 0;   

    gert::Shape xShape_;
    gert::Shape indexShape_;
    gert::Shape yShape_;

    gert::Stride xStride_;
    gert::Stride indexStride1_;
    gert::Stride indexStride2_;
    gert::Stride indexStride3_;
    gert::Stride indexStride4_;
    gert::Stride yStride_;

    ge::DataType xDtype_;
    ge::DataType indexDtype_;
    
    const char *opName_ = "Index";
    IndexNonContinuousTilingData m_tilingData_; 
    std::vector<gert::Stride> indexstrideList = {};
};

} // namespace optiling
#endif // Index_NO_CONTIGUOUS_TILING_H