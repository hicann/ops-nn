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
 * \file swiglu_group_quant_tiling.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_TILING_H
#define SWIGLU_GROUP_QUANT_TILING_H

#include "swiglu_group_quant_tiling_base.h"

namespace optiling {

class SwigluGroupQuantTiling {
public:
    explicit SwigluGroupQuantTiling(gert::TilingContext* tilingContext) : context_(tilingContext)
    {
    }
    ~SwigluGroupQuantTiling() = default;

    static ge::graphStatus GetPlatformInfoCommon(gert::TilingContext* context,
                                                  uint64_t& coreNum, uint64_t& ubSize);
    
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    ge::graphStatus CalcMxQuantOpTiling();
    ge::graphStatus CalcMxFp4QuantOpTiling();
    ge::graphStatus CalcBlockQuantOpTiling();
    ge::graphStatus CalcGroupIndexTiling();
    void SetTilingData();
    void SetTilingKey();
private:
    using TotalSizeFunc = int64_t (SwigluGroupQuantTiling::*)(int64_t, int64_t) const;

    ge::graphStatus GetClampLimitAttr(const gert::RuntimeAttrs* attrs);
    ge::graphStatus CheckWeightInfo();
    ge::graphStatus CheckGroupIndexInfo();
    ge::graphStatus CheckOutputInfo(ge::DataType xDtype);
    void InitCoreTiling();
    void CalcDAndRowFactorTiling(int64_t rowOnceLoop, int64_t dStep, TotalSizeFunc calcTotalSize);
    void SetFullDTiling();
    void SetSplitDTiling();
    void SetRowLoopTiling();
    int64_t AddWeightSize(int64_t totalSize, int64_t rowFactor) const;
    int64_t AddYOriginSize(int64_t totalSize, int64_t rowFactor, int64_t dFactor) const;
    int64_t CalcMxQuantTotalSize(int64_t rowFactor, int64_t dFactor) const;
    int64_t CalcBlockQuantTotalSize(int64_t rowFactor, int64_t dFactor) const;
    int64_t CalcMxFp4QuantTotalSize(int64_t rowFactor, int64_t dFactor) const;

    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    SwigluGroupQuantTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t splitD_ = 0;
    int64_t scaleCol_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_= 0;
    int64_t dLoop_ = 0;
    int64_t dFactor_ = 0;
    int64_t tailDFactor_ = 0;
    int64_t quantMode_ = 0;
    int64_t splitFactor_ = 0;
    ge::DataType dstType_ = ge::DT_FLOAT8_E4M3FN;
    int64_t roundScale_ = 0;
    double clampLimit_ = 0.0;
    int64_t hasClampLimit_ = 0;
    int64_t outputOrigin_ = 0;
    bool isMxFp4Quant_ = false;
    bool hasWeight_ = false;
    int64_t g_ = 0;
    int64_t gLoop_ = 0;
    int64_t gFactor_ = 0;
    int64_t tailGFactor_ = 0;
    bool hasGroupIndex_ = false;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

}  // namespace optiling
#endif  // SWIGLU_GROUP_QUANT_TILING_H
