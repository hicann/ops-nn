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
 * \file swiglu_group_quant_hifp8_tiling.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_HIFP8_TILING_H
#define SWIGLU_GROUP_QUANT_HIFP8_TILING_H

#include "swiglu_group_quant_tiling.h"

namespace optiling {

class SwigluGroupQuantHifp8Tiling {
public:
    explicit SwigluGroupQuantHifp8Tiling(gert::TilingContext* tilingContext) : context_(tilingContext)
    {
    }
    ~SwigluGroupQuantHifp8Tiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus CheckAllInputs();
    ge::graphStatus ProcessTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    void SetTilingData();
    void SetTilingKey();
private:
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckOutputDtype();
    ge::graphStatus CheckWeightInfo();
    ge::graphStatus CheckGroupIndexInfo();
    ge::graphStatus CheckOutputInfo();
    ge::graphStatus CheckYShape(size_t xDimNum, const gert::Shape& xStorageShape);
    ge::graphStatus CheckYScaleShape();
    ge::graphStatus CheckYOriginShape(size_t xDimNum, const gert::Shape& xStorageShape);
    void CalcTileTokens();
    ge::graphStatus CalcCoreDistribution();

    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    SwigluGroupQuantHifp8TilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    int64_t totalTokens_ = 0;
    int64_t dim2H_ = 0;
    int64_t dimH_ = 0;
    bool hasWeight_ = false;
    bool isGroup_ = false;
    int64_t groupNum_ = 0;
    bool hasClamp_ = false;
    bool outputOrigin_ = false;
    float clampLimit_ = -1.0f;
    float dstTypeMax_ = 15.0f;
    int64_t usedCoreNum_ = 1;
    int64_t tokensPerCore_ = 0;
    int64_t tileTokens_ = 0;
    int64_t tileLength_ = 0;
};

}  // namespace optiling
#endif  // SWIGLU_GROUP_QUANT_HIFP8_TILING_H
