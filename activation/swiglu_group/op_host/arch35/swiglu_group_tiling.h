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
 * \file swiglu_group_tiling.h
 * \brief
 */

#ifndef SWIGLU_GROUP_TILING_H
#define SWIGLU_GROUP_TILING_H


#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace optiling {
// ----------算子TilingData定义----------
BEGIN_TILING_DATA_DEF(SwigluGroupTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, splitD);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(int64_t, hasClampLimit);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, gLoop);
TILING_DATA_FIELD_DEF(int64_t, gFactor);
TILING_DATA_FIELD_DEF(int64_t, tailGFactor);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroup, SwigluGroupTilingData)

// ----------算子CompileInfo定义----------
struct SwigluGroupCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ----------算子Tiling入参信息解析及check类----------
class SwigluGroupTiling {
public:
    explicit SwigluGroupTiling(gert::TilingContext* tilingContext) : context_(tilingContext)
    {
    }
    ~SwigluGroupTiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    ge::graphStatus CalcGroupIndexTiling();
    void SetTilingData();
    void SetTilingKey();
private:
    ge::graphStatus GetClampLimitAttr(const gert::RuntimeAttrs* attrs);
    ge::graphStatus CheckWeightInfo();
    ge::graphStatus CheckGroupIndexInfo();
    ge::graphStatus CheckOutputInfo(ge::DataType xDtype);
    void InitCoreTiling();
    void CalcDAndRowFactorTiling(int64_t rowOnceLoop, int64_t dStep);
    void SetFullDTiling();
    void SetSplitDTiling();
    void SetRowLoopTiling();
    int64_t AddWeightSize(int64_t totalSize, int64_t rowFactor) const;
    int64_t CalcTotalSize(int64_t rowFactor, int64_t dFactor) const;

    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    SwigluGroupTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t splitD_ = 0;
    int64_t xElemBytes_ = 2;
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
    float clampLimit_ = 0.0f;
    int64_t hasClampLimit_ = 0;
    bool hasWeight_ = false;
    int64_t g_ = 0;
    int64_t gLoop_ = 0;
    int64_t gFactor_ = 0;
    int64_t tailGFactor_ = 0;
    bool hasGroupIndex_ = false;
};

}  // namespace optiling
#endif  // SWIGLU_GROUP_TILING_H
