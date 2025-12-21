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
 * \file clipped_swiglu_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_CLIPPED_SWIGLU_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_CLIPPED_SWIGLU_H_

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../op_graph/clipped_swiglu_proto.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
BEGIN_TILING_DATA_DEF(ClippedSwigluTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNumAll);
TILING_DATA_FIELD_DEF(int64_t, dimBatchSize);
TILING_DATA_FIELD_DEF(int64_t, dim2H);
TILING_DATA_FIELD_DEF(int64_t, isLongH);
TILING_DATA_FIELD_DEF(int64_t, isGroup);
TILING_DATA_FIELD_DEF(int64_t, isInterleaved);
TILING_DATA_FIELD_DEF(float, gluAlpha);
TILING_DATA_FIELD_DEF(float, gluLimit);
TILING_DATA_FIELD_DEF(float, gluBias);
TILING_DATA_FIELD_DEF(int64_t, ubMaxPair);
TILING_DATA_FIELD_DEF(int64_t, groupNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ClippedSwiglu, ClippedSwigluTilingData)

struct ClippedSwigluCompileInfo {
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
};

class ClippedSwigluTiling : public TilingBaseClass {
public:
    explicit ClippedSwigluTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext)
    {}
    ~ClippedSwigluTiling() override
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
    void DumpTilingInfo() override;
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CheckAndGetXAndAttrs();
    ge::graphStatus CheckAndGetGroupIndex();
    ge::graphStatus CheckAndGetAttrs();
    ge::graphStatus CheckY();
    ge::graphStatus CountMaxPair();

private:
    uint64_t tilingKey_ = 0;
    ClippedSwigluTilingData tilingData_;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t coreNumAll_ = 0;
    uint64_t ubSize_ = 0;
    int64_t xDims_ = 0;
    int64_t cutDim_ = 0;
    int64_t dimBatchSize_ = 1;
    int64_t dim2H_ = 1;
    int64_t isLongH_ = 0;
    int64_t xCutDimNum_ = 0;
    ge::DataType xDtype_ = ge::DT_FLOAT;
    int64_t isGroup_ = 1;
    int64_t isInterleaved_ = 1;
    float gluLimit_ = 0.0;
    float gluAlpha_ = 0.0;
    float gluBias_ = 0.0;
    int64_t ubMaxPair_ = 0;// UB最多可以处理的元素pair数
    int64_t groupNum_ = 0;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_CLIPPED_SWIGLU_H_
