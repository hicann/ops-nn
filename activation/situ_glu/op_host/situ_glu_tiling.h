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
 * \file situ_glu_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SITU_GLU_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SITU_GLU_H_

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../op_graph/situ_glu_proto.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
BEGIN_TILING_DATA_DEF(SituGluTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNumAll);
TILING_DATA_FIELD_DEF(int64_t, dimBatchSize);
TILING_DATA_FIELD_DEF(int64_t, dim2H);
TILING_DATA_FIELD_DEF(int64_t, isLongH);
TILING_DATA_FIELD_DEF(int64_t, ubMaxPair);
TILING_DATA_FIELD_DEF(float, beta);
TILING_DATA_FIELD_DEF(float, linearBeta);
TILING_DATA_FIELD_DEF(int64_t, activateLeft);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SituGlu, SituGluTilingData)

struct SituGluCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class SituGluTiling : public TilingBaseClass {
public:
    explicit SituGluTiling(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext) {}
    ~SituGluTiling() override {}

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
    ge::graphStatus CheckY();
    ge::graphStatus CountMaxPair();

private:
    uint64_t tilingKey_ = 0;
    SituGluTilingData tilingData_;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t coreNumAll_ = 0;
    uint64_t ubSize_ = 0;
    int64_t xDims_ = 0;
    int64_t cutDim_ = 0;
    int64_t dimBatchSize_ = 1;
    int64_t dim2H_ = 1;
    int64_t cutDimSize_ = 0; // size of x at cutDim (before halving), for CheckY validation
    int64_t isLongH_ = 0;
    ge::DataType xDtype_ = ge::DT_FLOAT;
    float beta_ = 1.0f;
    float linearBeta_ = 0.0f;
    int64_t activateLeft_ = 1;
    int64_t ubMaxPair_ = 0;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SITU_GLU_H_
