/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_TILING_H
#define SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_TILING_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/swiglu_backward_group_quant_with_dual_axis_tilingdata.h"

namespace optiling {

struct CoreCompileInfo {};

struct SwigluBackwardGroupQuantWithDualAxisMxTilingParam {
    int64_t totalCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t ubSize = 0;
    int64_t totalRows = 0;
    int64_t dimBatch = 1;
    int64_t dimM = 0;
    int64_t dimN = 0;
    int64_t gradWeightTileH = 0;
    int64_t gradWeightTileTokens = 1;
    int64_t numGroups = 1;
    int64_t hasGroupIndex = 0;
    int64_t hasWeight = 0;
    int64_t weightDtype = 1;
    int64_t hasClampLimit = 0;
    int64_t quantMode = 1;
    int64_t dstType = 36;
    int64_t mode = 1;
    float alpha = 1.0f;
    float clampLimit = -1.0f;
    float bias = 0.0f;
};

class SwigluBackwardGroupQuantWithDualAxisMxTiling {
public:
    explicit SwigluBackwardGroupQuantWithDualAxisMxTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus ParseAttrs();
    ge::graphStatus CheckDtypes();
    ge::graphStatus CheckShapes();
    ge::graphStatus ComputeTiling();
    ge::graphStatus SaveTiling();
    void SetTilingKey();

    gert::TilingContext* context_ = nullptr;
    SwigluBackwardGroupQuantWithDualAxisMxTilingParam param_;
};

ge::graphStatus TilingForSwigluBackwardGroupQuantWithDualAxisMx(gert::TilingContext* context);

} // namespace optiling
#endif
