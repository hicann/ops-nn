/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_tiling_base.h
 * \brief SwigluGroupGrad shared tiling utilities (arch35, Ascend950)
 */

#ifndef SWIGLU_GROUP_GRAD_TILING_BASE_H_
#define SWIGLU_GROUP_GRAD_TILING_BASE_H_

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/swiglu_group_grad_tiling_key.h"

namespace optiling {

struct SwigluGroupGradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct SwigluGroupGradInputInfo {
    ge::DataType gradYDtype = ge::DT_UNDEFINED;
    int64_t totalRows = 1;
    int64_t dim2H = 1;
    int64_t H = 1;
    int64_t isWeight = 0;
    int64_t isYOrigin = 0;
    int64_t isGroupIndex = 0;
    int64_t hasClamp = 0;
    float clampLimit = 0.0f;
    int64_t groupIndexG = 0;
};

constexpr int64_t GRAD_Y_INDEX = 0;
constexpr int64_t X_INDEX = 1;
constexpr int64_t WEIGHT_INDEX = 2;
constexpr int64_t YORIGIN_INDEX = 3;
constexpr int64_t GROUP_INDEX_INDEX = 4;
constexpr int64_t CLAMPLIMIT_ATTR_INDEX = 0;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;

constexpr int64_t VL_FP32_ARCH35 = 64;

ge::graphStatus Tiling4SwigluGroupGradArch35(gert::TilingContext* context);
ge::graphStatus TilingPrepare4SwigluGroupGradArch35(gert::TilingParseContext* context);

ge::graphStatus GetSwigluGroupGradPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum);
ge::graphStatus GetSwigluGroupGradShapeAttrsInfo(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData);

} // namespace optiling

#endif // SWIGLU_GROUP_GRAD_TILING_BASE_H_
