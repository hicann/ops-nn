/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_v2_grad_tiling_base.h
 * \brief
 */

#ifndef OP_IMPL_AVG_POOL_V2_GRAD_TILING_BASE_H_
#define OP_IMPL_AVG_POOL_V2_GRAD_TILING_BASE_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "avg_pool_v2_grad_tiling_common.h"

namespace optiling
{
struct AvgPoolV2GradCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

struct AvgPoolV2GradCommon {
    int64_t nDim; 
    int64_t cDim; 
    int64_t hDim; 
    int64_t wDim; 
    std::string padModeStr;
};

ge::graphStatus Tiling4AvgPoolV2Grad(gert::TilingContext* context);

ge::graphStatus TilingPrepare4AvgPoolV2Grad(gert::TilingParseContext* context);

ge::graphStatus GetAvgPoolV2GradPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum);

ge::graphStatus GetAvgPoolV2GradShapeAttrsInfo(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData);

}  // namespace optiling
 
#endif
