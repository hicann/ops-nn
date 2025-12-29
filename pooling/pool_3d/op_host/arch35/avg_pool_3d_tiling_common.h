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
 * \file avg_pool_3d_tiling_common.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_3D_TILING_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_3D_TILING_COMMON_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "pool_3d_tiling_common.h"

namespace optiling
{
 
struct AvgPool3DCommon {
    int64_t nDim;
    int64_t cDim;
    int64_t dDim;
    int64_t hDim;
    int64_t wDim;
};
ge::graphStatus Tiling4AvgPool3DRegBase(gert::TilingContext* context);

ge::graphStatus GetAvgPool3DPlatformInfo(gert::TilingContext *context_, uint64_t& ubSize, uint64_t& coreNum);
ge::graphStatus GetAvgPool3DShapeAttrsInfo(gert::TilingContext *context_, Pool3DInputInfo& inputData);

}  // namespace optiling
 
#endif