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
 * \file max_pool_3d_tiling_common.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_TILING_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_TILING_COMMON_H_

#include <array>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "pool_3d_tiling_common.h"

namespace optiling
{
struct MaxPool3DCommon {
    int64_t nDim;
    int64_t cDim;
    int64_t dDim;
    int64_t hDim;
    int64_t wDim;
    std::string padModeStr;
};

struct MaxPool3DCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

ge::graphStatus GetMaxPool3DPlatformInfo(gert::TilingContext *context_, uint64_t& ubSize, uint64_t& coreNum);
ge::graphStatus GetMaxPool3DShapeAttrsInfo(gert::TilingContext *context_, Pool3DInputInfo& inputData);

BEGIN_TILING_DATA_DEF(MaxPool3DTilingData)
TILING_DATA_FIELD_DEF(int64_t, dInDim);
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, nOutDim);
TILING_DATA_FIELD_DEF(int64_t, dOutDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kD);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, sD);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, dD);
TILING_DATA_FIELD_DEF(int64_t, dH);
TILING_DATA_FIELD_DEF(int64_t, dW);
TILING_DATA_FIELD_DEF(int64_t, frontPad);
TILING_DATA_FIELD_DEF(int64_t, backendPad);
TILING_DATA_FIELD_DEF(int64_t, topPad);
TILING_DATA_FIELD_DEF(int64_t, bottomPad);
TILING_DATA_FIELD_DEF(int64_t, leftPad);
TILING_DATA_FIELD_DEF(int64_t, rightPad);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3D, MaxPool3DTilingData);


}  // namespace optiling

#endif