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
 * \file avg_pool_tiling_common.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_TILING_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_TILING_COMMON_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "avg_pool_common.h"

namespace optiling
{
using TilingBaseClass = Ops::NN::Optiling::TilingBaseClass;
 
struct AvgPoolCommon {
    int64_t nDim; 
    int64_t cDim; 
    int64_t hDim; 
    int64_t wDim; 
    std::string padModeStr;
};

ge::graphStatus Tiling4AvgPoolRegBase(gert::TilingContext* context);

ge::graphStatus GetAvgPoolPlatformInfo(gert::TilingContext *context, uint64_t& ubSize, uint64_t& coreNum);

ge::graphStatus GetAvgPoolShapeAttrsInfo(gert::TilingContext *context, AvgPoolInputInfo& inputData);

}  // namespace optiling
 
#endif