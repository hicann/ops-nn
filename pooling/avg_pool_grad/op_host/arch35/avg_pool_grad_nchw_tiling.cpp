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
 * \file avg_pool_grad_nchw_tiling.cpp
 * \brief
 */

#include "avg_pool_grad_nchw_tiling.h"

namespace optiling {

//////////////////////////////// AvgPoolGradNCHWTiling /////////////////////////////////
ge::graphStatus AvgPoolGradNCHWTiling::GetPlatformInfo() {
    return GetAvgPoolGradPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolGradNCHWTiling::GetShapeAttrsInfo() {
    return GetAvgPoolGradShapeAttrsInfo(context_, inputData);
}

REGISTER_TILING_TEMPLATE("AvgPoolGrad", AvgPoolGradNCHWTiling, 0);

} // namespace optiling
