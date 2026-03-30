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
 * \file max_pool_with_argmax_v3_simt_tiling.cpp
 * \brief
 */
#include <cctype>
#include <algorithm>
#include "platform/platform_ascendc.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "avg_pool_grad_simt_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

ge::graphStatus AvgPoolGradTilingSIMT::GetPlatformInfo() {
    return GetAvgPoolGradPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolGradTilingSIMT::GetShapeAttrsInfo() {
    return GetAvgPoolGradShapeAttrsInfo(context_, inputData);
}

REGISTER_OPS_TILING_TEMPLATE(AvgPoolGrad, AvgPoolGradTilingSIMT, 100);
} // namespace optiling