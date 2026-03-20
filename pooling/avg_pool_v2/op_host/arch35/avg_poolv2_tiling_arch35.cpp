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
 * \file avg_poolv2_tiling_arch35.cpp
 * \brief tiling function of avg_pool_v2
 */
#include "register/op_impl_registry.h"
#include "avg_pool_v2_common_tiling.h"

using namespace ge;

struct AvgPoolV2TilingParseInfo {};

namespace gert {

ge::graphStatus TilingPrepareForAvgPoolV2(TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAvgPoolV2(TilingContext* context)
{
    return optiling::Tiling4AvgPoolV2RegBase(context);
}

// register op tiling interface of AvgPoolV2 (runtime2.0)
IMPL_OP_OPTILING(AvgPoolV2).Tiling(TilingForAvgPoolV2).TilingParse<AvgPoolV2TilingParseInfo>(TilingPrepareForAvgPoolV2);
} // namespace gert
