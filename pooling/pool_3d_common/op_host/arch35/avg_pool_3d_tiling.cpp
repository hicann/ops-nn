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
 * \file avg_pool_3d_tiling.cpp
 * \brief
 */

#include "pool_tiling_templates_registry.h"
#include "avg_pool_3d_tiling_common.h"

using namespace AscendC;
using optiling::PoolTilingRegistry;
namespace optiling
{

ge::graphStatus Tiling4AvgPool3DRegBase(gert::TilingContext* context)
{
    return PoolTilingRegistry::GetInstance().DoTilingImpl(context);
}

}  // namespace optiling