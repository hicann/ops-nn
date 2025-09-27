/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file op_cache_tiling.cpp
 * \brief
 */

#include "op_cache_tiling.h"

namespace optiling {
bool TilingPrepareForOpCache(gert::TilingContext* /*context*/)
{
    return true;
}

bool TilingPrepareForOpCache(gert::TilingParseContext* /*context*/)
{
    return true;
}

bool GenTiling(
    const std::string& /*op_type*/, const BatchmatmulCompileParas& /*compile_params*/,
    BatchmatmulRunParas& /*run_params*/, CacheTilingData& /*tiling*/, gert::TilingContext* /*context*/)
{
    return true;
}

bool GenWqbmmTiling(
    const std::string& /*op_type*/, const WeightQuantBatchMatmulCacheTilingParas& /*compile_params*/,
    WeightQuantBatchMatmulCacheTilingData& /*cacheTiling*/)
{
    return true;
}
} // namespace optiling
