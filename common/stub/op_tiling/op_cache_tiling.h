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
 * \file cop_ache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_OP_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_OP_CACHE_TILING_H

#include <array>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "op_cache_def_tiling.h"

namespace optiling {
const std::string WQBMM_MSD = "wqbmm_msd";
const std::string WQBMM_CUSTOM = "wqbmm_custom";

bool TilingPrepareForOpCache(gert::TilingContext* context) __attribute__((weak));
bool TilingPrepareForOpCache(gert::TilingParseContext* context) __attribute__((weak));

bool GenTiling(
    const std::string& op_type, const BatchmatmulCompileParas& compile_params, BatchmatmulRunParas& run_params,
    CacheTilingData& tiling, gert::TilingContext* context);

bool GenWqbmmTiling(
    const std::string& op_type, const WeightQuantBatchMatmulCacheTilingParas& compile_params,
    WeightQuantBatchMatmulCacheTilingData& cacheTiling) __attribute__((weak));

} // namespace optiling

#endif