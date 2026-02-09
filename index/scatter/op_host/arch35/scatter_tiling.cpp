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
 * \file scatter_tiling.cpp
 * \brief
 */
#include "scatter_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "scatter_tiling_arch35.h"

namespace optiling {

ge::graphStatus Tiling4Scatter(gert::TilingContext* context) {
  auto compileInfoPtr = static_cast<const ScatterKvCompileInfo*>(context->GetCompileInfo());
  ScatterTiling scatterTiling(context);
  return scatterTiling.DoTiling();
}

static ge::graphStatus TilingPrepareForScatter(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepareForScatter running");
  auto compile_info = GetCompileInfoPtr<ScatterKvCompileInfo>(context);
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  auto platform_info = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platform_info);
  auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
  compile_info->core_num = ascendc_platform.GetCoreNumAiv();
  uint64_t ub_size;
  ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  compile_info->ub_size = ub_size;
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Scatter op.
IMPL_OP_OPTILING(Scatter).Tiling(Tiling4Scatter).TilingParse<ScatterKvCompileInfo>(TilingPrepareForScatter);
}  // namespace optiling