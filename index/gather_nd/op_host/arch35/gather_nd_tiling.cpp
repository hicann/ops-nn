/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_nd_tiling.cpp
 * \brief
 */

#include "gather_nd_tiling.h"
#include "gather_nd_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace optiling {

ge::graphStatus Tiling4GatherNd(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do Tiling4GatherNd");
  auto compile_info = static_cast<const GatherNdCompileInfo*>(context->GetCompileInfo());

  OP_LOGD(context->GetNodeName(), "GatherNdSimt running.");
  GatherNdSimtTiling tilingObj(context);
  return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4GatherNd(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepareGatherNdForAscendC entering.");
  auto gatherNdCompileInfo = context->GetCompiledInfo<GatherNdCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, gatherNdCompileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  gatherNdCompileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
  OP_CHECK_IF((gatherNdCompileInfo->core_num <= 0),
                  OP_LOGE(context->GetNodeName(), "Failed to core num."),
                  return ge::GRAPH_FAILED);
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  gatherNdCompileInfo->ub_size = static_cast<int64_t>(ubSize);
  OP_CHECK_IF((gatherNdCompileInfo->ub_size <= 0),
                  OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of GatherNd op.
IMPL_OP_OPTILING(GatherNd).Tiling(Tiling4GatherNd).TilingParse<GatherNdCompileInfo>(TilingPrepare4GatherNd);
}  // namespace optiling
