/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_util.cpp
 * \brief
 */
#include "scatter_tiling_util.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"

namespace optiling {
ge::graphStatus TilingPrepareScatterNdAddForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init ScatterNdAdd SimtTiling.");
    auto ci = context->GetCompiledInfo<ScatterCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->core_num <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to core num."),
                    return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ub_size <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4Scatter(gert::TilingParseContext* context) {
  TilingPrepareScatterNdAddForAscendC(context);
  OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4ScatterNdAdd Simt GRAPH_SUCCESS.");
  return ge::GRAPH_SUCCESS;
}

bool CheckScatterOpsTensorShape(const gert::TilingContext* context, const CalcShapeInfo& calc_shape_info) {
  if (calc_shape_info.var_shape != calc_shape_info.out_shape) {
    OP_LOGE(context->GetNodeName(), "the length of var must be same as the length of output.");
    return false;
  }

  if (calc_shape_info.indices_shape.GetDimNum() == 1 && calc_shape_info.indices_shape.GetDim(0) == 1 &&
      calc_shape_info.var_shape.GetDimNum() - calc_shape_info.adds_shape.GetDimNum() == 1) {
    OP_LOGI(context->GetNodeName(), "Input indices is a scalar.");
  }

  return true;
}
}  // namespace optiling