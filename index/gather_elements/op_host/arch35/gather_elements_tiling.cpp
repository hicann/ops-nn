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
 * \file gather_elements_tiling.cpp
 * \brief
 */
#include "gather_elements_tiling.h"
#include "gather_elements_tiling_arch35.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"

namespace
{
  const int32_t BLOCK_SIZE = 32;
  const int32_t PARAMS_CUT_INTO_SLICE_UB = 149984;
  const int32_t PARAMS_CARRY_BLOCK_UB = 730 * 1024;
  const int32_t CONVERT_TO_AICPU_UB = 3000 * 1024;
  const int32_t INDICES_NUM_THRESHOULD = 2048;
  const int32_t RESERVED_UB_SIZE = 2 * 1024;
  const int32_t DIVISION_BY_ZERO_LOG = -100000;
  const int32_t LARGE_MODEL_MIN_AXIS = 128;
  const int32_t INT_MAX_NUM = 2147483647;
  const int32_t HALF = 2;
  const int32_t MIN_AXIS = 256;
  const int32_t MIN_PRE = 3000;

  // A: params larger than cache_ub
  // B: indices larger than the number contained in one block for each core
  // C: remaining indices larger than one block

  const int64_t TILING_MODE_X_LARGE_INDICES_LARGE = 1;
  const int64_t TILING_MODE_X_SMALL_INDICES_LARGE = 2;
  const int64_t TILING_MODE_X_SLICE_INDICES_LARGE = 3;
  const int64_t TILING_MODE_DIF = 3;
  const int64_t TILING_MODE_X_LARGE_INDICES_LARGE_DIFF_SHAPE = 4;
  const int64_t TILING_MODE_X_SMALL_INDICES_LARGE_DIFF_SHAPE = 5;
  const int64_t TILING_MODE_X_SLICE_INDICES_LARGE_DIFF_SHAPE = 6;
  // tiling mode when params and indices are so large that both are cut into slices
  const int64_t TILING_MODE_FOR_LAST_AXIS = 7;
  const int64_t TILING_MODE_FOR_LAST_AXIS_VGATHER = 8;
  const int64_t TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE = 9;
  const int64_t TILING_MODE_FOR_LAST_AXIS_CUT_VGATHER = 10;
  const int64_t TILING_MODE_FOR_LAST_AXIS_VGATHER_310P = 11;

  const size_t DIM_0 = 0;
  const size_t DIM_1 = 1;
  const size_t DIM_2 = 2;
  const size_t DIM_3 = 3;
  const size_t DIM_4 = 4;
  const size_t DIM_5 = 5;
  const size_t DIM_6 = 6;
  const size_t DIM_7 = 7;

  const size_t MAX_DIMS = 8;
  const int64_t PARAMS_AXIS_PRE_NONE = 1;
  const int64_t LEAST_REPEAT_TIME = 1;

  const size_t INDEX_ATTR_AXIS = 0;
  const size_t SIZE_INT32 = 4;
  const size_t EXTRA_THREE_UB = 3;
  const std::string OP_NAME = "GatherElements";
} // namespace

namespace optiling {
static ge::graphStatus Tiling4GatherElements(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "Tiling4GatherElements running begin");
  return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareGatherElementsForAscendC(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareGatherElementsForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<GatherElementsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->core_num <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to core num."),
                    return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ub_size <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GatherElements(gert::TilingParseContext *context)
{
  OP_LOGD(context->GetNodeName(), "TilingPrepare4GatherElements running.");
  auto compile_info = context->GetCompiledInfo<GatherElementsCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4GatherElements Simt Mode success!");
  return TilingPrepareGatherElementsForAscendC(context);
}

// register tiling interface of the GatherElements op.
IMPL_OP_OPTILING(GatherElements)
    .Tiling(Tiling4GatherElements)
    .TilingParse<GatherElementsCompileInfo>(TilingPrepare4GatherElements);
} // namespace optiling