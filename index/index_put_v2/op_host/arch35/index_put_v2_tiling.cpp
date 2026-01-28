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
 * \file index_put_v2.cc
 * \brief IndexPutV2 tiling file
 */
#include <vector>
#include "index_put_v2_tiling.h"
#include "index/index/op_host/arch35/index_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

namespace optiling {
using namespace Ops::Base;

static ge::graphStatus Tiling4IndexPutV2(gert::TilingContext* context) {
  OP_LOGD("indexputv2", "Tilingt2.0 start");
  auto param = context->GetTilingData<IndexPutV2Params>();
  OP_CHECK_NULL_WITH_CONTEXT(context, param);
  auto compile_info = reinterpret_cast<const IndexPutV2CompileInfo*>(context->GetCompileInfo());
  OP_LOGD(context->GetNodeName(), "Tiling4IndexPut dsl compile_info is Null, running Simt tiling.");
  IndexSimtTiling tilingObj(context);
  tilingObj.isIndexPut_ = true;
  tilingObj.coreNum_ = compile_info->core_num;
  tilingObj.ubSize_ = compile_info->ub_max_size;
  return tilingObj.DoTiling();
}

static ge::graphStatus TilingPrepare4IndexPutV2(gert::TilingParseContext* context){
  OP_LOGD(context->GetNodeName(), "Start init TilingPrepare4IndexPutV2.");
  auto ci = context->GetCompiledInfo<IndexPutV2CompileInfo>();
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
  ci->ub_max_size = static_cast<int64_t>(ubSize);
  OP_CHECK_IF((ci->ub_max_size <= 0),
                  OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the IndexPutV2 op.
IMPL_OP_OPTILING(IndexPutV2)
    .Tiling(Tiling4IndexPutV2)
    .TilingParse<IndexPutV2CompileInfo>(TilingPrepare4IndexPutV2);
}  // namespace optiling
