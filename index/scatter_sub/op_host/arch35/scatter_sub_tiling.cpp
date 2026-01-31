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
 * \file scatter_sub_tiling.cpp
 * \brief scatter_sub_tiling
 */
#include "scatter_sub_tiling.h"
#include "../../../scatter_add/op_host/arch35/scatter_add_tiling_base.h"

namespace optiling {
// -----------------ScatterSub Util START------------------
static ge::graphStatus TilingPrepare4ScatterSub(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ScatterSub(gert::TilingContext* context) 
{
  OP_LOGD(context->GetNodeName(), "ScatterSubTiling running begin");
  auto compileInfo = reinterpret_cast<const ScatterSubCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

  OP_LOGD(context->GetNodeName(), "ScatterSubTiling is ascendc. runing ascendc tiling.");
  return ScatterAddTilingForAscendC(context); 
}

// register tiling interface of the ScatterSubTiling op.
IMPL_OP_OPTILING(ScatterSub).Tiling(Tiling4ScatterSub).TilingParse<ScatterSubCompileInfo>(TilingPrepare4ScatterSub);
}  // namespace optiling
