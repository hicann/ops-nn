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
 * \file layer_norm_tiling_arch35.cpp
 * \brief
 */

#include "op_host/tiling_util.h"
#include "op_api/runtime2_util.h"
#include "layer_norm_tiling_arch35.h"

namespace optiling {
static ge::graphStatus TilingPrepare4LayerNorm(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4LayerNorm.");
  LayerNormOpInfo* compile_info = GetCompileInfoPtr<LayerNormOpInfo>(context);
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  OP_LOGD(context->GetNodeName(), "TilingPrepare4LayerNorm for ascendc enter.");
  return TilingPrepare4LayerNormV3ForAscendC(context, compile_info->regbaseCompileInfo);
}

static ge::graphStatus Tiling4LayerNorm(gert::TilingContext* context) {
  // compile info
  const LayerNormOpInfo* compile_info = reinterpret_cast<const LayerNormOpInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  OP_LOGD(context->GetNodeName(), "LayerNorm ascendc tiling enter.");
  return Tiling4LayerNormV3ForAscendC(context);
}

// register tiling interface of LayerNorm op.
IMPL_OP_OPTILING(LayerNorm).Tiling(Tiling4LayerNorm).TilingParse<LayerNormOpInfo>(TilingPrepare4LayerNorm);
}  // namespace optiling
