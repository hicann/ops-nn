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
 * \file batch_norm_tiling.cpp
 * \brief
 */

#include "batch_norm_tiling.h"

namespace optiling {

static ge::graphStatus Tiling4BatchNorm(gert::TilingContext* context)
{
  return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4BatchNorm(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepare4BatchNorm enter.");

  auto compileInfo = GetCompileInfoPtr<BatchNormCompileInfo>(context);
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
  OP_CHECK_IF((compileInfo->coreNum <= 0),
              OP_LOGE(context->GetNodeName(),
              "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
              return ge::GRAPH_FAILED);

  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  compileInfo->ubSize = ubSizePlatForm;
  OP_CHECK_IF((compileInfo->ubSize <= 0),
              OP_LOGE(context->GetNodeName(),
              "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
              return ge::GRAPH_FAILED);
  compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
  OP_CHECK_IF((compileInfo->blockSize <= 0),
              OP_LOGE(context->GetNodeName(),
              "Get block Size failed, block size: %u", static_cast<uint32_t>(compileInfo->blockSize)),
              return ge::GRAPH_FAILED);
  compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
  OP_CHECK_IF((compileInfo->vectorLength <= 0),
              OP_LOGE(context->GetNodeName(),
              "Get vector Length failed, vector Length: %u", static_cast<uint32_t>(compileInfo->vectorLength)),
              return ge::GRAPH_FAILED);

  OP_LOGD(context->GetNodeName(),
          "TilingPrepare4BatchNorm exit, coreNum: %lu, ubSize: %lu, blockSize: %lu, vectorLen: %u.",
          compileInfo->coreNum, compileInfo->ubSize, compileInfo->blockSize, compileInfo->vectorLength);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchNorm)
    .Tiling(Tiling4BatchNorm)
    .TilingParse<BatchNormCompileInfo>(TilingPrepare4BatchNorm);

} // namespace optiling
