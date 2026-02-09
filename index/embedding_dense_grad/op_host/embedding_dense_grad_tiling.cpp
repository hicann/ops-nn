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
 * \file embedding_dense_grad_tiling.cpp
 * \brief
 */
#include "embedding_dense_grad_tiling.h"
#include "embedding_dense_grad_tiling_arch35.h"
#include "log/log.h"

using namespace ge;

namespace {
const size_t INPUT_INDEX_GRAD = 0;
const size_t INPUT_INDEX_INDICES = 1;
const size_t INDEX_ATTR_NUM_WEIGHTS = 0;
const size_t INDEX_ATTR_PADDING_IDX = 1;
const int64_t FLOAT_32_BYTES = 4;
const int64_t INT_32_BYTES = 4;
const int64_t INT_64_BYTES = 8;
const int64_t BLOCK_SIZE = 32;
const size_t SYNC_PER_CORE_BYTES = 32;
const int64_t UB_SIZE_BYTES = static_cast<int64_t>(192) * 1024 - static_cast<int64_t>(16) * 1024;
}  // namespace

namespace optiling {

static ge::graphStatus EmbeddingDenseGradTiling(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "EmbeddingDenseGradTiling running begin");
  const EmbeddingDenseGradCompileInfo* compile_info = static_cast<const EmbeddingDenseGradCompileInfo*>(context->GetCompileInfo());

  OP_LOGD(context->GetNodeName(), "EmbeddingDenseGradTiling tik compile_info is Null! Running AscendC tiling!");
  uint32_t maxCoreNum = compile_info->maxCoreNum;
  uint32_t ubSizePlatform = compile_info->ubSizePlatform;
  uint32_t maxThreadNum = compile_info->maxThreadNum;

  return Tiling4EmbeddingDenseGradSimd(context, maxCoreNum, ubSizePlatform, maxThreadNum);
}

static ge::graphStatus TilingPrepareForEmbeddingDenseGrad(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "TilingPrepareForEmeddingDenseGrad running.");
  auto compile_info = context->GetCompiledInfo<EmbeddingDenseGradCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  OP_LOGD(context->GetNodeName(), "AscendC Prepare func running!");
  auto platformInfo = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compile_info->maxCoreNum = ascendcPlatform.GetCoreNumAiv();
  OP_CHECK_IF((compile_info->maxCoreNum <= 0),
                  OP_LOGE(context->GetNodeName(), "The core num is invalid."),
                  return ge::GRAPH_FAILED);
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  compile_info->ubSizePlatform = static_cast<uint32_t>(ubSize);
  OP_CHECK_IF((compile_info->ubSizePlatform <= 0),
                  OP_LOGE(context->GetNodeName(), "The ubSize is invalid."),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingDenseGrad)
    .Tiling(EmbeddingDenseGradTiling)
    .TilingParse<EmbeddingDenseGradCompileInfo>(TilingPrepareForEmbeddingDenseGrad);
}  // namespace optiling
