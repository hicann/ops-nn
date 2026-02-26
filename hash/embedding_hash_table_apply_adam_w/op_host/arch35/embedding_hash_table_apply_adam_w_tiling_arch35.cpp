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
 * \file embedding_hash_table_apply_adam_w_tiling_arch35.cpp
 * \brief
 */

#include "embedding_hash_table_apply_adam_w_tiling_arch35.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint32_t INPUT_KEYS_IDX = 1;
constexpr uint32_t INPUT_VALUES_IDX = 2;
constexpr uint32_t ATTR_EMBEDDINGDIM_IDX = 0;
constexpr uint32_t ATTR_TABLE_SIZE_IDX = 1;
constexpr uint32_t MAX_UINT32 = 4294967295;

// values support datatype
static const std::unordered_map<ge::DataType, uint64_t> VALUES_DATA_TYPE_TO_INT{{ge::DataType::DT_FLOAT16, 2},
                                                                                {ge::DataType::DT_BF16, 2},
                                                                                {ge::DataType::DT_FLOAT, 4},
                                                                                {ge::DataType::DT_INT32, 4},
                                                                                {ge::DataType::DT_INT64, 8}};

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  coreNum_ = ascendcPlatform.GetCoreNumAiv();
  OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(opName, "EmbeddingHashTableApplyAdamWTiling fail to get coreNum."), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::GetShapeAttrsInfo() {
  auto const keyShape = context_->GetInputShape(INPUT_KEYS_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
  auto const keyShapeVal = keyShape->GetStorageShape();
  int64_t keyShapeSize = keyShapeVal.GetShapeSize();
  OP_CHECK_IF((keyShapeSize < 0) || (keyShapeSize > MAX_UINT32), OP_LOGE(opName, 
      "keyShape has a value of %ld, which exceeds the representable range of the uint32_t type.", keyShapeSize), 
      return ge::GRAPH_FAILED);
  keyNum_ = static_cast<uint32_t>(keyShapeSize);

  auto values = context_->GetInputDesc(INPUT_VALUES_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, values);
  auto valuesDtype = values->GetDataType();

  auto iter = VALUES_DATA_TYPE_TO_INT.find(valuesDtype);
  if (iter != VALUES_DATA_TYPE_TO_INT.end()) {
    bitWidth_ = iter->second;
  } else {
    OP_LOGD(context_->GetNodeName(), "valuesDtype = %u not supported. please check.", valuesDtype);
    return ge::GRAPH_FAILED;
  }

  auto const attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const uint32_t* tableSize = attrs->GetAttrPointer<uint32_t>(ATTR_TABLE_SIZE_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, tableSize);
  tableSize_ = static_cast<uint32_t>(*tableSize);
  const uint32_t* embeddingDim = attrs->GetAttrPointer<uint32_t>(ATTR_EMBEDDINGDIM_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, embeddingDim);
  embeddingDim_ = static_cast<uint32_t>(*embeddingDim);
  const bool* amsgrad = attrs->GetAttrPointer<bool>(2);
  OP_CHECK_NULL_WITH_CONTEXT(context_, amsgrad);
  amsgrad_ = static_cast<uint32_t>(*amsgrad);
  const bool* maximize = attrs->GetAttrPointer<bool>(3);
  OP_CHECK_NULL_WITH_CONTEXT(context_, maximize);
  maximize_ = static_cast<uint32_t>(*maximize);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::DoOpTiling() {
  tilingData.set_keyNum(keyNum_);
  tilingData.set_bitWidth(bitWidth_);
  tilingData.set_tableSize(tableSize_);
  tilingData.set_embeddingDim(embeddingDim_);
  tilingData.set_amsgrad(amsgrad_);
  tilingData.set_maximize(maximize_);

  if (embeddingDim_ < MAX_THREAD) {
    blockX_ = (embeddingDim_ + MIN_THREAD - 1) / MIN_THREAD * MIN_THREAD;
    blockY_ = MAX_THREAD / blockX_;
    blockNum_ = (keyNum_ + blockY_ - 1) / blockY_;
  } else {
    blockX_ = MAX_THREAD;
    blockY_ = 1;
    blockNum_ = keyNum_ * ((embeddingDim_ + MAX_THREAD - 1) / MAX_THREAD);
  }
  tilingData.set_blockX(blockX_);
  tilingData.set_blockY(blockY_);
  tilingData.set_blockNum(blockNum_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t EmbeddingHashTableApplyAdamWTiling::GetTilingKey() const {
  uint64_t tilingKey = 100;  // 100: embedding base tilingKey
  return tilingKey + bitWidth_;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::GetWorkspaceSize() {
  workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingHashTableApplyAdamWTiling::PostTiling() {
  auto workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(std::min(static_cast<uint32_t>(blockNum_), coreNum_));
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4EmbeddingHashTableApplyAdamW(gert::TilingContext* context) {
  EmbeddingHashTableApplyAdamWTiling hashTiling(context);
  return hashTiling.DoTiling();
}

static ge::graphStatus TilingPrepare4EmbeddingHashTableApplyAdamW(gert::TilingParseContext*) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingHashTableApplyAdamW)
    .Tiling(Tiling4EmbeddingHashTableApplyAdamW)
    .TilingParse<EmbeddingHashTableApplyAdamWCompileInfo>(TilingPrepare4EmbeddingHashTableApplyAdamW);
}  // namespace optiling