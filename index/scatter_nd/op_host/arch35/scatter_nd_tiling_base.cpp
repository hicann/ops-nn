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
 * \file scatter_nd_tiling_base.cpp
 * \brief
 */

#include "scatter_nd_tiling_base.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
using namespace AscendC;

namespace optiling {

namespace {
constexpr uint16_t INPUT_IDX_INDICES = 0;
constexpr uint16_t INPUT_IDX_UPDATES = 1;
constexpr uint16_t OUTPUT_IDX_SHAPE = 0;
constexpr uint16_t RANK_MIN_VALUE = 1;
constexpr uint16_t RANK_MAX_VALUE = 7;
constexpr uint64_t MIN_TILING_SIZE = 128;
constexpr uint64_t MIN_BLOCK_SIZE = 512;
constexpr uint32_t DCACHE_SIZE = static_cast<uint32_t>(32) * 1024;
constexpr uint32_t REGBASE_CCEC_CACHE_SIZE = static_cast<uint32_t>(8) * 1024;
constexpr uint32_t RESERVED_WORKSPACE_SIZE = static_cast<uint32_t>(16) * 1024 * 1024;
constexpr uint32_t INPUT_ADDRESS_IN_INT32 = 0;
constexpr uint32_t INPUT_ADDRESS_IN_INT64 = 20;
constexpr uint32_t ONE = 1;
constexpr uint32_t TWO = 2;
constexpr uint32_t THREE = 3;

constexpr uint32_t MIN_PRE_NUM = 256;
constexpr uint16_t ARR_LENGTH = 2;
constexpr uint64_t DETERMIN_BLOCK_SIZE = 1024;
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint32_t SORT_UTIL_SIZE = 20;
constexpr int32_t UB_AGLIN_VALUE = 32;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t UINT32_BYTES = 4;
constexpr uint32_t UB_RESERVED = static_cast<uint32_t>(128) * 256;

constexpr uint32_t MAX_SIZE_QUANT = 128 * 32;

constexpr uint16_t THRESHOLD_TOP = 256;
constexpr uint16_t THRESHOLD_MID = 128;
constexpr uint16_t THRESHOLD_BOTTOM = 16;
constexpr float THRESHOLD_FACTOR = 0.8;
} // namespace

static const std::unordered_map<ge::DataType, uint64_t> INDICE_DATA_TYPE_TO_INT {
  {ge::DataType::DT_INT32, 100},
  {ge::DataType::DT_INT64, 200}
};

static const std::unordered_map<ge::DataType, uint64_t> UPDATE_DATA_TYPE_TO_INT {
  {ge::DataType::DT_INT32, 1},
  {ge::DataType::DT_FLOAT16, 2},
  {ge::DataType::DT_FLOAT, 3}
};

static const std::set<ge::DataType> scatterNdDeterminsiticType = { ge::DT_FLOAT, ge::DT_FLOAT16};

template <typename T1, typename T2>
static inline auto CeilDiv(T1 a, T2 b) -> T1 {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

ge::graphStatus ScatterNdTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  OP_CHECK_IF(platformInfo == nullptr,
          OP_LOGE(opName, "fail to get platform info"),
          return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  OP_CHECK_IF((aivNum <= 0),
          OP_LOGE(opName, "ScatterNdTiling fail to get coreNum."),
          return ge::GRAPH_FAILED);
  coreNum_ = aivNum;
  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  OP_CHECK_IF((ubSizePlatForm <= DCACHE_SIZE),
          OP_LOGE(opName, "ub size less than Dcache Size. please check"),
          return ge::GRAPH_FAILED);
  // UB Size Need reserve space for Dcache / CCEC Compile Stack.
  ubSize_ = static_cast<uint32_t>(ubSizePlatForm - DCACHE_SIZE);
  auto res = context_->SetLocalMemorySize(ubSize_);
  OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
          OP_LOGE(opName, "SetLocalMemorySize ubSize = %u failed.", ubSize_),
          return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdTiling::GetShapeAttrsInfo() {

  auto indices = context_->GetInputTensor(INPUT_IDX_INDICES);
  OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
  indiceShapeSize_ = indices->GetShapeSize();
  auto indiceShape = indices->GetStorageShape();
  auto indiceDims = indiceShape.GetDimNum();
  rankSize_ = indiceShape.GetDim(indiceDims - 1);
  OP_CHECK_IF((RANK_MIN_VALUE > static_cast<uint16_t>(rankSize_) ||
                  static_cast<uint16_t>(rankSize_) > RANK_MAX_VALUE),
                  OP_LOGE(opName,
                 " rankSize %u out of range[1, 7], please check.", rankSize_),
                  return ge::GRAPH_FAILED);
  
  auto updates = context_->GetInputTensor(INPUT_IDX_UPDATES);
  OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
  updateShapeSize_ = updates->GetShapeSize();
  updateDtype = context_->GetInputDesc(INPUT_IDX_UPDATES)->GetDataType();
  indiceDtype = context_->GetInputDesc(INPUT_IDX_INDICES)->GetDataType();
  auto shapeValue = context_->GetOutputShape(OUTPUT_IDX_SHAPE)->GetStorageShape();
  uint64_t shapeRank = shapeValue.GetDimNum();
  OP_CHECK_IF((shapeRank < rankSize_),
          OP_LOGE(opName,
         " shapeRank %lu less than rank %u, please check.", shapeRank, rankSize_),
          return ge::GRAPH_FAILED);

  for (uint64_t idx = 0; idx < shapeRank; idx++) {
    outPutShape[idx] = shapeValue.GetDim(idx);
    outputShapeSize_ *= outPutShape[idx];
  } 

  if (context_->GetDeterministic() && scatterNdDeterminsiticType.find(updateDtype) != scatterNdDeterminsiticType.end()) {
    isDeterminTemplate_ = static_cast<uint64_t>(1);
  }
  
  updateDtypeSize_ = ge::GetSizeByDataType(updateDtype);
  indiceDtypeSize_ = ge::GetSizeByDataType(indiceDtype);

  preAxis_ = indiceShapeSize_ / rankSize_;
  afterAxis_ = updateShapeSize_ / preAxis_;
  outputSize_ = shapeValue.GetShapeSize();
  OP_CHECK_IF((outputSize_ <= 0),
          OP_LOGE(opName,
         " shape should not be empty !"),
          return ge::GRAPH_FAILED);
  rankFusedAxis_ = outputSize_ / afterAxis_;

  return ge::GRAPH_SUCCESS;
}

uint64_t ScatterNdTiling::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
{
  std::vector<int64_t> shapeVec = { lastAxisNum };
  ge::Shape srcShape(shapeVec);
  AscendC::SortConfig config;
  config.hasSrcIndex = false;
  config.hasDstIndex = true;
  config.type = AscendC::SortType::RADIX_SORT;
  config.isDescend = isDescend;
  uint32_t maxValue = 0;
  uint32_t minValue = 0;
  AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
  OP_LOGI("RadixSortTilingForAscendC", "Need tmp buffer %u byte for ac sort api", maxValue);
  return maxValue;
}

ge::graphStatus ScatterNdTiling::getRestAvailableSize(uint64_t sampleNum, uint64_t valueTypeBytes, uint64_t originalSize, uint64_t postAxisSize_,
    ge::DataType idType)
{
  uint64_t indicesDtypeSize = ge::GetSizeByDataType(idType);
  OP_CHECK_IF(indicesDtypeSize <= 0, OP_LOGE(opName, "get indicesType size fail."),
                  return ge::GRAPH_FAILED);
  auto ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
  uint64_t occupy = sampleNum * Ops::Base::CeilAlign(FLOAT_BYTES * postAxisSize_, ubBlock) + sampleNum * Ops::Base::CeilAlign(FLOAT_BYTES * postAxisSize_, ubBlock) + 
      Ops::Base::CeilAlign(sampleNum * indicesDtypeSize, ubBlock) * THREE + Ops::Base::CeilAlign(sampleNum * UINT32_BYTES, ubBlock) + TWO * TWO * UB_AGLIN_VALUE +
      TWO * UB_AGLIN_VALUE + GetSortTmpSize(idType, sampleNum, false) + Ops::Base::CeilAlign(postAxisSize_ * FLOAT_BYTES, ubBlock);
  return originalSize - occupy;
}

void ScatterNdTiling::BlockTiling() {
  auto typeSize = ge::GetSizeByDataType(updateDtype);
  alignFactor_ = Ops::Base::GetUbBlockSize(context_) / typeSize;
  auto blockFactor = CeilDiv(updateShapeSize_, coreNum_);
  auto blockAlignFactor = CeilDiv(blockFactor, alignFactor_) * alignFactor_;
  blockTilingSize_ = std::max(static_cast<uint64_t>(blockAlignFactor), MIN_TILING_SIZE);
  blockNum_ = CeilDiv(updateShapeSize_, blockTilingSize_);
  tailBlockTilingSize_ = updateShapeSize_ - blockTilingSize_ * (blockNum_ - static_cast<uint64_t>(1));
  OP_LOGD(opName, "updateShapeSize_ = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
     " blockTilingSize = %d, tailBlockTilingSize = %d", updateShapeSize_,
      blockFactor, blockAlignFactor, blockTilingSize_, tailBlockTilingSize_);

  auto clrBlockFactor = CeilDiv(outputShapeSize_, coreNum_);
  auto clrBlockAlignFactor = CeilDiv(clrBlockFactor, alignFactor_) * alignFactor_;
  clrBlockTilingSize_ = std::max(static_cast<uint64_t>(clrBlockAlignFactor), MIN_BLOCK_SIZE);
  clrBlockNum_ = CeilDiv(outputShapeSize_, clrBlockTilingSize_);
  clrTailBlockTilingSize_ = outputShapeSize_ - clrBlockTilingSize_ * (clrBlockNum_ - static_cast<uint64_t>(1));
  OP_LOGD(opName, "outputShapeSize = %lld, clrBlockFactor = %lld,"
     " clrBlockTilingSize = %d, clrTailBlockTilingSize = %d", outputShapeSize_,
      clrBlockFactor, clrBlockTilingSize_, clrTailBlockTilingSize_);
  return;
}

ge::graphStatus ScatterNdTiling::UbTiling() {
  // halfUbSize for double buffer
  auto halfUbSize = ubSize_ / BUFFER_NUM;
  auto indiceNum = indiceShapeSize_ / rankSize_;
  sliceSize_ = updateShapeSize_ / indiceNum;
  OP_CHECK_IF(sliceSize_ == static_cast<uint64_t>(0),
                  OP_LOGE(opName, "sliceSize %lu is zero. please check.", sliceSize_),
                  return ge::GRAPH_FAILED);
  auto updateTypeSize = ge::GetSizeByDataType(updateDtype);
  
  auto indiceTypeSize = ge::GetSizeByDataType(indiceDtype);
  // sliceUb : the required size of UB for one scatter operation;
  auto sliceUb = sliceSize_ * updateTypeSize + rankSize_ * indiceTypeSize;
  sliceUb = CeilDiv(sliceUb, alignFactor_) * alignFactor_;
  if (sliceUb > halfUbSize) {
    // for scatter operator. At least  rank size index need to be move in UB.
    ubTilingSize_ = (halfUbSize - rankSize_ * indiceTypeSize) / updateTypeSize;
  } else {
    // calculate the size of updates that need to be move in UB
    auto maxIndiceCnt = halfUbSize / sliceUb;
    ubTilingSize_ = maxIndiceCnt * sliceSize_;
  }
  OP_LOGD(opName, "sliceUb = %lu, halfUbSize = %u, ubTilingSize = %u",
    sliceUb, halfUbSize, ubTilingSize_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdTiling::ScatterNdDeterministicTiling()
{
  uint64_t halfUBSize = static_cast<uint64_t>(ubSize_ / TWO);
  bool DeterministicFlag = false;
  if (halfUBSize < (static_cast<uint64_t>(THREE) * FLOAT_BYTES * afterAxis_ + UB_RESERVED) || (preAxis_ < (static_cast<uint64_t>(coreNum_) / TWO)) ||
      afterAxis_ * FLOAT_BYTES > MAX_SIZE_QUANT) {
    DeterministicFlag = true;
  }
  if (DeterministicFlag) {
    OP_LOGD(opName, "ScatterNd Deterministic Non-Quant branch start");
    ubSize_ = static_cast<uint32_t>(ubSize_ / BUFFER_NUM);
    perCoreHandleCol_ = Ops::Base::CeilDiv(afterAxis_, coreNum_);
    logicCoreNum_ = Ops::Base::CeilDiv(afterAxis_, perCoreHandleCol_);
    tailCoreHandleCol_ = static_cast<uint64_t>(afterAxis_) - (logicCoreNum_ - static_cast<uint64_t>(1)) * perCoreHandleCol_;
    auto postUbBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_)) / updateDtypeSize_;
    if (perCoreHandleCol_ * updateDtypeSize_ + indiceShapeSize_ * indiceDtypeSize_ > ubSize_) {
      indicesUbFactor_ = MIN_PRE_NUM;
      auto indicesUbSize = indicesUbFactor_ * rankSize_ * indiceDtypeSize_;
      auto updateUbAvailableSize = (ubSize_ - indicesUbSize) / updateDtypeSize_;
      updatesUbFactor_ = Ops::Base::FloorAlign(updateUbAvailableSize, postUbBlock);
    } else {
      updatesUbFactor_ = Ops::Base::CeilAlign(perCoreHandleCol_, postUbBlock);
      indicesUbFactor_ = Ops::Base::FloorAlign(ubSize_ - updatesUbFactor_ * updateDtypeSize_ * BUFFER_NUM,
                      static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / (rankSize_ * indiceDtypeSize_);
    }

    indicesLoopSize_ = Ops::Base::CeilDiv(preAxis_, indicesUbFactor_);
    indicesTailUbFactor_ = static_cast<uint64_t>(preAxis_) - (indicesLoopSize_ - static_cast<uint64_t>(1)) * indicesUbFactor_;  
    updatesLoopSize_ = Ops::Base::CeilDiv(perCoreHandleCol_, updatesUbFactor_);
    updatesTailUbFactor_ = static_cast<uint64_t>(perCoreHandleCol_) - (updatesLoopSize_ - static_cast<uint64_t>(1)) * updatesUbFactor_;
    tailCoreColsLoopSize_ = Ops::Base::CeilDiv(tailCoreHandleCol_, updatesUbFactor_);
    tailCoreColsTailUbFactor_ = static_cast<uint64_t>(tailCoreHandleCol_) - (tailCoreColsLoopSize_ - static_cast<uint64_t>(1)) * updatesUbFactor_;

    return ge::GRAPH_SUCCESS;
  } 

  OP_LOGD(opName, "ScatterNd Deterministic Quant branch start");
  // 量化分支，sort按indices分核，indices搬多少行，update搬多少行
  isDeterministic_ = static_cast<uint64_t>(1);
  perCoreHandleIndices_ = Ops::Base::CeilDiv(preAxis_, coreNum_);
  uint32_t shiftOffset = UB_AGLIN_VALUE;
  // * updatesSum * updateFloatSum * indices * sortedIndices * sortedOriginIndex * uniqueIndices * shiftOffset * sortTmp
  uint64_t oneIdOccupyBytes = updateDtypeSize_ * afterAxis_ + static_cast<uint64_t>(FLOAT_BYTES) * afterAxis_ + 
  indiceDtypeSize_ * static_cast<uint64_t>(rankSize_) + static_cast<uint64_t>(TWO) * indiceDtypeSize_ + static_cast<uint64_t>(UINT32_BYTES) 
  + static_cast<uint64_t>(TWO) * shiftOffset * TWO + static_cast<uint64_t>(SORT_UTIL_SIZE);
  indicesUbFactor_ = Ops::Base::CeilDiv(halfUBSize, oneIdOccupyBytes);

  int32_t restSize = -1;
  while (restSize <= 0) {
    --indicesUbFactor_;
    restSize = getRestAvailableSize(indicesUbFactor_, updateDtypeSize_, halfUBSize, afterAxis_, indiceDtype);
  }
  if (indicesUbFactor_ > preAxis_) {
    indicesUbFactor_ = preAxis_;
  }
  indicesUbFactor_ = indicesUbFactor_ > perCoreHandleIndices_ ? perCoreHandleIndices_ : indicesUbFactor_;
  logicCoreNum_ = Ops::Base::CeilDiv(preAxis_, perCoreHandleIndices_);
  auto tailCoreHandleIndicesNum = preAxis_ - (logicCoreNum_ - static_cast<uint64_t>(1)) * perCoreHandleIndices_;
  indicesLoopSize_ = Ops::Base::CeilDiv(perCoreHandleIndices_, indicesUbFactor_);
  indicesTailUbFactor_ = perCoreHandleIndices_ - (indicesLoopSize_ - static_cast<uint64_t>(1)) * indicesUbFactor_;
  tailCoreIndicesLoopSize_ = Ops::Base::CeilDiv(tailCoreHandleIndicesNum, indicesUbFactor_);
  tailCoreIndicesTailUbFactor_ = tailCoreHandleIndicesNum - (tailCoreIndicesLoopSize_ - static_cast<uint64_t>(1)) * indicesUbFactor_;
  updatesUbFactor_ = afterAxis_; 

  float idxOutputRate = static_cast<float>(preAxis_) / rankFusedAxis_;
  float threshold = 0.0;
  uint64_t sizeAfterAxis_ = afterAxis_ * updateDtypeSize_;
  if (sizeAfterAxis_ >= THRESHOLD_TOP) {
      threshold = 1.0;
  } else if (sizeAfterAxis_ >= THRESHOLD_MID) {
      threshold = THRESHOLD_FACTOR;
  } else if (sizeAfterAxis_ >= THRESHOLD_BOTTOM) {
      threshold = static_cast<float>(sizeAfterAxis_) / THRESHOLD_MID * THRESHOLD_FACTOR;
  } else {
      threshold = 0.0;
  }

  if (idxOutputRate < threshold) {
    isIdxSplit = static_cast<uint64_t>(1);
  }

  auto ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
  if (isIdxSplit == static_cast<uint64_t>(0)) {
    perCoreHandleOutputRows_ = Ops::Base::CeilDiv(rankFusedAxis_, coreNum_);
    deQuantizeCoreNum_ = Ops::Base::CeilDiv(rankFusedAxis_, perCoreHandleOutputRows_);
    tailCoreHandleOutputRows_ = rankFusedAxis_ - (deQuantizeCoreNum_ - 1) * perCoreHandleOutputRows_;
  
    uint64_t oneRowOccupyBytes = afterAxis_ * UINT32_BYTES + afterAxis_ * FLOAT_BYTES;
    rowsInUb_ = ubSize_ / TWO / oneRowOccupyBytes;
    restSize = -1;
    do {
      uint64_t occupyBytes = rowsInUb_ * Ops::Base::CeilAlign(afterAxis_ * UINT32_BYTES, ubBlock) + 
                             rowsInUb_ * Ops::Base::CeilAlign(afterAxis_ * FLOAT_BYTES, ubBlock) + 
                             TWO * UB_AGLIN_VALUE + Ops::Base::CeilAlign(afterAxis_ * FLOAT_BYTES, ubBlock);
      restSize = (ubSize_ / TWO) - occupyBytes;
      if (restSize > 0) {
        break;
      }
      --rowsInUb_;
    } while (restSize <= 0);
    rowsInUb_ = rowsInUb_ > perCoreHandleOutputRows_ ? perCoreHandleOutputRows_ : rowsInUb_;
    deQuantizeCoreNum_ = Ops::Base::CeilDiv(rankFusedAxis_, perCoreHandleOutputRows_);
    tailCoreHandleOutputRows_ = rankFusedAxis_ - (deQuantizeCoreNum_ - static_cast<uint64_t>(1)) * perCoreHandleOutputRows_;
    
    return ge::GRAPH_SUCCESS;
  }
  
  uint64_t oneRowOccupyBytes = afterAxis_ * UINT32_BYTES + TWO * afterAxis_ * FLOAT_BYTES;
  rowsInUb_ = halfUBSize / oneRowOccupyBytes;
  restSize = -1;
  do {
    uint64_t occupyBytes = rowsInUb_ * Ops::Base::CeilAlign(afterAxis_ * UINT32_BYTES, ubBlock) +
                           rowsInUb_ * Ops::Base::CeilAlign(afterAxis_ * FLOAT_BYTES, ubBlock) * TWO +
                           Ops::Base::CeilAlign(rowsInUb_ * indiceDtypeSize_, ubBlock) + UB_AGLIN_VALUE;
    restSize = halfUBSize - occupyBytes;
    if (restSize > 0) {
      break;
    }
    --rowsInUb_;
  } while (restSize <= 0);
  rowsInUb_ = rowsInUb_ > perCoreHandleIndices_ ? perCoreHandleIndices_ : rowsInUb_;
  deQuantizeCoreNum_ = logicCoreNum_;
  dequantLoopSize = Ops::Base::CeilDiv(perCoreHandleIndices_, rowsInUb_);
  dequantTailUbFactor = perCoreHandleIndices_ - (dequantLoopSize - 1) * rowsInUb_;
  tailCoreDequantLoopSize = Ops::Base::CeilDiv(tailCoreHandleIndicesNum, rowsInUb_);
  tailCoreDequantTailUbFactor = tailCoreHandleIndicesNum - (tailCoreDequantLoopSize - 1) * rowsInUb_;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdTiling::DoOpTiling()
{
  if (isDeterminTemplate_) {
    OP_LOGD(opName, "ScatterNdDeterministicTiling is running");
    OP_CHECK_IF(ScatterNdDeterministicTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName, "ScatterNdDeterministicTiling fail."),
                    return ge::GRAPH_FAILED);
    SetStride();
    SetTilingData();
    OP_LOGD(opName, "ScatterNdDeterministicTiling success.");
    return ge::GRAPH_SUCCESS;
  }

  OP_LOGD(opName, "ScatterNd simt template is running");
  BlockTiling();
  ge::graphStatus res = UbTiling();
  if (res == ge::GRAPH_FAILED) {
    return ge::GRAPH_FAILED;
  }
  SetStride();
  SetTilingData();
  OP_LOGD(opName, "ScatterNd simt template success.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdTiling::DoLibApiTiling()
{
  return ge::GRAPH_SUCCESS;
}

uint64_t ScatterNdTiling::GetTilingKey() const
{
  uint64_t tilingKey = 0;
  auto iter = INDICE_DATA_TYPE_TO_INT.find(indiceDtype);
  if (iter != INDICE_DATA_TYPE_TO_INT.end()) {
    tilingKey += iter->second;
  } else {
    OP_LOGE(opName, "indiceDtype = %u not supported. please check.", indiceDtype);
  }
  iter = UPDATE_DATA_TYPE_TO_INT.find(updateDtype);
  if (iter != UPDATE_DATA_TYPE_TO_INT.end()) {
    tilingKey += iter->second;
  } else {
    OP_LOGE(opName, "updateDtype = %u not supported. please check.", updateDtype);
  }
  if (indiceShapeSize_ < UINT32_MAX && updateShapeSize_ < UINT32_MAX && outputShapeSize_ < UINT32_MAX) {
    tilingKey += INPUT_ADDRESS_IN_INT32;
  } else {
    tilingKey += INPUT_ADDRESS_IN_INT64;
  }
  OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
  return tilingKey;
}

ge::graphStatus ScatterNdTiling::GetWorkspaceSize()
{
  workspaceSize_ = RESERVED_WORKSPACE_SIZE;

  if (isDeterminTemplate_ && isDeterministic_) {
    uint64_t postVarAlignSize = Ops::Base::CeilAlign(afterAxis_ * sizeof(float), static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / sizeof(float);
    workspaceSize_ += logicCoreNum_ * perCoreHandleIndices_ * afterAxis_ * indiceDtypeSize_ +
                      logicCoreNum_ * perCoreHandleIndices_ * postVarAlignSize * FLOAT_BYTES +
                      outputShapeSize_ * TWO * UINT32_BYTES + rankStrideFusedAxis_ * UINT32_BYTES;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdTiling::PostTiling()
{
  auto workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  context_->SetTilingKey(GetTilingKey());
  auto largeBlockDim = blockNum_ > clrBlockNum_ ? blockNum_ : clrBlockNum_;
  if (isDeterminTemplate_) {
    largeBlockDim = coreNum_;
  }
  context_->SetBlockDim(largeBlockDim);
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                          context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

void ScatterNdTiling::SetStride()
{
  auto outPutShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE)->GetStorageShape();
  rankStrideFusedAxis_ = outPutShape.GetDim(rankSize_ - ONE);
  strideList[rankSize_ - ONE] = static_cast<uint64_t>(1);
  for (int16_t dim = static_cast<int16_t>(rankSize_ - TWO); dim >= 0; --dim) {
    strideList[dim] = strideList[dim + 1] * outPutShape.GetDim(dim + 1);
    rankStrideFusedAxis_ += strideList[dim] * outPutShape.GetDim(dim);
  }
  return;
}

void ScatterNdTiling::SetTilingData()
{
  tilingData.set_clrBlockNum(clrBlockNum_);
  tilingData.set_clrBlockTilingSize(clrBlockTilingSize_);
  tilingData.set_clrTailBlockTilingSize(clrTailBlockTilingSize_);
  tilingData.set_blockNum(blockNum_);
  tilingData.set_blockTilingSize(blockTilingSize_);
  tilingData.set_tailBlockTilingSize(tailBlockTilingSize_);
  tilingData.set_ubTilingSize(ubTilingSize_);
  tilingData.set_sliceSize(sliceSize_);
  tilingData.set_rankSize(rankSize_);
  tilingData.set_strideList(strideList);
  tilingData.set_outPutShape(outPutShape);
  tilingData.set_ubSize(ubSize_);
  tilingData.set_outputSize(outputSize_);
  tilingData.set_rowsInUb(rowsInUb_);
  tilingData.set_perCoreHandleCol(perCoreHandleCol_);
  tilingData.set_perCoreHandleIndices(perCoreHandleIndices_);
  tilingData.set_perCoreHandleOutputRows(perCoreHandleOutputRows_);
  tilingData.set_tailCoreHandleOutputRows(tailCoreHandleOutputRows_);
  tilingData.set_logicCoreNum(logicCoreNum_);
  tilingData.set_deQuantizeCoreNum(deQuantizeCoreNum_);
  tilingData.set_indicesUbFactor(indicesUbFactor_);
  tilingData.set_indicesLoopSize(indicesLoopSize_);
  tilingData.set_indicesTailUbFactor(indicesTailUbFactor_);
  tilingData.set_updatesUbFactor(updatesUbFactor_);
  tilingData.set_updatesLoopSize(updatesLoopSize_);
  tilingData.set_updatesTailUbFactor(updatesTailUbFactor_);
  tilingData.set_tailCoreHandleCol(tailCoreHandleCol_);
  tilingData.set_tailCoreColsLoopSize(tailCoreColsLoopSize_);
  tilingData.set_tailCoreColsTailUbFactor(tailCoreColsTailUbFactor_);
  tilingData.set_tailCoreIndicesLoopSize(tailCoreIndicesLoopSize_);
  tilingData.set_tailCoreIndicesTailUbFactor(tailCoreIndicesTailUbFactor_);
  tilingData.set_preAxis(preAxis_);
  tilingData.set_afterAxis(afterAxis_);
  tilingData.set_rankFusedAxis(rankFusedAxis_);
  tilingData.set_rankStrideFusedAxis(rankStrideFusedAxis_);
  tilingData.set_dequantLoopSize(dequantLoopSize);
  tilingData.set_dequantTailUbFactor(dequantTailUbFactor);
  tilingData.set_tailCoreDequantLoopSize(tailCoreDequantLoopSize);
  tilingData.set_tailCoreDequantTailUbFactor(tailCoreDequantTailUbFactor);
  tilingData.set_isIdxSplit(isIdxSplit);
  tilingData.set_isDeterministic(isDeterministic_);
  tilingData.set_isDeterminTemplate(isDeterminTemplate_);
  return;
}

template <typename T>
static std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}

void ScatterNdTiling::DumpTilingInfo()
{
  std::ostringstream info;
  info << "clrBlockNum: " << tilingData.get_clrBlockNum() << std::endl;
  info << "clrBlockTilingSize: " << tilingData.get_clrBlockTilingSize() << std::endl;
  info << "clrTailBlockTilingSize: " << tilingData.get_clrTailBlockTilingSize() << std::endl;
  info << "blockNum: " << tilingData.get_blockNum() << std::endl;
  info << "rankSize: " << tilingData.get_rankSize() << std::endl;
  info << "blockTilingSize: " << tilingData.get_blockTilingSize() << std::endl;
  info << "tailBlockTilingSize: " << tilingData.get_tailBlockTilingSize() << std::endl;
  info << "ubTilingSize: " << tilingData.get_ubTilingSize() << std::endl;
  info << "sliceSize: " << tilingData.get_sliceSize() << std::endl;
  info << "outPutShape: " << ToString(tilingData.get_outPutShape(), MAX_SHAPE_RANK).c_str() << std::endl;
  info << "strideList: " << ToString(tilingData.get_strideList(), MAX_RANK_COUNT).c_str() << std::endl;
  info << "ubSize: " << tilingData.get_ubSize() << std::endl;
  info << "outputSize: " << tilingData.get_outputSize() << std::endl;
  info << "rowsInUb: " << tilingData.get_rowsInUb() << std::endl;
  info << "perCoreHandleCol: " << tilingData.get_perCoreHandleCol() << std::endl;
  info << "perCoreHandleIndices: " << tilingData.get_perCoreHandleIndices() << std::endl;
  info << "perCoreHandleOutputRows: " << tilingData.get_perCoreHandleOutputRows() << std::endl;
  info << "tailCoreHandleOutputRows: " << tilingData.get_tailCoreHandleOutputRows() << std::endl;
  info << "logicCoreNum: " << tilingData.get_logicCoreNum() << std::endl;
  info << "deQuantizeCoreNum: " << tilingData.get_deQuantizeCoreNum() << std::endl;
  info << "indicesUbFactor: " << tilingData.get_indicesUbFactor() << std::endl;
  info << "indicesLoopSize: " << tilingData.get_indicesLoopSize() << std::endl;
  info << "indicesTailUbFactor: " << tilingData.get_indicesTailUbFactor() << std::endl;
  info << "updatesUbFactor: " << tilingData.get_updatesUbFactor() << std::endl;
  info << "updatesLoopSize: " << tilingData.get_updatesLoopSize() << std::endl;
  info << "updatesTailUbFactor: " << tilingData.get_updatesTailUbFactor() << std::endl;
  info << "tailCoreHandleCol: " << tilingData.get_tailCoreHandleCol() << std::endl;
  info << "tailCoreColsLoopSize: " << tilingData.get_tailCoreColsLoopSize() << std::endl;
  info << "tailCoreColsTailUbFactor: " << tilingData.get_tailCoreColsTailUbFactor() << std::endl;
  info << "tailCoreIndicesLoopSize: " << tilingData.get_tailCoreIndicesLoopSize() << std::endl;
  info << "tailCoreIndicesTailUbFactor: " << tilingData.get_tailCoreIndicesTailUbFactor() << std::endl;
  info << "preAxis: " << tilingData.get_preAxis() << std::endl;
  info << "afterAxis: " << tilingData.get_afterAxis() << std::endl;
  info << "rankFusedAxis: " << tilingData.get_rankFusedAxis() << std::endl;
  info << "rankStrideFusedAxis: " << tilingData.get_rankStrideFusedAxis() << std::endl;  
  info << "dequantLoopSize: " << tilingData.get_dequantLoopSize() << std::endl;
  info << "dequantTailUbFactor: " << tilingData.get_dequantTailUbFactor() << std::endl;
  info << "tailCoreDequantLoopSize: " << tilingData.get_tailCoreDequantLoopSize() << std::endl;
  info << "tailCoreDequantTailUbFactor: " << tilingData.get_tailCoreDequantTailUbFactor() << std::endl;
  info << "isIdxSplit: " << tilingData.get_isIdxSplit() << std::endl;
  info << "isDeterministic: " << tilingData.get_isDeterministic() << std::endl;
  info << "isDeterminTemplate: " << tilingData.get_isDeterminTemplate() << std::endl;
  
  OP_LOGI(opName, "Tiling inf is: %s", info.str().c_str());
}

ge::graphStatus TilingScatterNd(gert::TilingContext* context)
{
  ScatterNdTiling tilingObj(context);
  return tilingObj.DoTiling();
}

} // namespace optiling