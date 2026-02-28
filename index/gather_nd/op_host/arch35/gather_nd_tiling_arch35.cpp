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
 * \file gather_nd_tiling_arch35.cpp
 * \brief
 */

#include "log/log.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_key.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "gather_nd_tiling_arch35.h"

using namespace AscendC;
using namespace Ops::Base;

namespace optiling{
// x dtype
constexpr uint64_t INPUT_DTYPE_B8 = 1;
constexpr uint64_t INPUT_DTYPE_B16 = 2;
constexpr uint64_t INPUT_DTYPE_B32 = 4;
constexpr uint64_t INPUT_DTYPE_B64 = 8;

constexpr int64_t DTYPE_B8_KEY = 0;
constexpr int64_t DTYPE_B16_KEY = 1;
constexpr int64_t DTYPE_B32_KEY = 2;
constexpr int64_t DTYPE_B64_KEY = 3;

constexpr size_t GATHER_ND_X_IDX = 0;
constexpr size_t GATHER_ND_INDICES_IDX = 1;
constexpr uint32_t DCACHE_SIZE = 32768;
constexpr uint32_t INDICES_REDUENT_SIZE = 512;
constexpr uint32_t UB_SIZE_ALIGN = 512;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t MIN_INDICES_NUM = 640;
constexpr uint64_t DIM_8 = 8;

const static uint64_t SIMD_THRES = 2048;
const static uint64_t BUFFER_NUM = 2;
const static uint64_t INDICES_SIZE = 16 * 1024UL;
constexpr int32_t RESERVE_BUFFER_SIZE = 1024;

const static uint64_t SIMD_TILING_KEY = 11111111;

#ifdef DAVID_FPGA
constexpr uint32_t SINGLE_CORE_MIN_NUM = 128;
#else
constexpr uint32_t SINGLE_CORE_MIN_NUM = 1024;
#endif
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16777216;

const static int32_t MIN_OUT_UB_SIZE = 16 * 1024;
const static int64_t NUM_TWO = 2;
const static uint64_t K_THRES = 3;
const static uint64_t SIMD_GA_ALL_LOAD_V_GATHER_BASE_TILING_KEY = 4000UL;
const static int32_t HELP_BUFFER_SIZE= 1024;
const static uint32_t NEG_INDICES_SUPPORT_BASE_KEY = 100U;
const static int64_t SIMD_VECTOR_REG= 256;
const static int32_t MIN_INDICES_UB_SIZE = 1024;
const static uint32_t WARP_THREAD_NUM = 32;
const static int32_t GATHER_ENABLE_SIZE = 512;


bool GatherNdSimtTiling::IsCapable() {
  return true;
}

ge::graphStatus GatherNdSimtTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = static_cast<const GatherNdCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
        return ge::GRAPH_FAILED);
    coreNum_ = static_cast<int64_t>(compileInfoPtr->core_num);
    ubSize_ = static_cast<int64_t>(compileInfoPtr->ub_size);
    OP_LOGD("GatherNdSimt", "Get aivNum form compileInfo is: %ld, ubSize: %ld", coreNum_, ubSize_);
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
    OP_LOGD("GatherNdSimt", "Get aivNum form ascendcPlatform is: %ld, ubSize: %ld", coreNum_, ubSize_);
  }
  OP_CHECK_IF((coreNum_ <= 0 || ubSize_ <= 0),
      OP_LOGE("GatherNdSimt",
      "coreNum and ubSize should not be samller than 0, but got coreNum [%lu] and ubSize [%lu], please check.",
      coreNum_, ubSize_), return ge::GRAPH_FAILED);

  ubSize_ = ubSize_ - DCACHE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherNdSimtTiling::GetIndicesShapeInfo() {
  // Indices shape
  auto indicesDtype = context_->GetInputDesc(GATHER_ND_INDICES_IDX)->GetDataType();
  if (indicesDtype == ge::DT_INT64) {
    indicesDtypeSize_ = INPUT_DTYPE_B64;
  } else if (indicesDtype == ge::DT_INT32) {
    indicesDtypeSize_ = INPUT_DTYPE_B32;
  } else {
    OP_LOGE("GatherNdSimt", "indices dtype error!");
  }

  const gert::Shape orgIndicesGeShape = context_->GetInputShape(GATHER_ND_INDICES_IDX)->GetStorageShape();
  uint64_t curIndicesDimLen = orgIndicesGeShape.GetDimNum();
  if (curIndicesDimLen == static_cast<uint64_t>(0)) {
    OP_LOGE("GatherNdSimt", "indices shape is empty!");
  }

  rank_ = orgIndicesGeShape.GetDim(curIndicesDimLen - 1);
  indicesShape_[1] = rank_;
  for (uint64_t i = static_cast<uint64_t>(0); i < curIndicesDimLen - static_cast<uint64_t>(1); i++) {
    OP_LOGD("GatherNdSimt", "IndicesShape: indices axis is %ld, shape value is %ld", i, orgIndicesGeShape.GetDim(i));
    indicesNum_ = indicesNum_ * orgIndicesGeShape.GetDim(i);
  }
  indicesShape_[0] = indicesNum_;
  // check zero shape
  isZeroShape_ = isZeroShape_ || (indicesNum_ == static_cast<uint64_t>(0));
  OP_LOGD("GatherNdSimt", "IndicesShape: dim0 %ld, dim1 %ld", indicesShape_[0], indicesShape_[1]);
  return ge::GRAPH_SUCCESS;
}

void GatherNdSimtTiling::CalcSimdTiling() {
  uint64_t blockFactor = indicesNum_ / coreNum_;
  uint64_t tailBlockFactor = indicesNum_ - blockFactor * coreNum_;
  uint64_t ubBlockSize = static_cast<uint64_t>(GetUbBlockSize(context_));
  uint64_t ubAviable = static_cast<uint64_t>((ubSize_ - INDICES_SIZE) / ubBlockSize * ubBlockSize / xDtypeSize_ / BUFFER_NUM);
  uint64_t needCoreNum = tailBlockFactor;
  if (blockFactor > static_cast<uint64_t>(0)) {
      needCoreNum = coreNum_;
  }
  tilingData_.set_needCoreNum(needCoreNum);
  tilingData_.set_blockFactor(blockFactor);
  tilingData_.set_tailBlockFactor(tailBlockFactor);
  tilingData_.set_maxElement(ubAviable);
  tilingData_.set_indicesUbSize(INDICES_SIZE);
  tilingData_.set_dtypeSize(xDtypeSize_);
  tilingData_.set_blockNum(needCoreNum);
}

ge::graphStatus GatherNdSimtTiling::GetXShapeInfo() {
  // x
  auto xDtype = context_->GetInputDesc(GATHER_ND_X_IDX)->GetDataType();
  if ((xDtype == ge::DT_INT64) || (xDtype == ge::DT_UINT64) || (xDtype == ge::DT_DOUBLE)) {
    xDtypeSize_ = INPUT_DTYPE_B64;
  } else if ((xDtype == ge::DT_INT32) || (xDtype == ge::DT_FLOAT) || (xDtype == ge::DT_UINT32)) {
    xDtypeSize_ = INPUT_DTYPE_B32;
  } else if ((xDtype == ge::DT_FLOAT16) || (xDtype == ge::DT_BF16) || (xDtype == ge::DT_INT16) || (xDtype == ge::DT_UINT16)) {
    xDtypeSize_ = INPUT_DTYPE_B16;
  } else if ((xDtype == ge::DT_INT8) || (xDtype == ge::DT_UINT8) ||(xDtype == ge::DT_BOOL)) {
    xDtypeSize_ = INPUT_DTYPE_B8;
  } else {
    OP_LOGE("GatherNdSimt", "x dtype error!");
    return ge::GRAPH_FAILED;
  }

  const gert::Shape orgXGeShape = context_->GetInputShape(GATHER_ND_X_IDX)->GetStorageShape();
  int64_t curXDimLen = orgXGeShape.GetDimNum();
  if (curXDimLen == 0) {
    OP_LOGE("GatherNdSimt", "x shape is empty!");
    return ge::GRAPH_FAILED;
  }
  for (int64_t i = 0; i < int64_t(rank_); i++) {
    OP_LOGD("GatherNdSimt", "XShape: x axis is %ld, shape value is %ld", i, orgXGeShape.GetDim(i));
    xShape_[i] = orgXGeShape.GetDim(i);
    gatherDimSize_ = gatherDimSize_ * orgXGeShape.GetDim(i);
    isZeroShape_ = isZeroShape_ || (xShape_[i] == static_cast<uint64_t>(0));
  }
  for (int64_t i = static_cast<int64_t>(rank_); i < curXDimLen; i++) {
    OP_LOGD("GatherNdSimt", "IndicesShape: indices axis is %ld, shape value is %ld", i, orgXGeShape.GetDim(i));
    gatherSize_ = gatherSize_ * orgXGeShape.GetDim(i);
    isZeroShape_ = isZeroShape_ || (gatherSize_ == static_cast<uint64_t>(0));
  }
  xShape_[rank_] = gatherSize_;
  return ge::GRAPH_SUCCESS;
}

bool GatherNdSimtTiling::XDtypeImprove() {
  uint64_t lastAxisByte = gatherSize_ * xDtypeSize_;
  if ((xDtypeSize_ < INPUT_DTYPE_B64) && ((lastAxisByte % INPUT_DTYPE_B64) == 0)) {
    OP_LOGD("GatherNdSimt", "XDtypeImprove lastAxisByte %ld, improve to INPUT_DTYPE_B64", lastAxisByte);
    improveDtypeSize_ = INPUT_DTYPE_B64;
    return true;
  }

  if ((xDtypeSize_ < INPUT_DTYPE_B32) && ((lastAxisByte % INPUT_DTYPE_B32) == 0)) {
    OP_LOGD("GatherNdSimt", "XDtypeImprove lastAxisByte %ld, improve to INPUT_DTYPE_B32", lastAxisByte);
    improveDtypeSize_ = INPUT_DTYPE_B32;
    return true;
  }

  if ((xDtypeSize_ < INPUT_DTYPE_B16) && ((lastAxisByte % INPUT_DTYPE_B16) == 0)) {
    OP_LOGD("GatherNdSimt", "XDtypeImprove lastAxisByte %ld, improve to INPUT_DTYPE_B16", lastAxisByte);
    improveDtypeSize_ = INPUT_DTYPE_B16;
    return true;
  }
  return false;
}

ge::graphStatus GatherNdSimtTiling::GetShapeAttrsInfo() {
  auto getResult = GetIndicesShapeInfo();
  if (getResult != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  getResult = GetXShapeInfo();
  if (getResult != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  // kernel check indicesShape[0] == 0, zero shape condition, indicesShape[1] == 0 broadcast condition
  if (isZeroShape_) {
    indicesShape_[0] = static_cast<uint64_t>(0);
    indicesNum_ = static_cast<uint64_t>(0);
  }

    // attr
  negativeIndexSupport_ = *(context_->GetAttrs()->GetAttrPointer<bool>(0));
  if (negativeIndexSupport_) {
    tilingData_.set_negativeIndexSupport(1U);
  } else {
    tilingData_.set_negativeIndexSupport(0U);
  }
  tilingData_.set_supportOutOfBoundIndex(0U);

  // clean redunt x shape
  for (int64_t i = static_cast<int64_t>(rank_ + static_cast<uint64_t>(1)); i < static_cast<int64_t>(GATHER_ND_X_DIM_LEN); i++) {
    xShape_[i] = static_cast<uint64_t>(1);
  }

  uint64_t startStride = static_cast<uint64_t>(1);
  if (rank_ >= static_cast<uint64_t>(1) && rank_ <= DIM_8) {
    for (int32_t i = static_cast<int32_t>(rank_ - static_cast<uint64_t>(1)); i >= 0; i--) {
      strideShape_[i] = startStride;
      startStride *= xShape_[i];
    }
  }

  outputSize_ = indicesNum_ * gatherSize_;

  // to improve performance
  if ((!isZeroShape_) && (rank_ > static_cast<uint64_t>(0)) && !negativeIndexSupport_ && XDtypeImprove()){
    OP_LOGD("GatherNdSimt", "XDtypeImprove before gatherSize_ %ld, xDtypeSize_ is %ld",
            gatherSize_, xDtypeSize_);
    gatherSize_ = gatherSize_ / (improveDtypeSize_ / xDtypeSize_);
    xShape_[rank_] = gatherSize_;
    outputSize_ = indicesNum_ * gatherSize_;
    xDtypeSize_ = improveDtypeSize_;
    OP_LOGD("GatherNdSimt", "XDtypeImprove after gatherSize_ %ld, xDtypeSize_ is %ld",
            gatherSize_, xDtypeSize_);
  }

  tilingData_.set_indicesNum(indicesNum_);
  tilingData_.set_gatherSize(gatherSize_);
  tilingData_.set_outputSize(outputSize_);
  tilingData_.set_xShape(xShape_);
  tilingData_.set_strideShape(strideShape_);
  tilingData_.set_indicesShape(indicesShape_);
  tilingData_.set_rank(rank_);

  return ge::GRAPH_SUCCESS;
}


bool GatherNdSimtTiling::IsGaAllLoad() {
  int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context_));
  int64_t ubSize = static_cast<int64_t>(ubSize_) + static_cast<int64_t>(DCACHE_SIZE);
  int64_t aSize = static_cast<int64_t>(gatherSize_) * static_cast<int64_t>(xDtypeSize_);
  int64_t gSize = static_cast<int64_t>(gatherDimSize_);
  int64_t aSizeAligned = (aSize + ubBlockSize - 1) / ubBlockSize * ubBlockSize;
  if (aSizeAligned == SIMD_VECTOR_REG || aSizeAligned == SIMD_VECTOR_REG / NUM_TWO) {
    aSizeAligned += ubBlockSize;
  }
  int64_t gaSizeInUb = aSizeAligned * (gSize + 1);
  int64_t indicesSize = static_cast<int64_t>(indicesNum_) * static_cast<int64_t>(rank_);

  // 单核处理的索引数量要大于索引范围
  if (indicesNum_ / coreNum_ <= gatherDimSize_) {
    return false;
  }

  if (aSizeAligned > MIN_OUT_UB_SIZE) {
    return false;
  }

  // 重复度小于2时，尾轴大小限制在32~2048
  if ((gatherSize_ <= WARP_THREAD_NUM || gatherSize_ >= SIMD_THRES) && indicesNum_ / coreNum_ / gatherDimSize_ < static_cast<uint32_t>(NUM_TWO)) {
    return false;
  }

  // k轴 <= 3
  if (rank_ > K_THRES) {
    return false;
  }

  // int64索引不做
  if (indicesDtypeSize_ != INPUT_DTYPE_B32) {
    return false;
  }

  int64_t minIndicesSize = std::min(indicesSize, static_cast<int64_t>(MIN_INDICES_UB_SIZE / indicesDtypeSize_));
  int64_t remainSpace = ubSize - static_cast<int64_t>(HELP_BUFFER_SIZE) - NUM_TWO * static_cast<int64_t>(MIN_OUT_UB_SIZE) - 
    minIndicesSize * static_cast<int64_t>(indicesDtypeSize_) - gaSizeInUb; // res = 214k = 219136
  return remainSpace >= 0;
}


ge::graphStatus GatherNdSimtTiling::GaAllLoadTiling() {
  int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context_));
  ubSize_ += DCACHE_SIZE;
  int64_t aSize = static_cast<int64_t>(gatherSize_) * static_cast<int64_t>(xDtypeSize_);
  int64_t gSize = static_cast<int64_t>(gatherDimSize_);
  int64_t totalCoreNum = static_cast<int64_t>(coreNum_);
  int64_t aSizeAligned = (aSize + ubBlockSize - 1) / ubBlockSize * ubBlockSize;
  if (aSizeAligned == SIMD_VECTOR_REG || aSizeAligned == SIMD_VECTOR_REG / NUM_TWO) {  // 规避gather bank 冲突
    aSizeAligned += ubBlockSize;
  }
  
  int64_t gaSizeInUb = aSizeAligned * (gSize + 1);  // 需要多一块做清零
  
  gaAllLoadTilingdata_.set_xShape(xShape_);
  gaAllLoadTilingdata_.set_rank(rank_);
  gaAllLoadTilingdata_.set_gSize(gSize);
  gaAllLoadTilingdata_.set_aSize(aSize);
  gaAllLoadTilingdata_.set_aSizeAligned(aSizeAligned);
  gaAllLoadTilingdata_.set_indicesSize(indicesNum_);

  ubSize_ -= static_cast<uint64_t>(HELP_BUFFER_SIZE);

  // 先分核
  int64_t indicesOuter = std::max(int64_t(1), totalCoreNum);
  int64_t indicesInner = CeilDiv(static_cast<int64_t>(indicesNum_), indicesOuter);
  indicesOuter = CeilDiv(static_cast<int64_t>(indicesNum_), indicesInner);
  int64_t indicesTail = static_cast<int64_t>(indicesNum_) -  indicesInner * (indicesOuter - static_cast<int64_t>(1));

  int64_t needCoreNum = indicesOuter;
  gaAllLoadTilingdata_.set_usedCoreNum(needCoreNum);
  gaAllLoadTilingdata_.set_indicesOuter(indicesOuter);
  gaAllLoadTilingdata_.set_normalCoreIndicesNum(indicesInner);
  gaAllLoadTilingdata_.set_tailCoreIndicesNum(indicesTail);

  // 再分Ub
  int64_t xBufferSizeLoopGa = 0;
  int64_t indicesBufferSizeLoopGa = 0;
  int64_t yBufferSizeLoopGa = 0;

  int64_t allLoadXBytes = CeilAlign(gaSizeInUb, ubBlockSize);
  
  xBufferSizeLoopGa = allLoadXBytes;
  indicesBufferSizeLoopGa = std::min(indicesInner * ubBlockSize, static_cast<int64_t>(ubSize_ - NUM_TWO * MIN_OUT_UB_SIZE - allLoadXBytes));
  indicesBufferSizeLoopGa = CeilAlign(indicesBufferSizeLoopGa, ubBlockSize);
  yBufferSizeLoopGa = (static_cast<int64_t>(ubSize_) - indicesBufferSizeLoopGa - xBufferSizeLoopGa) / NUM_TWO;
  
  gaAllLoadTilingdata_.set_xBufferSize(xBufferSizeLoopGa);
  gaAllLoadTilingdata_.set_indicesBufferSize(indicesBufferSizeLoopGa);
  gaAllLoadTilingdata_.set_yBufferSize(yBufferSizeLoopGa);

  // vgather
  if (aSizeAligned <= GATHER_ENABLE_SIZE) {
    gaAllLoadBaseTilingKey_ = SIMD_GA_ALL_LOAD_V_GATHER_BASE_TILING_KEY;
  }

  return ge::GRAPH_SUCCESS;
}


ge::graphStatus GatherNdSimtTiling::DoOpTiling() {
  if (isZeroShape_) {
    return DoZeroShapeOpTiling();
  } else if (rank_ == static_cast<uint64_t>(0)) {
    return DoBroadCastOpTiling();
  } else if (IsGaAllLoad()) {
    isGaAllLoad_ = true;
    return GaAllLoadTiling();
  }
  if (gatherSize_ * xDtypeSize_ >= SIMD_THRES && 
    ubSize_ >= BUFFER_NUM * INDICES_SIZE && indicesNum_ >= coreNum_ / BUFFER_NUM) {
    isSimd_ = true;
    CalcSimdTiling();
    return ge::GRAPH_SUCCESS;
  }

  if (rank_ > static_cast<uint64_t>(1) && gatherSize_ * xDtypeSize_ <= SIMD_THRES && indicesNum_ >= MIN_INDICES_NUM) {
    isMixKernel_ = true;
    return DoMixKernelOpTiling();
  }

  return DoNormOpTiling();
}

ge::graphStatus GatherNdSimtTiling::DoZeroShapeOpTiling() {
  OP_LOGD("GatherNdSimt", "DoZeroShapeOpTiling");
  xUbSize_ = ubSize_ - INDICES_REDUENT_SIZE;
  indicesUbSize_ = INDICES_REDUENT_SIZE;
  tilingData_.set_xUbSize(xUbSize_);
  tilingData_.set_indicesUbSize(indicesUbSize_);

  ubFactor_ = static_cast<uint64_t>(0);
  blockFactor_ = static_cast<uint64_t>(0);
  blockNum_ = static_cast<uint64_t>(1);
  tilingData_.set_ubFactor(ubFactor_);
  tilingData_.set_blockFactor(blockFactor_);
  tilingData_.set_blockNum(blockNum_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherNdSimtTiling::DoBroadCastOpTiling() {
  OP_LOGD("GatherNdSimt", "DoBroadCastOpTiling");
  uint64_t ubCopySize = ubSize_ / static_cast<uint64_t>(2) / static_cast<uint64_t>(UB_SIZE_ALIGN) * static_cast<uint64_t>(UB_SIZE_ALIGN);
  xUbSize_ = ubCopySize;
  indicesUbSize_ = ubCopySize;
  tilingData_.set_xUbSize(xUbSize_);
  tilingData_.set_indicesUbSize(indicesUbSize_);

  ubFactor_ = std::min(static_cast<uint64_t>(xUbSize_ / xDtypeSize_), gatherSize_);
  blockFactor_ = (indicesNum_ + coreNum_ - static_cast<uint64_t>(1)) / coreNum_;
  blockNum_ = (indicesNum_ + blockFactor_ - static_cast<uint64_t>(1)) / blockFactor_;
  tilingData_.set_ubFactor(ubFactor_);
  tilingData_.set_blockFactor(blockFactor_);
  tilingData_.set_blockNum(blockNum_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherNdSimtTiling::DoNormOpTiling() {
  OP_LOGD("GatherNdSimt", "DoNormOpTiling");
  uint64_t xRatio = gatherSize_ * xDtypeSize_;
  uint64_t indicesRatio = rank_ * indicesDtypeSize_;
  OP_LOGD("GatherNdSimt", "xRatio is %ld, indicesRatio is %ld", xRatio, indicesRatio);
  uint64_t gatherCount = (static_cast<uint64_t>(ubSize_) - static_cast<uint64_t>(INDICES_REDUENT_SIZE)) / (xRatio + indicesRatio);
  OP_LOGD("GatherNdSimt", "gatherCount is %ld, ubSize is %lu", gatherCount, ubSize_);

  if (gatherCount == static_cast<uint64_t>(0)) {
    xUbSize_ = ubSize_ - INDICES_REDUENT_SIZE;
    indicesUbSize_ = INDICES_REDUENT_SIZE;
  } else {
    xUbSize_ = gatherCount * xRatio;
    indicesUbSize_ = gatherCount * indicesRatio;
    xUbSize_ = xUbSize_ / UB_SIZE_ALIGN * UB_SIZE_ALIGN;
    indicesUbSize_ = (indicesUbSize_ + INDICES_REDUENT_SIZE) / UB_SIZE_ALIGN * UB_SIZE_ALIGN;
  }

  OP_LOGD("GatherNdSimt", "xUbSize is %ld, indicesUbSize is %ld", xUbSize_, indicesUbSize_);
  tilingData_.set_xUbSize(xUbSize_);
  tilingData_.set_indicesUbSize(indicesUbSize_);

  OP_LOGD("GatherNdSimt", "xUbSize_ * coreNum_ is %ld, outputSize is %ld",
          xUbSize_ * coreNum_, outputSize_);
  if (xUbSize_ * coreNum_ < outputSize_) {
    ubFactor_ = std::min(static_cast<uint64_t>(xUbSize_ / xDtypeSize_), outputSize_);
  } else {
    ubFactor_ = (outputSize_ + coreNum_ - static_cast<uint64_t>(1)) / coreNum_;
    ubFactor_ = (ubFactor_ + BLOCK_SIZE - static_cast<uint64_t>(1)) / BLOCK_SIZE * BLOCK_SIZE;
    ubFactor_ = std::max(ubFactor_, static_cast<uint64_t>(SINGLE_CORE_MIN_NUM / xUbSize_));
    ubFactor_ = std::min(ubFactor_, static_cast<uint64_t>(xUbSize_ / xDtypeSize_));
  }

  if (ubFactor_ == static_cast<uint64_t>(0)) {
    OP_LOGE("GatherNdSimt", "ubFactor_ is zero!");
    return ge::GRAPH_FAILED;
  }
  uint64_t blockCounter = (outputSize_ + ubFactor_ - static_cast<uint64_t>(1)) / ubFactor_;
  OP_LOGD("GatherNdSimt", "ubFactor is %ld, blockCounter is %ld", ubFactor_, blockCounter);
  blockFactor_ = (blockCounter + coreNum_ - static_cast<uint64_t>(1)) / coreNum_;
  blockNum_ = (blockCounter + blockFactor_ - static_cast<uint64_t>(1)) / blockFactor_;
  OP_LOGD("GatherNdSimt", "blockFactor is %ld, blockNum is %ld", blockFactor_, blockNum_);

  tilingData_.set_ubFactor(ubFactor_);
  tilingData_.set_blockFactor(blockFactor_);
  tilingData_.set_blockNum(blockNum_);

  return ge::GRAPH_SUCCESS;
}

// 仅提升 rank > 1  && indicesNum_ >= 640场景
ge::graphStatus GatherNdSimtTiling::DoMixKernelOpTiling() {
  OP_LOGD("GatherNd", "DoMixKernelOpTiling");
  uint64_t xLastAxisBytes = gatherSize_ * xDtypeSize_;
  uint64_t singleIndicesBytes = indicesDtypeSize_ * rank_;

  uint64_t availableSingleBuf =  (ubSize_ - static_cast<uint64_t>(HELP_BUFFER_SIZE + RESERVE_BUFFER_SIZE)) / BUFFER_NUM;

  uint64_t singleUbProNum = availableSingleBuf / (xLastAxisBytes + singleIndicesBytes);
  if (singleUbProNum == static_cast<uint64_t>(0)) {
      OP_LOGE("GatherNdSimt", "singleUbProNum is zero!");
      return ge::GRAPH_FAILED;
  }

  uint64_t ubBlockSize = static_cast<uint64_t>(GetUbBlockSize(context_));
  uint64_t yUbSize = (singleUbProNum *  xLastAxisBytes + ubBlockSize - static_cast<uint64_t>(1)) / ubBlockSize * ubBlockSize;
  uint64_t indicesUbSize = (singleUbProNum * singleIndicesBytes + ubBlockSize - static_cast<uint64_t>(1)) / ubBlockSize * ubBlockSize;
  tilingData_.set_xUbSize(yUbSize);
  tilingData_.set_indicesUbSize(indicesUbSize);
  tilingData_.set_singleUbProNum(singleUbProNum);

  uint64_t normalCoreIndicesNum = (indicesNum_ + coreNum_ - static_cast<uint64_t>(1)) / coreNum_;
  uint64_t usedCoreNum = (indicesNum_ + normalCoreIndicesNum - static_cast<uint64_t>(1)) / normalCoreIndicesNum;
  uint64_t tailCoreIndicesNum = indicesNum_ - (usedCoreNum - static_cast<uint64_t>(1)) * normalCoreIndicesNum;
  tilingData_.set_normalCoreIndicesNum(normalCoreIndicesNum);
  tilingData_.set_tailCoreIndicesNum(tailCoreIndicesNum);
  tilingData_.set_blockNum(usedCoreNum);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherNdSimtTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t GatherNdSimtTiling::GetTilingKey() const {
  if (isSimd_) {
    OP_LOGD("GatherNdSimt", "tilingKey is %lu", SIMD_TILING_KEY);
    return SIMD_TILING_KEY;
  } else if (isGaAllLoad_) {
    uint64_t tilingKey = gaAllLoadBaseTilingKey_;
    uint64_t tilingBase = negativeIndexSupport_ ? static_cast<uint64_t>(NEG_INDICES_SUPPORT_BASE_KEY) : static_cast<uint64_t>(0);
    tilingKey += tilingBase;
    return tilingKey;
  }

  if (isMixKernel_) {
    constexpr uint64_t DIGIT_TEN = 10;
    constexpr uint64_t DIGIT_HUNDRED = 100;

    uint64_t units = (indicesNum_ > static_cast<uint64_t>(INT32_MAX) || outputSize_ > static_cast<uint64_t>(INT32_MAX)) ? static_cast<uint64_t>(1) : static_cast<uint64_t>(0);
    uint64_t tens = indicesDtypeSize_;
    uint64_t hundreds = xDtypeSize_;
    uint64_t tilingKeyMixKernel = hundreds * DIGIT_HUNDRED + tens * DIGIT_TEN + units;
    OP_LOGD("GatherNdSimt", "tilingKey is %lu", tilingKeyMixKernel);
    return tilingKeyMixKernel;
  }

  uint32_t indicesKey = (indicesDtypeSize_ == INPUT_DTYPE_B64 || rank_ == static_cast<uint32_t>(0) || isZeroShape_) ?
                        static_cast<uint32_t>(DTYPE_B64_KEY) : static_cast<uint32_t>(DTYPE_B32_KEY);
  uint32_t xKey = DTYPE_B64_KEY;
  if (xDtypeSize_ == INPUT_DTYPE_B32) {
    xKey = static_cast<uint32_t>(DTYPE_B32_KEY);
  } else if (xDtypeSize_ == INPUT_DTYPE_B16) {
    xKey = static_cast<uint32_t>(DTYPE_B16_KEY);
  } else if (xDtypeSize_ == INPUT_DTYPE_B8) {
    xKey = static_cast<uint32_t>(DTYPE_B8_KEY);
  }
  uint32_t supportAbnormalIndexKey = (negativeIndexSupport_ || supportOutOfBoundIndex_) ? static_cast<uint32_t>(1) : static_cast<uint32_t>(0);
  uint32_t sizeAddrType = ((gatherSize_ > static_cast<uint64_t>(INT32_MAX) && indicesNum_ > static_cast<uint64_t>(INT32_MAX)) || outputSize_ > static_cast<uint64_t>(INT32_MAX)) ? 
    static_cast<uint32_t>(1) : static_cast<uint32_t>(0);
  uint32_t templateKey = (rank_ == static_cast<uint64_t>(0) || isZeroShape_) ? static_cast<uint32_t>(1) : static_cast<uint32_t>(0);
  uint32_t rankKey = (rank_ == static_cast<uint64_t>(0) || isZeroShape_) ? static_cast<uint32_t>(0) : static_cast<uint32_t>(1);
  if (rankKey == static_cast<uint32_t>(0)) {
    supportAbnormalIndexKey = static_cast<uint32_t>(0);
    indicesKey = static_cast<uint32_t>(0);
  }
  uint64_t tilingKey = Ops::NN::Optiling::GET_TILINGKEY(indicesKey, xKey, supportAbnormalIndexKey, sizeAddrType, templateKey,
                                     static_cast<uint32_t>(1), static_cast<uint32_t>(1), static_cast<uint32_t>(3),
                                     static_cast<uint32_t>(2), rankKey);
  OP_LOGD("GatherNdSimt", "tilingKey is %lu", tilingKey);
  return tilingKey;
}

ge::graphStatus GatherNdSimtTiling::GetWorkspaceSize() {
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = ASCENDC_TOOLS_WORKSPACE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherNdSimtTiling::PostTiling() {
  if (isGaAllLoad_) {
    context_->SetBlockDim(gaAllLoadTilingdata_.get_usedCoreNum());
    gaAllLoadTilingdata_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                      context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(gaAllLoadTilingdata_.GetDataSize());
  } else {
    context_->SetBlockDim(tilingData_.get_blockNum());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetLocalMemorySize(ubSize_);
  }
  return ge::GRAPH_SUCCESS;
}

}