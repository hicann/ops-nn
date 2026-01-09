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
 * \file gather_v2_tiling.cpp
 * \brief
 */
#include "gather_v2_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_info.h"
#include "common/inc/tiling_base/tiling_key.h"
#include "util/const_util.h"

namespace optiling {

const static int32_t INPUT_X_INDEX = 0;
const static int32_t INPUT_INDICES_INDEX = 1;
const static int32_t INPUT_AXIS_INDEX = 2;
const static int32_t OUTPUT_Y_INDEX = 0;
const static int32_t ATTR_BATCH_DIMS_INDEX = 0;
const static int32_t ATTR_NEG_INDEX_SUPPORT = 1;
const static int64_t SIMD_THRES = 2048;
const static int64_t SIMD_TWO_DIM_THRES = 2048;
const static int64_t SIMD_B8_THRES = 218;
const static int64_t SIMD_B16_THRES = 360;
const static int64_t SIMD_B32_THRES = 620;
const static int64_t BUFFER_NUM = 2;
const static int64_t INDICES_SIZE = 8192;
constexpr int32_t DCACHE_SIZE = 128 * 1024;
#ifdef DAVID_FPGA
const static int64_t SMALL_CASE_THREAD_NUM = 64;
#else
const static int64_t SMALL_CASE_THREAD_NUM = 128;
#endif
const static int32_t NUM_FOUR = 4;
const static int32_t NUM_THREE = 3;
const static int32_t NUM_TWO = 2;
const static int32_t NUM_ONE = 1;
const static int32_t NUM_ZERO = 0;
const static int32_t NUM_HUNDRED = 100;

const static int64_t INPUT_DTYPE_B64 = 8;
const static int64_t INPUT_DTYPE_B32 = 4;
const static int64_t INPUT_DTYPE_B16 = 2;
const static int64_t INPUT_DTYPE_B8 = 1;

const static uint64_t DTYPE_B128_KEY = 4;
const static uint64_t DTYPE_B64_KEY = 3;
const static uint64_t DTYPE_B32_KEY = 2;
const static uint64_t DTYPE_B16_KEY = 1;
const static uint64_t DTYPE_B8_KEY = 0;
const static uint64_t SIMD_TILING_KEY = 1000000099UL;
const static uint64_t SIMD_TWO_DIM_TILING_KEY = 1000000299UL;
const static uint64_t SIMT_TWO_DIM_BASE_KEY = 2000000000UL;
const static uint64_t EMPTY_TILING_KEY = 3000000000UL;
const static int64_t MIN_OUTPUT_FULL_LOAD_SIZE = 2048;
const static int64_t PER_BLOCK_MIN_NUM = 256;
const static int64_t MAX_THREAD_NUM = 2048;
constexpr int32_t DCACHE = 32 * 1024;

const static uint64_t SIMD_LAST_GATHER_BASE_TILING_KEY = 1100000000UL;
const static uint64_t SIMD_GA_ALL_LOAD_BASE_TILING_KEY = 3000UL;
const static uint32_t NEG_INDICES_SUPPORT_BASE_KEY = 100U;
const static int32_t MIN_OUT_UB_SIZE = 16 * 1024;
const static int32_t TILING_SIMT = 0;
const static int32_t TILING_SIMD = 1;
const static int32_t TILING_LAST_GATHER = 2;
const static int32_t TILING_GA_ALL_LOAD = 3;
const static int32_t TILING_AFTER_GDIM = 4;
const static int32_t TILING_SIMD_TWO_DIM = 5;
const static int32_t TILING_SIMT_TWO_DIM = 6;
const static int32_t TILING_EMPTY = 7;
const static int32_t B8_AND_B16_GATHER_UPPER = 65536;
const static int64_t SPLIT_OUT_THRES = 2048;
const static int32_t MIN_INDICES_UB_SIZE = 1024;
const static int32_t WARP_THREAD_NUM = 32;
const static int32_t SIMD_VECTOR_REG= 256;
const static int32_t HELP_BUFFER_SIZE= 256;
const static int32_t DATA_CACHE_SIZE= 512;
const static int64_t MIN_TILING_BITS_SIZE_PER_CORE = 32768; // 4KB
const static int64_t BITS_NUM = 8;

template <typename T1, typename T2>
static inline auto CeilDiv(T1 a, T2 b) -> T1 {
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
static inline auto CeilAlign(T1 a, T2 b) -> T1 {
    return CeilDiv(a, b) * b;
}

template <typename T1, typename T2>
static inline auto min(T1 a, T2 b) -> T1 {
    return a < b ? a : b;
}

template <typename T1, typename T2>
static inline auto max(T1 a, T2 b) -> T1 {
    return a > b ? a : b;
}

static const std::set<ge::DataType> X_SUPPORT_DTYPE = {
  ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8,  ge::DT_INT8,  ge::DT_UINT16,
  ge::DT_INT16, ge::DT_UINT32,  ge::DT_INT32, ge::DT_UINT64, ge::DT_INT64, ge::DT_BOOL,
  ge::DT_COMPLEX64, ge::DT_COMPLEX32, ge::DT_DOUBLE
};

static const std::set<ge::DataType> INDICES_SUPPORT_DTYPE = {
  ge::DT_INT32, ge::DT_INT64
};

inline static bool IsSupportDtype(const std::set<ge::DataType> &supportDtype, const ge::DataType dtype)
{
  return (supportDtype.count(dtype) != 0);
}

void Gatherv2TilingBase::Reset() {
  opName_ = nullptr;
}

ge::graphStatus Gatherv2TilingBase::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo == nullptr) {
    auto compileInfoPtr = reinterpret_cast<const GatherV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                    return ge::GRAPH_FAILED);
    aivNum_ = compileInfoPtr->core_num;
    ubSize_ = compileInfoPtr->ub_size;
    OP_LOGD(opName_, "Get ubSize form compileInfo is: %ld", ubSize_);
    OP_LOGD(opName_, "Get aivNum form compileInfo is: %ld", aivNum_);
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize_ = static_cast<int64_t>(ubSizePlatform);
    OP_LOGD(opName_, "Get ubSize form ascendcPlatform is: %ld", ubSize_);
    OP_LOGD(opName_, "Get aivNum form ascendcPlatform is: %ld", aivNum_);
  }
  aicoreParams_.blockDim = aivNum_;
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus Gatherv2TilingBase::GetXInfoAndCheck() {
  // x
  xDtype_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();
  OP_CHECK_IF(!IsSupportDtype(X_SUPPORT_DTYPE, xDtype_), OP_LOGE(context_->GetNodeName(),
    "The dtype only support float32, float16, bfloat16, int64, uint64, int32, uint32, int16, uint16, int8, uint8, \
bool currently, please check."), return ge::GRAPH_FAILED);

  xShape_ = context_->GetInputShape(INPUT_X_INDEX)->GetStorageShape();

  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus Gatherv2TilingBase::GetIndicesInfoAndCheck() {
  // check dtype
  indicesDtype_ = context_->GetInputDesc(INPUT_INDICES_INDEX)->GetDataType();
  indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
  OP_CHECK_IF(!IsSupportDtype(INDICES_SUPPORT_DTYPE, indicesDtype_), OP_LOGE(context_->GetNodeName(),
    "The dtype only support int32, int64 currently, please check."), return ge::GRAPH_FAILED);

  indicesShape_ = context_->GetInputShape(INPUT_INDICES_INDEX)->GetStorageShape();

  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus Gatherv2TilingBase::GetAxisInfoAndCheck() {
  OP_CHECK_IF(!(Ops::Base::GetConstInt(context_, INPUT_AXIS_INDEX, axis_)),
                  OP_LOGE(context_->GetNodeName(),
                                                  "get const data axis failed"), return false);

  size_t min_x_dim = static_cast<size_t>(axis_ < 0 ? -axis_ : axis_ + 1);
  auto xAxisNum = xShape_.GetDimNum();
  if (xAxisNum == 0 && xShape_.GetShapeSize() != 0) {
    if (indicesShape_.GetShapeSize() != 1) {
      OP_LOGE(context_->GetNodeName(), "When x is scalar, index can have only 1 value, but got %ld value",
        indicesShape_.GetShapeSize());
    }
    xAxisNum = 1;
  }
  if (xAxisNum < min_x_dim) {
    OP_LOGE(context_->GetNodeName(), "Shape must be at least rank %lu, but is rank %lu", min_x_dim, xAxisNum);
    return ge::GRAPH_FAILED;
  }
  if (axis_ < 0) {
    axis_ = xAxisNum + axis_;
  }
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus Gatherv2TilingBase::GetAttrsInfoAndCheck() {
  auto attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const int64_t* batchDimsPtr = attrs->GetAttrPointer<int64_t>(ATTR_BATCH_DIMS_INDEX);
  inputBatchDims_ = *batchDimsPtr;
  batchDims_ = *batchDimsPtr;
  if (batchDims_ != 0) {
    if (batchDims_ < static_cast<int64_t>(-indicesShape_.GetDimNum()) ||
        batchDims_ > static_cast<int64_t>(indicesShape_.GetDimNum())) {
      OP_LOGE(context_->GetNodeName(), "Expected batch_dims in the range [%lu, %lu], but got %ld",
              -indicesShape_.GetDimNum(), indicesShape_.GetDimNum() - 1, batchDims_);
      return ge::GRAPH_FAILED;
    }
    if (batchDims_ < 0) {
      batchDims_ = indicesShape_.GetDimNum() + batchDims_;
    }

    if (batchDims_ >= static_cast<int64_t>(xShape_.GetDimNum())) {
      OP_LOGE(context_->GetNodeName(), "batch_dims (%ld) must be less than rank(x) (%lu).", batchDims_,
              xShape_.GetDimNum());
      return ge::GRAPH_FAILED;
    }
    if (axis_ < batchDims_) {
      OP_LOGE(context_->GetNodeName(), "batch_dims (%ld) must be less than or equal to axis (%ld).", batchDims_, axis_);
      return ge::GRAPH_FAILED;
    }
    for (int32_t i = 0; i < batchDims_; i++) {
      if (xShape_.GetDim(i) != indicesShape_.GetDim(i)) {
        OP_LOGE(context_->GetNodeName(), "x.shape[%d]: %ld, should be equal to indices.shape[%d]: %ld", i,
                xShape_.GetDim(i), i, indicesShape_.GetDim(i));
        return ge::GRAPH_FAILED;
      }
    }
  }

  negativeIndexSupport_ = *(attrs->GetAttrPointer<bool>(ATTR_NEG_INDEX_SUPPORT));
  if (negativeIndexSupport_) {
    gatherV2TilingData_.set_negativeIndexSupport(NUM_ONE);
  } else {
    gatherV2TilingData_.set_negativeIndexSupport(NUM_ZERO);
  }
  gatherV2TilingData_.set_supportOutOfBoundIndex(NUM_ONE);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Gatherv2TilingBase::GetShapeAttrsInfo() {
  opName_ = context_->GetNodeName();
  OP_CHECK_IF(GetXInfoAndCheck() != ge::GRAPH_SUCCESS,
                  OP_LOGE(opName_, "input x check failed."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(GetIndicesInfoAndCheck() != ge::GRAPH_SUCCESS,
                  OP_LOGE(opName_, "input indices check failed."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(GetAxisInfoAndCheck() != ge::GRAPH_SUCCESS,
                  OP_LOGE(opName_, "input axis check failed."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(GetAttrsInfoAndCheck() != ge::GRAPH_SUCCESS,
                 OP_LOGE(opName_, "input attrs check failed."),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

/**
 * marge axis
 * indices: [batch_size, gather_size]
 * x:       [batch_size, outer_size, gather_dim_size, inner_size]
 * out:     [batch_size, outer_size, gather_size(N/batch_size), inner_size]
 */
ge::graphStatus Gatherv2TilingBase::MargeAxis() {
  gatherDimSize_ = xShape_.GetDimNum() == 0 ? 1 : xShape_.GetDim(axis_);
  auto indices = context_->GetInputTensor(INPUT_INDICES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
  int64_t indicesSize = indices->GetShapeSize();

  for (int i = 0; i < batchDims_; i++) {
    batchSize_ *= xShape_.GetDim(i);
  }
  for (int i = batchDims_; i < axis_; i++) {
    outerSize_ *= xShape_.GetDim(i);
  }
  for (size_t i = axis_ + 1; i < xShape_.GetDimNum(); i++) {
    innerSize_ *= xShape_.GetDim(i);
  }

  gatherSize_ = indicesSize / batchSize_;

  innerSize_ = innerSize_ / (XDtypeImprove() / ge::GetSizeByDataType(xDtype_));

  gatherV2TilingData_.set_batchSize(batchSize_);
  gatherV2TilingData_.set_outerSize(outerSize_);
  gatherV2TilingData_.set_gatherDimSize(gatherDimSize_);
  gatherV2TilingData_.set_gatherSize(gatherSize_);
  gatherV2TilingData_.set_innerSize(innerSize_);
  ySize_ = batchSize_ * outerSize_ * gatherSize_ * innerSize_;
  auto x = context_->GetInputTensor(INPUT_X_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, x);
  int64_t xSize = x->GetShapeSize();
  gatherV2TilingData_.set_xSize(xSize);
  gatherV2TilingData_.set_indicesSize(indicesSize);
  gatherV2TilingData_.set_ySize(ySize_);
  return ge::GRAPH_SUCCESS;
}

int64_t Gatherv2TilingBase::XDtypeImprove() {
  int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
  improveDtypeSize_ = xDtypeSize;
  int64_t lastAxisBytes = innerSize_ * xDtypeSize;
  if ((xDtypeSize < INPUT_DTYPE_B64) && (lastAxisBytes % INPUT_DTYPE_B64) == 0) {
    OP_LOGD(opName_, "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B64", lastAxisBytes);
    improveDtypeSize_ = INPUT_DTYPE_B64;
    return INPUT_DTYPE_B64;
  }

  if ((xDtypeSize < INPUT_DTYPE_B32) && (lastAxisBytes % INPUT_DTYPE_B32) == 0) {
    OP_LOGD(opName_, "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B32", lastAxisBytes);
    improveDtypeSize_ = INPUT_DTYPE_B32;
    return INPUT_DTYPE_B32;
  }

  if ((xDtypeSize < INPUT_DTYPE_B16) && (lastAxisBytes % INPUT_DTYPE_B16) == 0) {
    OP_LOGD(opName_, "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B16", lastAxisBytes);
    improveDtypeSize_ = INPUT_DTYPE_B16;
    return INPUT_DTYPE_B16;
  }
  return xDtypeSize;
}

ge::graphStatus Gatherv2TilingBase::CalcEmptyCoreElement()
{
  int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
  int64_t innerSizeWoImprove_ = 1;
  int64_t ySizeWoImprove_ = 1;

  for (size_t i = axis_ + 1; i < xShape_.GetDimNum(); i++) {
    innerSizeWoImprove_ *= xShape_.GetDim(i);
  }

  ySizeWoImprove_ = batchSize_ * outerSize_ * gatherSize_ * innerSizeWoImprove_;
  int64_t ySizeLen_ = ySizeWoImprove_ * xDtypeSize; //后续用int8_t赋0，这里计算字节数即可

  needCoreNum_ = (ySizeLen_ * BITS_NUM + MIN_TILING_BITS_SIZE_PER_CORE - 1) / MIN_TILING_BITS_SIZE_PER_CORE;

  if (needCoreNum_ > aivNum_) {
    needCoreNum_ = aivNum_;
  }

  int64_t perCoreElements = (ySizeLen_ + needCoreNum_ - 1) / needCoreNum_;
  int64_t lastCoreElements = ySizeLen_ - (needCoreNum_ - 1) * perCoreElements;
  
  emptyTilingData_.set_needCoreNum(needCoreNum_);
  emptyTilingData_.set_perCoreElements(perCoreElements);
  emptyTilingData_.set_lastCoreElements(lastCoreElements);
  return ge::GRAPH_SUCCESS;
}

void Gatherv2TilingBase::CalcCoreElement() {
  while ((threadNum_ >= NUM_TWO * SMALL_CASE_THREAD_NUM) && (CeilDiv(ySize_, threadNum_) < (aivNum_ / NUM_TWO))) {
    threadNum_ = threadNum_ / NUM_TWO;
  }
  gatherV2TilingData_.set_threadNum(threadNum_);
  int64_t perCoreElements = CeilDiv(ySize_, aivNum_);
  if (ySize_ < threadNum_) {
    gatherV2TilingData_.set_needCoreNum(1);
    gatherV2TilingData_.set_perCoreElements(ySize_);
    gatherV2TilingData_.set_lastCoreElements(ySize_);
    needCoreNum_ = 1;
    return;
  }
  perCoreElements = (perCoreElements + threadNum_ - 1) / threadNum_ * threadNum_;  // 对齐到threadNum_的倍数
  needCoreNum_ = CeilDiv(ySize_, perCoreElements);
  int64_t lastCoreElements = ySize_ - perCoreElements * (needCoreNum_ - 1);
  gatherV2TilingData_.set_needCoreNum(needCoreNum_);
  gatherV2TilingData_.set_perCoreElements(perCoreElements);
  gatherV2TilingData_.set_lastCoreElements(lastCoreElements);
}

ge::graphStatus Gatherv2TilingBase::CalFullLoadTiling() {
  int64_t perCoreElements;
  int64_t pgDataSize = outerSize_ * gatherSize_;
  int64_t perCoreMinElements = std::max<int64_t>(PER_BLOCK_MIN_NUM / innerSize_ / improveDtypeSize_, static_cast<int64_t>(1));
  if ((outerSize_ * gatherSize_) < aivNum_ * perCoreMinElements) {
    perCoreElements = std::min(perCoreMinElements, pgDataSize);
  } else {
    perCoreElements = (pgDataSize + aivNum_ - 1) / aivNum_;
  }
  needCoreNum_ = (pgDataSize + perCoreElements -1) / perCoreElements;
  int64_t tailBlockData = pgDataSize - (needCoreNum_ -1) * perCoreElements;
  threadNum_ = std::min(perCoreElements * innerSize_, MAX_THREAD_NUM);

  int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
  int64_t ubAviable = (ubSize_ - DCACHE) / ubBlockSize * ubBlockSize / improveDtypeSize_;
  
  gatherV2TilingData_.set_needCoreNum(needCoreNum_);
  gatherV2TilingData_.set_perCoreElements(perCoreElements);
  gatherV2TilingData_.set_lastCoreElements(tailBlockData);
  gatherV2TilingData_.set_maxElement(ubAviable);
  gatherV2TilingData_.set_threadNum(threadNum_);
  gatherV2TilingData_.set_dtypeSize(improveDtypeSize_);
  return ge::GRAPH_SUCCESS;
}

void Gatherv2TilingBase::CalcSimdTiling() {
  int64_t blockFactor = batchSize_ * outerSize_ * gatherSize_ / aivNum_;
  int64_t tailBlockFactor = batchSize_ * outerSize_ * gatherSize_ - blockFactor * aivNum_;
  int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
  int64_t ubAviable = (ubSize_ - INDICES_SIZE) / ubBlockSize * ubBlockSize / improveDtypeSize_ / BUFFER_NUM;
  needCoreNum_ = tailBlockFactor;
  if (blockFactor > 0) {
    needCoreNum_ = aivNum_;
  }

  gatherV2TilingData_.set_needCoreNum(needCoreNum_);
  gatherV2TilingData_.set_perCoreElements(blockFactor);
  gatherV2TilingData_.set_lastCoreElements(tailBlockFactor);
  gatherV2TilingData_.set_maxElement(ubAviable);
  gatherV2TilingData_.set_indiceUbSize(INDICES_SIZE);
  gatherV2TilingData_.set_dtypeSize(improveDtypeSize_);
}

ge::graphStatus Gatherv2TilingBase::SimdTwoDimTiling() {
  int64_t blockFactor = gatherSize_ / aivNum_;
  int64_t tailBlockFactor = gatherSize_ - blockFactor * aivNum_;
  int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
  int64_t ubAviable = (ubSize_ - INDICES_SIZE) / ubBlockSize * ubBlockSize / improveDtypeSize_ / BUFFER_NUM;
  int32_t indiceFactor = INDICES_SIZE / indicesDtypeSize_;
  needCoreNum_ = blockFactor > 0 ? aivNum_ :tailBlockFactor;

  simdTwoDimTilingData_.set_needCoreNum(needCoreNum_);
  simdTwoDimTilingData_.set_negativeIndexSupport(negativeIndexSupport_);
  simdTwoDimTilingData_.set_indiceFactor(indiceFactor);
  simdTwoDimTilingData_.set_dtypeSize(improveDtypeSize_);

  simdTwoDimTilingData_.set_gatherDimSize(gatherDimSize_);
  simdTwoDimTilingData_.set_gatherSize(gatherSize_);
  simdTwoDimTilingData_.set_innerSize(innerSize_);
  simdTwoDimTilingData_.set_blockFactor(blockFactor);
  simdTwoDimTilingData_.set_tailBlockFactor(tailBlockFactor);
  simdTwoDimTilingData_.set_maxElement(ubAviable);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Gatherv2TilingBase::SimtTwoDimTiling() {
  int32_t threadNum = MAX_THREAD_NUM;
  while ((threadNum >= NUM_TWO * SMALL_CASE_THREAD_NUM) && (CeilDiv(ySize_, static_cast<int64_t>(threadNum)) < (aivNum_ / NUM_TWO))) {
    threadNum = threadNum / NUM_TWO;
  }
  simtTwoDimTilingData_.set_threadNum(threadNum);
  simtTwoDimTilingData_.set_negativeIndexSupport(negativeIndexSupport_);
  simtTwoDimTilingData_.set_gatherDimSize(gatherDimSize_);
  simtTwoDimTilingData_.set_innerSize(innerSize_);
  int64_t perCoreElements = CeilDiv(ySize_, aivNum_);
  if (ySize_ < threadNum) {
    simtTwoDimTilingData_.set_needCoreNum(1);
    simtTwoDimTilingData_.set_perCoreElements(ySize_);
    simtTwoDimTilingData_.set_lastCoreElements(ySize_);
    needCoreNum_ = 1;
    return ge::GRAPH_SUCCESS;
  }

  perCoreElements = (perCoreElements + threadNum - 1) / threadNum * threadNum;  // 对齐到threadNum_的倍数
  needCoreNum_ = CeilDiv(ySize_, perCoreElements);
  int64_t lastCoreElements = ySize_ - perCoreElements * (needCoreNum_ - 1);
  simtTwoDimTilingData_.set_needCoreNum(needCoreNum_);
  simtTwoDimTilingData_.set_perCoreElements(perCoreElements);
  simtTwoDimTilingData_.set_lastCoreElements(lastCoreElements);
  return ge::GRAPH_SUCCESS;
}

void Gatherv2TilingBase::ShowBaseTilingData() {
  OP_LOGI(opName_,
          "gatherV2TilingData is needCoreNum: %ld, threadNum is: %ld, batchSize: %ld, outerSize: %ld, gatherDimSize: "
          "%ld, gatherSize: %ld, innerSize: %ld, xSize: %ld, ySize: %ld, indicesSize: %ld, perCoreElements: "
          "%ld, lastCoreElements: %ld, negativeIndexSupport: %ld, supportOutOfBoundIndex: %ld",
          gatherV2TilingData_.get_needCoreNum(), gatherV2TilingData_.get_threadNum(),
          gatherV2TilingData_.get_batchSize(), gatherV2TilingData_.get_outerSize(),
          gatherV2TilingData_.get_gatherDimSize(), gatherV2TilingData_.get_gatherSize(),
          gatherV2TilingData_.get_innerSize(), gatherV2TilingData_.get_xSize(), gatherV2TilingData_.get_ySize(),
          gatherV2TilingData_.get_indicesSize(), gatherV2TilingData_.get_perCoreElements(),
          gatherV2TilingData_.get_lastCoreElements(), gatherV2TilingData_.get_negativeIndexSupport(),
          gatherV2TilingData_.get_supportOutOfBoundIndex());
}

void Gatherv2TilingBase::ShowLastGtaherSimdTilingData()
{
    std::string str;
    str += " needCoreNum:" + std::to_string(lastTilingdata_.get_needCoreNum());
    str += " indicesNum:" + std::to_string(lastTilingdata_.get_indicesNum());
    str += " splitIndices:" + std::to_string(lastTilingdata_.get_splitIndices());
    str += " inputNum:" + std::to_string(lastTilingdata_.get_inputNum());
    str += " splitMode:" + std::to_string(lastTilingdata_.get_splitMode());
    str += " coreInCols:" + std::to_string(lastTilingdata_.get_coreInCols());
    str += " inputUbSize:" + std::to_string(lastTilingdata_.get_inputUbSize());
    str += " outUbSize:" + std::to_string(lastTilingdata_.get_outUbSize());
    str += " indiceUbSize:" + std::to_string(lastTilingdata_.get_indiceUbSize());
    str += " ubCols:" + std::to_string(lastTilingdata_.get_ubCols());
    str += " ubRows:" + std::to_string(lastTilingdata_.get_ubRows());
    str += " gatherDimSize:" + std::to_string(lastTilingdata_.get_gatherDimSize());
    str += " gatherSize:" + std::to_string(lastTilingdata_.get_gatherSize());
    str += " blockFactor:" + std::to_string(lastTilingdata_.get_blockFactor());
    str += " tailBlockFactor:" + std::to_string(lastTilingdata_.get_tailBlockFactor());
    str += " gFactor:" + std::to_string(lastTilingdata_.get_gFactor());
    OP_LOGI(context_, "%s", str);
}

bool Gatherv2TilingBase::IsLastGatherAndFullLoad() {
  bool isGatherDimFullLoad = (gatherDimSize_ * improveDtypeSize_ <= ubSize_ / NUM_TWO || gatherDimSize_ * improveDtypeSize_ <= ubSize_ - NUM_TWO * MIN_OUT_UB_SIZE);
  bool isLastGather = innerSize_ == 1;
  bool isVGatherSupport = (improveDtypeSize_ <= static_cast<int32_t>(sizeof(int16_t))) ? gatherDimSize_ < B8_AND_B16_GATHER_UPPER : gatherDimSize_ < INT32_MAX;
  bool enableUbGather = batchSize_ == 1 && isLastGather &&  isGatherDimFullLoad &&  isVGatherSupport && gatherSize_ / gatherDimSize_ > 1 
                && gatherSize_ * improveDtypeSize_ >= 32 && improveDtypeSize_ <= static_cast<int32_t>(sizeof(int64_t));
  return enableUbGather;
}

bool Gatherv2TilingBase::IsGaAllLoad() {
  int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
  int64_t indicesDtypeSize = ge::GetSizeByDataType(indicesDtype_);
  int64_t aSize = innerSize_ * improveDtypeSize_;
  int64_t gSize = gatherDimSize_;
  int64_t aSizeAligned = (aSize + ubBlockSize - 1) / ubBlockSize * ubBlockSize;
  if (aSizeAligned == SIMD_VECTOR_REG || aSizeAligned == SIMD_VECTOR_REG / NUM_TWO) {
    aSizeAligned += ubBlockSize;
  }
  int64_t gaSizeInUb = aSizeAligned * (gSize + 1);
  if (aSize <= WARP_THREAD_NUM || aSize >=SIMD_THRES) {
    return false;
  }

  if (aSizeAligned * gSize <= DATA_CACHE_SIZE && improveDtypeSize_ == INPUT_DTYPE_B64) {
    return false;
  }

  int64_t minIndicesSize = std::min(gatherSize_, static_cast<int64_t>(MIN_INDICES_UB_SIZE / indicesDtypeSize));
  int64_t remainSpace = ubSize_ - HELP_BUFFER_SIZE - NUM_FOUR * MIN_OUT_UB_SIZE - minIndicesSize * indicesDtypeSize - gaSizeInUb;
  return batchSize_ == 1 && gatherSize_ / gatherDimSize_ > 1 && remainSpace >= 0;
}

bool Gatherv2TilingBase::IsAfterGdimFullLoad() {
  bool isInputAfterGdim = gatherDimSize_ * innerSize_ * improveDtypeSize_ < (ubSize_-DCACHE);
  bool isAllInputSize = (batchSize_ * outerSize_ * gatherDimSize_ * innerSize_) < INT32_MAX;
  bool isAllOutputSize = ySize_ < INT32_MAX;
  bool isMinOutputFullLoad = ySize_ * improveDtypeSize_ > MIN_OUTPUT_FULL_LOAD_SIZE;
  bool isInputAndOutputRepeat = (gatherSize_ / gatherDimSize_) > 1;
  bool enableFullLoad = (isInputAfterGdim && isAllInputSize && isMinOutputFullLoad && isInputAndOutputRepeat && isAllOutputSize 
       && batchSize_ == 1 && outerSize_ == 1 && inputBatchDims_ == 0 && innerSize_ * improveDtypeSize_< MAX_THREAD_NUM);
  return enableFullLoad;
}

bool Gatherv2TilingBase::IsSimdTwoDim() {
  bool isTwoDim = batchSize_ == 1 && outerSize_ == 1;
  bool isSimd = (innerSize_ * improveDtypeSize_ >= SIMD_TWO_DIM_THRES || 
                 (improveDtypeSize_ == INPUT_DTYPE_B8 && innerSize_ * improveDtypeSize_ >= SIMD_B8_THRES) || 
                 (improveDtypeSize_ == INPUT_DTYPE_B16 && innerSize_ * improveDtypeSize_ >= SIMD_B16_THRES) || 
                 (improveDtypeSize_ == INPUT_DTYPE_B32 && innerSize_ * improveDtypeSize_ >= SIMD_B32_THRES)) && 
                batchSize_ * outerSize_ * gatherSize_ >= aivNum_ / NUM_TWO;
  return isTwoDim && isSimd;
}

bool Gatherv2TilingBase::IsSimtTwoDim() {
  bool isTwoDim = batchSize_ == 1 && outerSize_ == 1;
  return isTwoDim;
}

ge::graphStatus Gatherv2TilingBase::DoOpTiling() {
  ubBlockSize_ = static_cast<int32_t>(Ops::Base::GetUbBlockSize(context_));
  vRegSize_ = static_cast<int32_t>(Ops::Base::GetVRegSize(context_));
  if (MargeAxis() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (gatherDimSize_ == 0) {
    tilingMode_ = TILING_EMPTY;
    return CalcEmptyCoreElement();
  }
  if (IsSimdTwoDim()) {
    tilingMode_ = TILING_SIMD_TWO_DIM;
    return SimdTwoDimTiling();
  }
  if (IsLastGatherAndFullLoad()) {
    tilingMode_ = TILING_LAST_GATHER;
    return LastGatherTiling();
  }
  if (IsGaAllLoad()) {
    tilingMode_ = TILING_GA_ALL_LOAD;
    return GaAllLoadTiling();
  }
  if (IsSimtTwoDim()) {
    tilingMode_ = TILING_SIMT_TWO_DIM;
    return SimtTwoDimTiling();
  }
  if (IsAfterGdimFullLoad()) {
    tilingMode_ = TILING_AFTER_GDIM;
    return CalFullLoadTiling();
  }
  if (((improveDtypeSize_ == INPUT_DTYPE_B8 && innerSize_ * improveDtypeSize_ >= SIMD_B8_THRES) || 
       (improveDtypeSize_ == INPUT_DTYPE_B16 && innerSize_ * improveDtypeSize_ >= SIMD_B16_THRES) || 
       (improveDtypeSize_ == INPUT_DTYPE_B32 && innerSize_ * improveDtypeSize_ >= SIMD_B32_THRES) || 
       innerSize_ * improveDtypeSize_ >= SIMD_THRES) && 
      ubSize_ >= NUM_TWO * INDICES_SIZE && batchSize_ * outerSize_ * gatherSize_ >= aivNum_ / NUM_TWO) {
    tilingMode_ = TILING_SIMD;
    CalcSimdTiling();
  } else {
    tilingMode_ = TILING_SIMT;
    CalcCoreElement();
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Gatherv2TilingBase::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t Gatherv2TilingBase::GetTilingKey() const {
  uint64_t tilingKey = 0UL;
  if (tilingMode_ == TILING_SIMD_TWO_DIM) {
    tilingKey = SIMD_TWO_DIM_TILING_KEY;
  } else if (tilingMode_ == TILING_SIMD) {
    tilingKey = SIMD_TILING_KEY;
  } else if (tilingMode_ == TILING_LAST_GATHER) {
    tilingKey = SIMD_LAST_GATHER_BASE_TILING_KEY + improveDtypeSize_;
    tilingKey += negativeIndexSupport_ ? NEG_INDICES_SUPPORT_BASE_KEY : 0;
  } else if (tilingMode_ == TILING_GA_ALL_LOAD) {
    tilingKey = SIMD_GA_ALL_LOAD_BASE_TILING_KEY;
    tilingKey += negativeIndexSupport_ ? NEG_INDICES_SUPPORT_BASE_KEY : 0;
  } else if (tilingMode_ == TILING_SIMT_TWO_DIM) {
    tilingKey = SIMT_TWO_DIM_BASE_KEY + static_cast<uint64_t>(improveDtypeSize_);
    bool needInt64Addr = static_cast<uint64_t>((gatherDimSize_ * innerSize_) > INT32_MAX
        || ySize_ > INT32_MAX);
    if (needInt64Addr) {
      tilingKey += static_cast<uint64_t>(NUM_HUNDRED);
    }
  } else if (tilingMode_ == TILING_EMPTY) {
    tilingKey = EMPTY_TILING_KEY;
  } else {
    uint64_t sizeAddrType;
    uint64_t xKey = DTYPE_B8_KEY;
    if (improveDtypeSize_ == INPUT_DTYPE_B64) {
      xKey = DTYPE_B64_KEY;
    } else if (improveDtypeSize_ == INPUT_DTYPE_B32) {
      xKey = DTYPE_B32_KEY;
    } else if (improveDtypeSize_ == INPUT_DTYPE_B16) {
      xKey = DTYPE_B16_KEY;
    }
    if(tilingMode_ == TILING_AFTER_GDIM){
      sizeAddrType = NUM_TWO; 
    } else {
      sizeAddrType = static_cast<uint64_t>(((batchSize_ * outerSize_ * gatherDimSize_ * innerSize_) > INT32_MAX
                            || ySize_ > INT32_MAX) ? 1 : 0);
    }
    tilingKey = Ops::NN::Optiling::GET_TILINGKEY(sizeAddrType, xKey);
  }

  OP_LOGD(opName_, "tilingKey is %lu", tilingKey);
  return tilingKey;
}

void Gatherv2TilingBase::CalcMaxUbcolAndIndiceFactor(int32_t &ubCols, int32_t ubRows, int32_t ubSize, int32_t minCols, int32_t inputNum, int32_t outputNum, int32_t indiceNUm, int32_t castUbRatio) {
    ubCols = minCols;
    int32_t castTargetDtypeSize = static_cast<int32_t>(sizeof(int32_t));
    while(ubRows *  (inputNum * improveDtypeSize_ * gatherDimSize_ + outputNum * ubCols * improveDtypeSize_) + indiceNUm * ubCols * indicesDtypeSize_ + castUbRatio * ubCols * castTargetDtypeSize <= ubSize) {
        ubCols += minCols;
    }
    ubCols -= minCols;
}

void Gatherv2TilingBase::LastGatherUbTiling(int32_t &inputUbSize, int32_t &indiceUbSize, int32_t &outUbSize, int32_t &indiceCastUbSize, int32_t &inputNum, int32_t &indicesNum, int32_t &ubCols, int32_t &ubRows, int64_t blockFactor, int64_t gFactor) {
  int32_t minCols = ((improveDtypeSize_ == static_cast<int32_t>(sizeof(int64_t))) ? NUM_TWO * vRegSize_ : vRegSize_) / improveDtypeSize_;
  int32_t indicesUbAlign = (improveDtypeSize_ <= static_cast<int32_t>(sizeof(int16_t))) ? NUM_TWO * vRegSize_ : vRegSize_;
  indicesUbAlign = (indicesDtypeSize_ == static_cast<int32_t>(sizeof(int64_t))) ? NUM_TWO * indicesUbAlign : indicesUbAlign; // int64使用traitTwo模式
  //int64索引 cast为int32需要额外空间
  int32_t castUbRatio = (indicesDtypeSize_ == static_cast<int32_t>(sizeof(int64_t))) ? 1 : 0;
  int32_t castTargetDtypeSize = static_cast<int32_t>(sizeof(int32_t));
  // 通用场景， 都开db
  if(NUM_TWO * ((gatherDimSize_ * improveDtypeSize_ + ubBlockSize_) + minCols * improveDtypeSize_ + (minCols * indicesDtypeSize_ + ubBlockSize_) 
      + MIN_OUT_UB_SIZE) + (castUbRatio * minCols * castTargetDtypeSize + indicesUbAlign)  < ubSize_  && blockFactor > 1) {
      inputNum = NUM_TWO;
      int32_t tmpUbsize = ubSize_  - (inputNum + NUM_TWO) * ubBlockSize_ - indicesUbAlign;
      ubRows = ((tmpUbsize - castUbRatio * minCols * castTargetDtypeSize) / NUM_TWO - minCols * indicesDtypeSize_) / (improveDtypeSize_ * gatherDimSize_ + minCols * improveDtypeSize_);
      ubRows = std::min(static_cast<int64_t>(ubRows), blockFactor);
      ubCols = minCols;
      CalcMaxUbcolAndIndiceFactor(ubCols, ubRows, tmpUbsize, minCols, inputNum, indicesNum, NUM_TWO, castUbRatio);
      while (ubRows *  ubCols * improveDtypeSize_ < MIN_OUT_UB_SIZE && ubRows >= NUM_TWO && ubCols < gFactor) {
          ubRows -= 1;
          CalcMaxUbcolAndIndiceFactor(ubCols, ubRows, tmpUbsize, minCols, inputNum, indicesNum, NUM_TWO, castUbRatio);
      }       
  } else {
      // 输入gather轴较大无法开db
      inputNum = 1;
      int32_t tmpUbsize = ubSize_  - (inputNum + NUM_TWO + indicesNum) * ubBlockSize_ - indicesUbAlign;
      ubRows = (tmpUbsize - NUM_TWO * minCols * indicesDtypeSize_ - castUbRatio * minCols * castTargetDtypeSize) / improveDtypeSize_ / (gatherDimSize_ + NUM_TWO * minCols);
      ubRows = std::min(static_cast<int64_t>(ubRows), blockFactor);
      ubCols = minCols;
      CalcMaxUbcolAndIndiceFactor(ubCols, ubRows, tmpUbsize, minCols, inputNum, indicesNum, NUM_TWO, castUbRatio);
      while (ubRows *  ubCols * improveDtypeSize_ < MIN_OUT_UB_SIZE && ubRows >= NUM_TWO && ubCols < gFactor) {
          ubRows -= 1;
          CalcMaxUbcolAndIndiceFactor(ubCols, ubRows, tmpUbsize, minCols, inputNum, indicesNum, NUM_TWO, castUbRatio);
      }         
  }
  outUbSize = CeilAlign(ubRows * ubCols * improveDtypeSize_, ubBlockSize_);
  inputUbSize = CeilAlign(ubRows * gatherDimSize_ * improveDtypeSize_, static_cast<int64_t>(ubBlockSize_));
  indiceUbSize = CeilAlign(ubCols * indicesDtypeSize_, ubBlockSize_);
  indiceCastUbSize = CeilAlign(castUbRatio * ubCols * castTargetDtypeSize, indicesUbAlign);
  indiceCastUbSize = std::max(indiceCastUbSize, indicesUbAlign);
  if (ubCols >= gFactor && ubRows * gFactor * improveDtypeSize_ >= SPLIT_OUT_THRES && gFactor > minCols) {
      ubCols = CeilDiv(gFactor,  static_cast<int64_t>(NUM_TWO));
      ubCols = CeilAlign(ubCols, minCols);
 }
}

ge::graphStatus Gatherv2TilingBase::LastGatherTiling() {
  int64_t blockFactor = CeilDiv(outerSize_, aivNum_);
  needCoreNum_ = CeilDiv(outerSize_, blockFactor);
  int64_t tailBlockFactor = outerSize_ - (needCoreNum_ - 1) * blockFactor;
  int64_t gFactor = gatherSize_;
  int64_t minCols = ((improveDtypeSize_ == sizeof(int64_t)) ? NUM_TWO * vRegSize_ : vRegSize_) / improveDtypeSize_;
  int16_t coreInCols = 1;
  if (needCoreNum_ <= aivNum_ / NUM_TWO && gFactor >= gatherDimSize_ && gFactor > minCols && needCoreNum_ != 0) {
      coreInCols = std::min(CeilDiv(gFactor, minCols), aivNum_ / needCoreNum_);
      gFactor = CeilDiv(gFactor, static_cast<int64_t>(coreInCols));
      needCoreNum_ = needCoreNum_ * coreInCols;
  }

  int32_t inputUbSize = 0;
  int16_t splitIndices = 1;
  
  int32_t indiceUbSize = 0;
  int32_t outUbSize = 0;
  int32_t indiceCastUbSize = 0;
  int32_t indicesNum = NUM_TWO;
  int32_t inputNum = NUM_TWO;
  int32_t ubCols = 0;
  int32_t ubRows = 0;
  int16_t splitMode = 0;
 
  LastGatherUbTiling(inputUbSize, indiceUbSize, outUbSize, indiceCastUbSize, inputNum, indicesNum, ubCols, ubRows, blockFactor, gFactor);

  if (indiceUbSize >= gFactor * indicesDtypeSize_) {
      splitMode = 0;
  } else {
      splitMode = 1;
  }
  
  lastTilingdata_.set_needCoreNum(needCoreNum_);
  lastTilingdata_.set_indicesNum(indicesNum);
  lastTilingdata_.set_splitIndices(splitIndices);
  lastTilingdata_.set_inputNum(inputNum);
  lastTilingdata_.set_splitMode(splitMode);
  lastTilingdata_.set_coreInCols(coreInCols);
  lastTilingdata_.set_inputUbSize(inputUbSize);
  lastTilingdata_.set_outUbSize(outUbSize);
  lastTilingdata_.set_indiceCastUbSize(indiceCastUbSize);
  lastTilingdata_.set_indiceUbSize(indiceUbSize);
  lastTilingdata_.set_ubCols(ubCols);
  lastTilingdata_.set_ubRows(ubRows);
  lastTilingdata_.set_gatherDimSize(gatherDimSize_);
  lastTilingdata_.set_gatherSize(gatherSize_);
  lastTilingdata_.set_blockFactor(blockFactor);
  lastTilingdata_.set_tailBlockFactor(tailBlockFactor);
  lastTilingdata_.set_gFactor(gFactor);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Gatherv2TilingBase::GaAllLoadTiling() {
  int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
  int64_t indicesDtypeSize = ge::GetSizeByDataType(indicesDtype_);
  int64_t aSize = innerSize_ * improveDtypeSize_;
  int64_t gSize = gatherDimSize_;
  int64_t pSize = outerSize_;
  int64_t totalCoreNum = aivNum_;
  int64_t aSizeAligned = (aSize + ubBlockSize - 1) / ubBlockSize * ubBlockSize;
  if (aSizeAligned == SIMD_VECTOR_REG || aSizeAligned == SIMD_VECTOR_REG / NUM_TWO) {  // 规避gather bank 冲突
    aSizeAligned += ubBlockSize;
  }
  
  int64_t gaSizeInUb = aSizeAligned * (gSize + 1);  // 需要多一块做清零
  
  gaAllLoadTilingdata_.set_gSize(gSize);
  gaAllLoadTilingdata_.set_aSize(aSize);
  gaAllLoadTilingdata_.set_aSizeAligned(aSizeAligned);
  gaAllLoadTilingdata_.set_indicesSize(gatherSize_);

  ubSize_ -= HELP_BUFFER_SIZE;
  int64_t minIndicesSize = std::min(gatherSize_, static_cast<int64_t>(MIN_INDICES_UB_SIZE / indicesDtypeSize));
  int64_t maxGaNumLoad = (ubSize_ - NUM_FOUR * MIN_OUT_UB_SIZE - minIndicesSize * indicesDtypeSize) / gaSizeInUb;

  // 先分核再分Ub
  int64_t pOuter = std::min(CeilDiv(pSize, maxGaNumLoad), totalCoreNum);
  int64_t pInner = CeilDiv(pSize, pOuter);
  pOuter = CeilDiv(pSize, pInner);
  int64_t pTail = pSize -  pInner * (pOuter - 1);

  int64_t indicesOuter = std::max(int64_t(1), totalCoreNum / pOuter);
  int64_t indicesInner = CeilDiv(gatherSize_, indicesOuter);
  indicesOuter = CeilDiv(gatherSize_, indicesInner);
  int64_t indicesTail = gatherSize_ -  indicesInner * (indicesOuter - 1);

  needCoreNum_ = pOuter * indicesOuter;
  gaAllLoadTilingdata_.set_usedCoreNum(needCoreNum_);
  gaAllLoadTilingdata_.set_indicesOuter(indicesOuter);
  gaAllLoadTilingdata_.set_normalCoreIndicesNum(indicesInner);
  gaAllLoadTilingdata_.set_tailCoreIndicesNum(indicesTail);
  gaAllLoadTilingdata_.set_pOuter(pOuter);
  gaAllLoadTilingdata_.set_normalCoreGaNum(pInner);
  gaAllLoadTilingdata_.set_tailCoreGaNum(pTail);

  // loop x -- 尝试是否能全载
  int64_t xBufferSizeLoopGa = 0;
  int64_t indicesBufferSizeLoopGa = 0;
  int64_t yBufferSizeLoopGa = 0;

  int64_t allLoadXBytes = CeilAlign(pInner * gaSizeInUb, ubBlockSize);
  bool isAllLoadX = (ubSize_ - NUM_FOUR * MIN_OUT_UB_SIZE - allLoadXBytes - MIN_INDICES_UB_SIZE) >= 0;
  if (isAllLoadX) {
    xBufferSizeLoopGa = allLoadXBytes;
    indicesBufferSizeLoopGa = std::min(indicesInner * indicesDtypeSize, (ubSize_ - NUM_FOUR * MIN_OUT_UB_SIZE - allLoadXBytes));
    indicesBufferSizeLoopGa = CeilAlign(indicesBufferSizeLoopGa, ubBlockSize);
    yBufferSizeLoopGa = (ubSize_ - indicesBufferSizeLoopGa - xBufferSizeLoopGa) / NUM_TWO;
  } else {
    xBufferSizeLoopGa =  CeilAlign(maxGaNumLoad * gaSizeInUb, ubBlockSize);
    indicesBufferSizeLoopGa = std::min(indicesInner * indicesDtypeSize, (ubSize_ - NUM_FOUR * MIN_OUT_UB_SIZE - xBufferSizeLoopGa));
    indicesBufferSizeLoopGa = CeilAlign(indicesBufferSizeLoopGa, ubBlockSize);
    yBufferSizeLoopGa = (ubSize_ - indicesBufferSizeLoopGa - xBufferSizeLoopGa) / NUM_TWO;
  }

  gaAllLoadTilingdata_.set_xBufferSize(xBufferSizeLoopGa);
  gaAllLoadTilingdata_.set_indicesBufferSize(indicesBufferSizeLoopGa);
  gaAllLoadTilingdata_.set_yBufferSize(yBufferSizeLoopGa);

  return ge::GRAPH_SUCCESS;
}

void Gatherv2TilingBase::DumpTilingInfo()
{
  if (tilingMode_ == TILING_LAST_GATHER) {
      ShowLastGtaherSimdTilingData();
  } else {
      ShowBaseTilingData();
  }
}
ge::graphStatus Gatherv2TilingBase::GetWorkspaceSize() {
  // 计算workspace大小
  workspaceSize_ = 16 * 1024 * 1024;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Gatherv2TilingBase::PostTiling() {
  context_->SetBlockDim(needCoreNum_);
  size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = workspaceSize_;
  if (tilingMode_ == TILING_SIMD_TWO_DIM) {
    simdTwoDimTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
    context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(simdTwoDimTilingData_.GetDataSize());
  } else if (tilingMode_ == TILING_LAST_GATHER) {
      lastTilingdata_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
      context_->GetRawTilingData()->GetCapacity());
      context_->GetRawTilingData()->SetDataSize(lastTilingdata_.GetDataSize());
  } else if (tilingMode_ == TILING_GA_ALL_LOAD) {
      gaAllLoadTilingdata_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
      context_->GetRawTilingData()->GetCapacity());
      context_->GetRawTilingData()->SetDataSize(gaAllLoadTilingdata_.GetDataSize());
  } else if (tilingMode_ == TILING_SIMT_TWO_DIM) {
    simtTwoDimTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
    context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(simtTwoDimTilingData_.GetDataSize());
  } else if (tilingMode_ == TILING_EMPTY) {
    emptyTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
    context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(emptyTilingData_.GetDataSize());
  } else {
    gatherV2TilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
    context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(gatherV2TilingData_.GetDataSize());
  }

  if (tilingMode_ == TILING_SIMT || tilingMode_ == TILING_SIMT_TWO_DIM) {
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
  }
  if (tilingMode_ == TILING_AFTER_GDIM) {
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE));
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
