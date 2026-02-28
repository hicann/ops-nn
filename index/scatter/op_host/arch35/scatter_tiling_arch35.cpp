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
 * \file scatter_tiling_arch35.cpp
 * \brief
 */
#include "scatter_tiling_arch35.h"
#include <map>
#include <string>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/platform_util.h"
#include "op_host/tiling_templates_registry.h"
#include "scatter_tiling.h"
#include "util/math_util.h"
#include "error_util.h"

using namespace AscendC;
using namespace ge;
using namespace std;

namespace optiling {
constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t MAX_INDICES_DIM = 2;
constexpr int32_t INPUT_INDEX = 0;
constexpr int32_t INDICES_INDEX = 1;
constexpr int32_t UPDATES_INDEX = 2;
constexpr int32_t REDUCE_INDEX = 0;
constexpr int32_t AXIS_INDEX = 1;
constexpr int32_t TILING_KEY_INT32_INT8 = 0;
constexpr int32_t TILING_KEY_INT32_UINT8 = 1;
constexpr int32_t TILING_KEY_INT32_FLOAT16 = 2;
constexpr int32_t TILING_KEY_INT32_BF16 = 3;
constexpr int32_t TILING_KEY_INT32_FLOAT = 4;
constexpr int32_t TILING_KEY_INT32_INT32 = 5;
constexpr int32_t TILING_KEY_INT64_INT8 = 6;
constexpr int32_t TILING_KEY_INT64_UINT8 = 7;
constexpr int32_t TILING_KEY_INT64_FLOAT16 = 8;
constexpr int32_t TILING_KEY_INT64_BF16 = 9;
constexpr int32_t TILING_KEY_INT64_FLOAT = 10;
constexpr int32_t TILING_KEY_INT64_INT32 = 11;
constexpr int32_t TILING_KEY_INT32_INT2 = 12;
constexpr int32_t TILING_KEY_INT64_INT2 = 13;
constexpr int32_t TILING_KEY_UINT64_INT32_INT8 = 14;
constexpr int32_t TILING_KEY_UINT64_INT32_UINT8 = 15;
constexpr int32_t TILING_KEY_UINT64_INT32_FLOAT16 = 16;
constexpr int32_t TILING_KEY_UINT64_INT32_BF16 = 17;
constexpr int32_t TILING_KEY_UINT64_INT32_FLOAT = 18;
constexpr int32_t TILING_KEY_UINT64_INT32_INT32 = 19;
constexpr int32_t TILING_KEY_UINT64_INT64_INT8 = 20;
constexpr int32_t TILING_KEY_UINT64_INT64_UINT8 = 21;
constexpr int32_t TILING_KEY_UINT64_INT64_FLOAT16 = 22;
constexpr int32_t TILING_KEY_UINT64_INT64_BF16 = 23;
constexpr int32_t TILING_KEY_UINT64_INT64_FLOAT = 24;
constexpr int32_t TILING_KEY_UINT64_INT64_INT32 = 25;
constexpr int32_t TILING_KEY_UINT64_INT32_INT2 = 26;
constexpr int32_t TILING_KEY_UINT64_INT64_INT2 = 27;
constexpr int32_t DIM3 = 3;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM0 = 0;
constexpr int32_t SECOND_LAST_DIM = -2;
constexpr int32_t SIMT_RESERVED_SIZE = 32 * 1024;
constexpr int32_t SIMT_PARAM_SIZE = 1024;
constexpr int32_t SIMD_RESERVED_SIZE = 8 * 1024;
constexpr int32_t PROMOTE_DTYPE_SIZE = 8;
constexpr int32_t MIN_FACTOR = 1024; // if < MIN_FACTOR, use one core only
constexpr int64_t MAX_INT32_NUM = 2147483647;
constexpr int64_t DB_BUFFER = 2;
constexpr int32_t SIMD_PERF_SIZE = 8 * 1024;
constexpr int64_t INDICES_SIZE = 4096;
constexpr int64_t SIMD_TEMP = 1;
constexpr int64_t SIMD_PERF_TEMP = 2;
constexpr int32_t TWO_INDICES = 2;
constexpr int32_t TWO_DIM_SIZE = 2;
constexpr int32_t ONE_DIM_SIZE = 1;
constexpr int32_t ZERO_DIM_SIZE = 0;
constexpr int32_t BATCH_DIM = 0;
constexpr int64_t IN_DTYPE_B64 = 8;

static map<const ge::DataType, const int32_t> g_dtypeLen = {{ge::DT_INT8, 1}, {ge::DT_UINT8, 1}, {ge::DT_FLOAT16, 2},
                                                            {ge::DT_FLOAT, 4}, {ge::DT_INT32, 4}, {ge::DT_BF16, 2},
                                                            {ge::DT_FLOAT8_E4M3FN, 1}, {ge::DT_FLOAT8_E5M2, 1}, {ge::DT_FLOAT8_E8M0, 1}};

std::map<std::tuple<bool, ge::DataType, ge::DataType>, int32_t> tilingKeyMap;

ge::graphStatus ScatterTiling::GetPlatformInfo() {
  auto platformPtr = context_->GetPlatformInfo();
  if (platformPtr == nullptr) {
    auto compileInfoPtr = static_cast<const ScatterKvCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
                    return ge::GRAPH_FAILED);
    aivCoreNum = static_cast<int32_t>(compileInfoPtr->core_num);
    ubSize = static_cast<int32_t>(compileInfoPtr->ub_size);
  } else {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
    aivCoreNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());

    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize = static_cast<int32_t>(ubSizePlatform);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::GetShapeAttrsInfo() {
  auto inputDesc = context_->GetInputDesc(INPUT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
  inputDtype = inputDesc->GetDataType();
  if (inputDtype != ge::DataType::DT_FLOAT16 && inputDtype != ge::DataType::DT_FLOAT &&
      inputDtype != ge::DataType::DT_BF16 && inputDtype != ge::DataType::DT_INT8 &&
      inputDtype != ge::DataType::DT_UINT8 && inputDtype != ge::DataType::DT_INT32 &&
      inputDtype != ge::DataType::DT_FLOAT8_E4M3FN && inputDtype != ge::DataType::DT_FLOAT8_E5M2 &&
      inputDtype != ge::DataType::DT_FLOAT8_E8M0) {
    OP_LOGE("Scatter", "invalid input dtype.");
    return ge::GRAPH_FAILED;
  }

  auto indicesDesc = context_->GetInputDesc(INDICES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
  indicesDtype = indicesDesc->GetDataType();
  if (indicesDtype != ge::DataType::DT_INT32 && indicesDtype != ge::DataType::DT_INT64) {
    OP_LOGE("Scatter", "invalid indices dtype.");
    return ge::GRAPH_FAILED;
  }

  auto updatesDesc = context_->GetInputDesc(UPDATES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, updatesDesc);
  updatesDtype = updatesDesc->GetDataType();
  if (updatesDtype != ge::DataType::DT_FLOAT16 && updatesDtype != ge::DataType::DT_FLOAT &&
      updatesDtype != ge::DataType::DT_BF16 && updatesDtype != ge::DataType::DT_INT8 &&
      updatesDtype != ge::DataType::DT_UINT8 && updatesDtype != ge::DataType::DT_INT32 &&
      updatesDtype != ge::DataType::DT_FLOAT8_E4M3FN && updatesDtype != ge::DataType::DT_FLOAT8_E5M2 &&
      updatesDtype != ge::DataType::DT_FLOAT8_E8M0) {
    OP_LOGE("Scatter", "invalid updates dtype.");
    return ge::GRAPH_FAILED;
  }

  if (updatesDtype != inputDtype) {
    OP_LOGE("Scatter", "dtype of input and updates should be same.");
    return ge::GRAPH_FAILED;
  }

  dtypeSize = g_dtypeLen[inputDtype];

  auto indices = context_->GetInputShape(INDICES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
  auto indicesShape = Ops::Base::EnsureNotScalar(indices->GetOriginShape());
  if (indicesShape.GetDimNum() > MAX_INDICES_DIM) {
    OP_LOGE("Scatter", "indices shape dim(=%zu) > 2, please check.",
                                    indicesShape.GetDimNum());
    return ge::GRAPH_FAILED;
  }
  indicesDim = indicesShape.GetDimNum();

  auto attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

  const char* reducePtr = attrs->GetAttrPointer<char>(REDUCE_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, reducePtr);
  std::string reduce(reducePtr);
  if (reduce != "update" && reduce != "none" && reduce != "") {
    OP_LOGE("Scatter", "reduce(=%s) only supports 'update'", reduce.c_str());
    return ge::GRAPH_FAILED;
  }

  const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(AXIS_INDEX);
  axis = (axisPtr == nullptr) ? 0 : *axisPtr;
  if (axis == 0) {
    OP_LOGE("Scatter", "axis does not support 0.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool ScatterTiling::IsCapable() {
  return true;
}

ge::graphStatus ScatterTiling::GetShapes() {
  // get input_shape
  auto inputShape = context_->GetInputShape(INPUT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
  auto indicesShape = context_->GetInputShape(INDICES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape);
  auto updatesShape = context_->GetInputShape(UPDATES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, updatesShape);

  auto indicesDimSize = indicesShape->GetOriginShape().GetDimNum();
  if (indicesDimSize == ZERO_DIM_SIZE && updatesShape->GetOriginShape().GetDim(BATCH_DIM) != 1) {
    OP_LOGE("Scatter", "when the dimension of indices are 0, batch axis of updates should be 1.");
    return ge::GRAPH_FAILED;
  }
  if (indicesDimSize != ZERO_DIM_SIZE && indicesDimSize != ONE_DIM_SIZE && indicesDimSize != TWO_DIM_SIZE) {
    OP_LOGE("Scatter", "dimension of indices should be 0, 1 or 2.");
    return ge::GRAPH_FAILED;
  }

  inputOriginShape = Ops::Base::EnsureNotScalar(inputShape->GetOriginShape());
  indicesOriginShape = Ops::Base::EnsureNotScalar(indicesShape->GetOriginShape());
  updatesOriginShape = Ops::Base::EnsureNotScalar(updatesShape->GetOriginShape());

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::CheckNullTensor() {
  if (inputOriginShape.GetDimNum() != updatesOriginShape.GetDimNum()) {
    OP_LOGE("Scatter", "input should be same dims with updates.");
    return ge::GRAPH_FAILED;
  }

  if (inputOriginShape.GetDimNum() * indicesOriginShape.GetDimNum() == 0) {
    OP_LOGE("Scatter", "input or indices shouldn't be null.");
    return ge::GRAPH_FAILED;
  }

  int64_t inputSize = inputOriginShape.GetShapeSize();
  int64_t indicesSize = indicesOriginShape.GetShapeSize();
  int64_t updatesSize = updatesOriginShape.GetShapeSize();
  if (inputSize == 0 || indicesSize == 0 || updatesSize == 0) {
    OP_LOGE("Scatter", "input %ld or indices %ld or updates %ld shape shouldn't be zero.",
            inputSize, indicesSize, updatesSize);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::MergeDims() {
  int32_t oldDims = inputOriginShape.GetDimNum();
  int32_t tmpAbsAxis = axis < 0 ? oldDims + axis : axis;

  if (tmpAbsAxis < 0 || tmpAbsAxis >= oldDims) {
    OP_LOGE("Scatter", "axis should be less than data dims.");
    return ge::GRAPH_FAILED;
  }

  size_t absAxis = size_t(tmpAbsAxis);
  inputNewShape.SetDimNum(0);
  updatesNewShape.SetDimNum(0);

  inputNewShape.AppendDim(inputOriginShape[0]);
  updatesNewShape.AppendDim(updatesOriginShape[0]);

  size_t inputSecondDims = 1;
  size_t updatesSecondDims = 1;
  for (size_t i = 1; i < absAxis; i++) {
    inputSecondDims *= inputOriginShape[i];
    updatesSecondDims *= updatesOriginShape[i];
  }
  inputNewShape.AppendDim(inputSecondDims);
  updatesNewShape.AppendDim(updatesSecondDims);

  inputNewShape.AppendDim(inputOriginShape[absAxis]);
  updatesNewShape.AppendDim(updatesOriginShape[absAxis]);

  size_t inputFourthDims = 1;
  size_t updatesFourthDims = 1;
  for (size_t i = absAxis + 1; i < inputOriginShape.GetDimNum(); i++) {
    inputFourthDims *= inputOriginShape[i];
    updatesFourthDims *= updatesOriginShape[i];
  }

  inputNewShape.AppendDim(inputFourthDims);
  updatesNewShape.AppendDim(updatesFourthDims);
  axis = SECOND_LAST_DIM;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::CheckShapes() {
  int64_t indicesOriginDim = indicesOriginShape.GetDimNum();
  if (indicesOriginDim == TWO_INDICES) {
    OP_CHECK_IF(indicesOriginShape[1] != 2,
                    OP_LOGE("Scatter", "when discrete, indicesOriginShape[1] should be 2."),
                    return ge::GRAPH_FAILED);
  }

  if (updatesNewShape[DIM0] != indicesOriginShape[DIM0]) {
    OP_LOGE("Scatter", "updatesShape[0] should be same with indicesShape[0].");
    return ge::GRAPH_FAILED;
  }

  if (updatesNewShape[DIM0] > inputNewShape[DIM0]) {
    OP_LOGE("Scatter", "updatesShape[0] should be less than inputShape[0].");
    return ge::GRAPH_FAILED;
  }

  if (updatesNewShape[DIM1] != inputNewShape[DIM1]) {
    OP_LOGE("Scatter", "updatesShape[1] should be same with inputShape[1].");
    return ge::GRAPH_FAILED;
  }

  if (axis == SECOND_LAST_DIM) {
    if (updatesNewShape[DIM3] != inputNewShape[DIM3]) {
      OP_LOGE("Scatter", "updateOriginShape[3] should be same with dataOriginShape[3].");
      return ge::GRAPH_FAILED;
    }
  } else if (axis == -1) {
    if (updatesNewShape[DIM2] != inputNewShape[DIM2]) {
      OP_LOGE("Scatter", "updatesShape[2] should be same with inputShape[2].");
      return ge::GRAPH_FAILED;
    }
  } else {
    OP_LOGE("Scatter", "axis only support -1 or -2!");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static map<const int32_t, const std::vector<int32_t>> g_factors = {{8, {8, 4, 2, 1}},  {4, {4, 2, 1}}, {2, {2, 1}}};

ge::graphStatus ScatterTiling::PromoteDtype() {
  if (simdTemp > 0) {
    return ge::GRAPH_SUCCESS;
  }
  int32_t factor = PROMOTE_DTYPE_SIZE / dtypeSize;
  if (axis == SECOND_LAST_DIM && updatesNewShape[DIM3] % factor == 0) {
    updatesNewShape[DIM3] /= factor;
    inputNewShape[DIM3] /= factor;
    dtypeSize = PROMOTE_DTYPE_SIZE;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::DoSimdTiling() {
  // [batch_size, outer_size, scatter_dim_size, inner_size]
  // split batch(b*o)
  int64_t spiltNum = updatesNewShape[DIM0] * updatesNewShape[DIM1];
  if (simdTemp == SIMD_PERF_TEMP) {
    // split batch(b*o*s)
    spiltNum = updatesNewShape[DIM0] * updatesNewShape[DIM1] * updatesNewShape[DIM2];
  }
  blockFactor = Ops::Base::CeilDiv(spiltNum, static_cast<int64_t>(aivCoreNum));
  aivCoreNum = Ops::Base::CeilDiv(spiltNum, blockFactor);
  tailBlockData = spiltNum - (aivCoreNum - 1) * blockFactor;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::GetTilingParam() {
  if (simdTemp > 0) {
    return DoSimdTiling();
  }
  int64_t totalDims = updatesNewShape[DIM0] * updatesNewShape[DIM1] * updatesNewShape[DIM2] * updatesNewShape[DIM3];
  if (totalDims > MAX_INT32_NUM) {
    isUint64 = true;
  }
  if (totalDims <= MIN_FACTOR) {
    // use one core
    blockFactor = totalDims;
    aivCoreNum = 1;
    tailCoreNum = 0;
    oneCoreTemp = true;
  } else if (totalDims <= aivCoreNum * MIN_FACTOR) {
    // use part of cores
    oneCoreTemp = false;
    blockFactor = MIN_FACTOR;
    aivCoreNum = (totalDims + MIN_FACTOR - 1) / MIN_FACTOR;
    if (totalDims % MIN_FACTOR == 0) {
      tailCoreNum = 0;
    } else {
      tailCoreNum = 1;
    }
  } else {
    // use all cores
    oneCoreTemp = false;
    blockFactor = (totalDims + aivCoreNum - 1) / aivCoreNum;
    tailCoreNum = aivCoreNum * blockFactor - totalDims;
  }
  context_->SetLocalMemorySize(ubSize - SIMT_RESERVED_SIZE);

  if (oneCoreTemp) {
    tilingData.set_simtThreadNum(simtThreadNum);
    if (totalDims <= simtThreadNum) {
        tilingData.set_simtUsedCore(1);
        tilingData.set_simtPerCoreNum(totalDims);
        tilingData.set_simtTailCoreNum(totalDims);
        return ge::GRAPH_SUCCESS;
    }
  }
  return ge::GRAPH_SUCCESS;
}

void ScatterTiling::SetTilingData() {
  tilingData.set_axis(axis);
  tilingData.set_indicesDim(indicesDim);
  tilingData.set_updatesDim0(updatesNewShape[DIM0]);
  tilingData.set_updatesDim1(updatesNewShape[DIM1]);
  tilingData.set_updatesDim2(updatesNewShape[DIM2]);
  tilingData.set_updatesDim3(updatesNewShape[DIM3]);
  tilingData.set_inputDim0(inputNewShape[DIM0]);
  tilingData.set_inputDim1(inputNewShape[DIM1]);
  tilingData.set_inputDim2(inputNewShape[DIM2]);
  tilingData.set_inputDim3(inputNewShape[DIM3]);
  tilingData.set_aivCoreNum(aivCoreNum);
  tilingData.set_blockFactor(blockFactor);
  tilingData.set_tailCoreNum(tailCoreNum);
  tilingData.set_tailBlockData(tailBlockData);
  tilingData.set_loopLength(loopLength);
  tilingData.set_indicesUbSize(indicesUbSize);
  tilingData.set_dtypeSize(static_cast<int64_t>(dtypeSize));
  // 32K for simt and 8K for simd
  tilingData.set_ubSize(ubSize - SIMT_RESERVED_SIZE - SIMD_RESERVED_SIZE);
}

ge::graphStatus ScatterTiling::DoOpTiling() {
  if (GetShapes() != ge::GRAPH_SUCCESS) {
    OP_LOGE("Scatter", "GetShapes failed!");
    return ge::GRAPH_FAILED;
  }

  if (CheckNullTensor() != ge::GRAPH_SUCCESS) {
    OP_LOGE("Scatter", "CheckNullTenosr failed!");
    return ge::GRAPH_FAILED;
  }

  if (MergeDims() != ge::GRAPH_SUCCESS) {
    OP_LOGE("Scatter", "MergeDims failed!");
    return ge::GRAPH_FAILED;
  }

  if (CheckShapes() != ge::GRAPH_SUCCESS) {
    OP_LOGE("Scatter", "CheckShapes failed!");
    return ge::GRAPH_FAILED;
  }

  int64_t srcStride = updatesNewShape[DIM2] * updatesNewShape[DIM3];
  if (srcStride * dtypeSize > MIN_FACTOR) {
    indicesUbSize = INDICES_SIZE;
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t maxUpdatesUbSize = Ops::Base::FloorAlign((ubSize - SIMD_RESERVED_SIZE - INDICES_SIZE) / DB_BUFFER, ubBlockSize);
    loopLength = maxUpdatesUbSize / dtypeSize;
    simdTemp = SIMD_TEMP;
    int64_t boNum = updatesNewShape[DIM0] * updatesNewShape[DIM1];
    if (boNum < aivCoreNum && srcStride > loopLength && updatesNewShape[DIM3] * dtypeSize > SIMD_PERF_SIZE) {
      simdTemp = SIMD_PERF_TEMP;
    }
  }

  if (PromoteDtype() != ge::GRAPH_SUCCESS) {
    OP_LOGE("Scatter", "PromoteDtype failed!");
    return ge::GRAPH_FAILED;
  }

  if (GetTilingParam() != ge::GRAPH_SUCCESS) {
    OP_LOGE("Scatter", "GetTilingParam failed!");
    return ge::GRAPH_FAILED;
  }
  SetTilingData();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

void ScatterTiling::InitTilingKeyMap() {
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_INT8)] = TILING_KEY_INT32_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT8_E5M2)] = TILING_KEY_INT32_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT8_E4M3FN)] = TILING_KEY_INT32_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT8_E8M0)] = TILING_KEY_INT32_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_UINT8)] = TILING_KEY_INT32_UINT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT16)] = TILING_KEY_INT32_FLOAT16;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_BF16)] = TILING_KEY_INT32_BF16;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT)] = TILING_KEY_INT32_FLOAT;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT32, ge::DataType::DT_INT32)] = TILING_KEY_INT32_INT32;

  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_INT8)] = TILING_KEY_INT64_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT8_E5M2)] = TILING_KEY_INT64_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT8_E4M3FN)] = TILING_KEY_INT64_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT8_E8M0)] = TILING_KEY_INT64_INT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_UINT8)] = TILING_KEY_INT64_UINT8;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT16)] = TILING_KEY_INT64_FLOAT16;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_BF16)] = TILING_KEY_INT64_BF16;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT)] = TILING_KEY_INT64_FLOAT;
  tilingKeyMap[std::make_tuple(false, ge::DataType::DT_INT64, ge::DataType::DT_INT32)] = TILING_KEY_INT64_INT32;

  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_INT8)] = TILING_KEY_UINT64_INT32_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT8_E5M2)] = TILING_KEY_UINT64_INT32_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT8_E4M3FN)] = TILING_KEY_UINT64_INT32_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT8_E8M0)] = TILING_KEY_UINT64_INT32_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_UINT8)] = TILING_KEY_UINT64_INT32_UINT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT16)] = TILING_KEY_UINT64_INT32_FLOAT16;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_BF16)] = TILING_KEY_UINT64_INT32_BF16;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_FLOAT)] = TILING_KEY_UINT64_INT32_FLOAT;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT32, ge::DataType::DT_INT32)] = TILING_KEY_UINT64_INT32_INT32;

  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_INT8)] = TILING_KEY_UINT64_INT64_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT8_E5M2)] = TILING_KEY_UINT64_INT64_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT8_E4M3FN)] = TILING_KEY_UINT64_INT64_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT8_E8M0)] = TILING_KEY_UINT64_INT64_INT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_UINT8)] = TILING_KEY_UINT64_INT64_UINT8;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT16)] = TILING_KEY_UINT64_INT64_FLOAT16;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_BF16)] = TILING_KEY_UINT64_INT64_BF16;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_FLOAT)] = TILING_KEY_UINT64_INT64_FLOAT;
  tilingKeyMap[std::make_tuple(true, ge::DataType::DT_INT64, ge::DataType::DT_INT32)] = TILING_KEY_UINT64_INT64_INT32;
}

uint64_t ScatterTiling::setSimtTilingKey(uint64_t& tilingKey) const {
  if (oneCoreTemp) {
    uint64_t factorStart = 300;
    tilingKey = factorStart + tilingKey;
  }
  return tilingKey;
}

uint64_t ScatterTiling::GetTilingKey() const {
  if (simdTemp > 0) {
    uint64_t factorStart = 100;
    uint64_t tilingKey = simdTemp * factorStart;
    return tilingKey;
  }
  uint64_t tilingKey = 0xFF;

  if (isUint64) {
    if (indicesDtype == ge::DataType::DT_INT32) {
      if (dtypeSize == PROMOTE_DTYPE_SIZE) {
        tilingKey = static_cast<uint64_t>(TILING_KEY_UINT64_INT32_INT2);
        return tilingKey;
      }
    } else {
      if (dtypeSize == PROMOTE_DTYPE_SIZE) {
        tilingKey = static_cast<uint64_t>(TILING_KEY_UINT64_INT64_INT2);
        return tilingKey;
      }
    }
  } else {
    if (indicesDtype == ge::DataType::DT_INT32) {
      if (dtypeSize == PROMOTE_DTYPE_SIZE) {
        tilingKey = static_cast<uint64_t>(TILING_KEY_INT32_INT2);
        tilingKey = setSimtTilingKey(tilingKey);
        return tilingKey;
      }
    } else {
      if (dtypeSize == PROMOTE_DTYPE_SIZE) {
        tilingKey = static_cast<uint64_t>(TILING_KEY_INT64_INT2);
        tilingKey = setSimtTilingKey(tilingKey);
        return tilingKey;
      }
    }
  }

  auto it = tilingKeyMap.find(std::make_tuple(isUint64, indicesDtype, inputDtype));
  if (it != tilingKeyMap.end()) {
    tilingKey = it->second;
  }

  tilingKey = setSimtTilingKey(tilingKey);
  return tilingKey;
}

ge::graphStatus ScatterTiling::GetWorkspaceSize() {
  workspaceSize_ = SYS_WORKSPACE_SIZE;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterTiling::PostTiling() {
  InitTilingKeyMap();
  tilingData.set_tilingKey(GetTilingKey());
  context_->SetTilingKey(GetTilingKey());
  context_->SetBlockDim(aivCoreNum);
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  workspaces[0] = workspaceSize_;
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

void ScatterTiling::DumpTilingInfo()
{
  std::ostringstream info;
  info << "tilingKey: " << tilingData.get_tilingKey();
  info << ", ubSize: " << tilingData.get_ubSize();
  info << ", aivCoreNum: " << tilingData.get_aivCoreNum();
  info << ", tailCoreNum: " << tilingData.get_tailCoreNum();
  info << ", blockFactor: " << tilingData.get_blockFactor();
  info << ", axis: " << tilingData.get_axis();
  info << ", indicesDim: " << tilingData.get_indicesDim();
  info << ", inputDim0: " << tilingData.get_inputDim0();
  info << ", inputDim1: " << tilingData.get_inputDim1();
  info << ", inputDim2: " << tilingData.get_inputDim2();
  info << ", inputDim3: " << tilingData.get_inputDim3();
  info << ", updatesDim0: " << tilingData.get_updatesDim0();
  info << ", updatesDim1: " << tilingData.get_updatesDim1();
  info << ", updatesDim2: " << tilingData.get_updatesDim2();
  info << ", updatesDim3: " << tilingData.get_updatesDim3();
  info << ", tailBlockData: " << tilingData.get_tailBlockData();
  info << ", loopLength: " << tilingData.get_loopLength();
  info << ", indicesUbSize: " << tilingData.get_indicesUbSize();
  info << ", dtypeSize: " << tilingData.get_dtypeSize();
  OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(Scatter, ScatterTiling, 10000);

} // namespace optiling