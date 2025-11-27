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
 * \file index_fill_tiling.cpp
 * \brief
 */

#include "index_fill_tiling.h"

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint64_t SIZE_OF_HALF = 2;
constexpr uint64_t ALIGNED_NUM = 32;
constexpr size_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t HALF_ALIGNED = 32;
constexpr uint64_t P_LIMIT = 1024;
constexpr uint64_t N_LIMIT = 64;
constexpr uint64_t P_GM_NUMS = 8192;
constexpr uint64_t FLOAT_ALIGNED = 8;
constexpr uint32_t FRONT_P_KEY = 2;
constexpr uint32_t TAIL_N_KEY = 3;
constexpr uint32_t TAIL_P_KEY = 4;
constexpr uint32_t TAIL_N_LIMIT = 256;
constexpr uint32_t COMPARE_ALIGNED = 128;

struct TilingDataStructIndexFill {
  uint32_t coreNum = 0;
  uint64_t N = 0; // x在axis上的维度值
  uint64_t indicesNum = 0; // 索引tensor长度
  uint64_t indicesProcessMode = 0; // 索引处理模式
  uint64_t frontCoreNumTaskIndices = 0;
  uint64_t tailCoreNumTaskIndices = 0;
  uint64_t frontCoreDataTaskIndices = 0; 
  uint64_t tailCoreDataTaskIndices = 0;
  uint64_t ubSize = 0;
  uint64_t P = 0;
  uint64_t Q = 0;

  uint32_t tilingKey = 0;
};

class IndexFillTilingOp {
public:
  explicit IndexFillTilingOp(gert::TilingContext* context_) : context(context_){};
  void Init();
private:
  void ComputeTilingKey();
  void IndicesTaskTiling();
  void SetKernelTiling();
  void TilingDataPrint() const;
  gert::TilingContext* context = nullptr;
  TilingDataStructIndexFill tilingData;
};

void IndexFillTilingOp::TilingDataPrint() const {
  OP_LOGD(context->GetNodeName(), "coreNum: %d.", tilingData.coreNum);
  OP_LOGD(context->GetNodeName(), "N: %lu.", tilingData.N);
  OP_LOGD(context->GetNodeName(), "indicesNum: %lu.", tilingData.indicesNum);
  OP_LOGD(context->GetNodeName(), "indicesProcessMode: %lu.", tilingData.indicesProcessMode);
  OP_LOGD(context->GetNodeName(), "frontCoreNumTaskIndices: %lu.", tilingData.frontCoreNumTaskIndices);
  OP_LOGD(context->GetNodeName(), "tailCoreNumTaskIndices: %lu.", tilingData.tailCoreNumTaskIndices);
  OP_LOGD(context->GetNodeName(), "frontCoreDataTaskIndices: %lu.", tilingData.frontCoreDataTaskIndices);
  OP_LOGD(context->GetNodeName(), "tailCoreDataTaskIndices: %lu.", tilingData.tailCoreDataTaskIndices);
  OP_LOGD(context->GetNodeName(), "ubSize: %lu.", tilingData.ubSize);
  OP_LOGD(context->GetNodeName(), "P: %lu.", tilingData.P);
  OP_LOGD(context->GetNodeName(), "Q: %lu.", tilingData.Q);

  OP_LOGD(context->GetNodeName(), "tilingKey: %d.", tilingData.tilingKey);
}

void IndexFillTilingOp::SetKernelTiling() {
  IndexFillTilingData kernelTiling;
  kernelTiling.set_coreNum(tilingData.coreNum);
  kernelTiling.set_N(tilingData.N);
  kernelTiling.set_indicesNum(tilingData.indicesNum);
  kernelTiling.set_indicesProcessMode(tilingData.indicesProcessMode);
  kernelTiling.set_frontCoreNumTaskIndices(tilingData.frontCoreNumTaskIndices);
  kernelTiling.set_tailCoreNumTaskIndices(tilingData.tailCoreNumTaskIndices);
  kernelTiling.set_frontCoreDataTaskIndices(tilingData.frontCoreDataTaskIndices); 
  kernelTiling.set_tailCoreDataTaskIndices(tilingData.tailCoreDataTaskIndices);
  kernelTiling.set_ubSize(tilingData.ubSize);
  kernelTiling.set_P(tilingData.P);
  kernelTiling.set_Q(tilingData.Q);

  context->SetBlockDim(tilingData.coreNum);
  context->SetTilingKey(tilingData.tilingKey);

  kernelTiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(kernelTiling.GetDataSize());
}

void IndexFillTilingOp::ComputeTilingKey() {
  const gert::StorageShape* xShape = context->GetInputShape(0);
  int64_t xDims = xShape->GetStorageShape().GetDimNum();
  auto attrs = context->GetAttrs();
  int64_t axis = *(attrs->GetAttrPointer<int64_t>(0));
  OP_LOGD(context->GetNodeName(), "axis: %d", static_cast<int32_t>(axis));
  axis = axis >= 0 ? axis : axis + xDims;
  tilingData.N = xShape->GetStorageShape().GetDim(axis);
  if ((axis != xDims - 1) || (xDims == 1)) {
    // 非尾轴场景，x reshape为[P, N, Q]
    tilingData.P = 1;
    tilingData.Q = 1;
    for (int64_t i = 0; i < axis; i++) {
      tilingData.P *= xShape->GetStorageShape().GetDim(i);
    }
    for (int64_t i = axis + 1; i < xDims; i++) {
      tilingData.Q *= xShape->GetStorageShape().GetDim(i);
    }

    if (tilingData.Q >= P_LIMIT) {
      tilingData.tilingKey = 0;
    } else if (tilingData.N >= N_LIMIT || tilingData.Q >= N_LIMIT) {
      tilingData.tilingKey = 1;
    } else {
      tilingData.tilingKey = FRONT_P_KEY;
    }
    auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_INT64 || dtype == ge::DT_BOOL || dtype == ge::DT_UINT8) {
      tilingData.tilingKey = 0;
    }
  } else {
    // 尾轴场景，x reshape为[P, N]
    tilingData.P = 1;
    for (int64_t i = 0; i < axis; i++) {
      tilingData.P *= xShape->GetStorageShape().GetDim(i);
    }

    if (tilingData.N >= tilingData.coreNum * TAIL_N_LIMIT) {
      tilingData.tilingKey = TAIL_N_KEY;
    } else {
      tilingData.tilingKey = TAIL_P_KEY;
    }
  }
}

void IndexFillTilingOp::IndicesTaskTiling() {
  const gert::StorageShape* indicesShape = context->GetInputShape(1);
  tilingData.indicesNum = indicesShape->GetStorageShape().GetDim(0);

  if (tilingData.N < tilingData.coreNum * ALIGNED_NUM) {
    // 按indicesNum分核
    tilingData.indicesProcessMode = 0;
  } else {
    // 按N分核
    tilingData.indicesProcessMode = 1;
  }

  uint64_t dataNums = 0;
  if (tilingData.indicesProcessMode == 0) {
    dataNums = tilingData.indicesNum;
  } else {
    dataNums = tilingData.N;
  }

  tilingData.tailCoreDataTaskIndices = dataNums / tilingData.coreNum;
  if (tilingData.tailCoreDataTaskIndices == 0) {
    tilingData.frontCoreNumTaskIndices = dataNums;
    tilingData.frontCoreDataTaskIndices = 1;
    tilingData.tailCoreNumTaskIndices = 0;
    tilingData.tailCoreDataTaskIndices = 0;
  } else {
    if (dataNums % tilingData.coreNum == 0) {
      tilingData.frontCoreNumTaskIndices = tilingData.coreNum;
      tilingData.frontCoreDataTaskIndices = tilingData.tailCoreDataTaskIndices;
      tilingData.tailCoreNumTaskIndices = 0;
      tilingData.tailCoreDataTaskIndices = 0;
    } else {
      tilingData.frontCoreNumTaskIndices = dataNums - tilingData.tailCoreDataTaskIndices * tilingData.coreNum;
      tilingData.frontCoreDataTaskIndices = tilingData.tailCoreDataTaskIndices + 1;
      tilingData.tailCoreNumTaskIndices = tilingData.coreNum - tilingData.frontCoreNumTaskIndices;
    }
  }
}

void IndexFillTilingOp::Init() {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  tilingData.coreNum = coreNum;
  ComputeTilingKey();
  IndicesTaskTiling();
  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  tilingData.ubSize = ubSizePlatForm;
  const gert::StorageShape* xShape = context->GetInputShape(0);
  int64_t xDims = xShape->GetStorageShape().GetDimNum();
  auto attrs = context->GetAttrs();
  int64_t axis = *(attrs->GetAttrPointer<int64_t>(0));
  axis = axis >= 0 ? axis : axis + xDims;
  // usrworkspacesize：标记Tensor+同步Tensor+复制Tensor
  size_t userWorkspaceSize = ((tilingData.N * SIZE_OF_HALF + tilingData.coreNum * sizeof(uint8_t) +
                               ALIGNED_NUM - 1) / ALIGNED_NUM) * ALIGNED_NUM;
  if (tilingData.tilingKey == 1) {
    userWorkspaceSize += tilingData.Q * FLOAT_ALIGNED * sizeof(uint32_t); // front n分支
  } else if (tilingData.tilingKey == FRONT_P_KEY) {
    if (tilingData.N * tilingData.Q < COMPARE_ALIGNED) {
      userWorkspaceSize += COMPARE_ALIGNED * tilingData.N * tilingData.Q * SIZE_OF_HALF;
    } else {
      userWorkspaceSize += P_GM_NUMS * SIZE_OF_HALF; // front p分支
    }
  } else if (tilingData.tilingKey == TAIL_P_KEY) {
    userWorkspaceSize += tilingData.N * HALF_ALIGNED * SIZE_OF_HALF; // tail p分支
  }

  size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
  currentWorkSpace[0] = SYS_WORK_SPACE_SIZE + userWorkspaceSize;

  SetKernelTiling();
  TilingDataPrint();
}

static ge::graphStatus IndexFillTilingFunc(gert::TilingContext* context)
{
  OP_LOGD(context->GetNodeName(), "Tiling for IndexFill start.");
  IndexFillTilingOp tilingOp(context);
  tilingOp.Init();
  OP_LOGD(context->GetNodeName(), "Tiling for IndexFill end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForIndexFill(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For IndexFill start.");
    auto compileInfo = context->GetCompiledInfo<IndexFillCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<uint64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
                    OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For IndexFill end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexFill)
    .Tiling(IndexFillTilingFunc)
    .TilingParse<IndexFillCompileInfo>(TilingPrepareForIndexFill);
}
