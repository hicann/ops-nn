/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_scatter_add.cc
 * \brief
 */

#include "inplace_scatter_add_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr int32_t BYTE_BLOCK = 32;
constexpr size_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t SMALL_TAIL = 128;
constexpr int32_t BIT_MOVE = 2;

struct TilingDataStructInplaceScatterAdd{
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;
  uint32_t NAligned = 0;
  uint32_t frontCoreNum = 0;
  uint32_t tailCoreNum = 0;
  uint32_t frontCoreIndicesNum = 0;
  uint32_t tailCoreIndicesNum = 0;
  uint32_t ubSize = 0;

  uint32_t tilingKey = 0b000; // 0b111 means varIsInt32, indexIsInt64, isBigTail
};

class InplaceScatterAddTilingOp {
public:
  explicit InplaceScatterAddTilingOp(gert::TilingContext* context_) : context(context_){};
  void Init();
private:
  void SetKernelTiling();
  void TilingDataPrint() const;
  uint64_t GetDTypeSize(ge::DataType dataType) const;
  gert::TilingContext* context = nullptr;
  TilingDataStructInplaceScatterAdd tilingData;
};

uint64_t InplaceScatterAddTilingOp::GetDTypeSize(ge::DataType dataType) const {
  switch (dataType) {
      case ge::DT_FLOAT:
          return sizeof(float);
      case ge::DT_INT32:
          return sizeof(int32_t);
      default:
          return 1;
  }
}

void InplaceScatterAddTilingOp::TilingDataPrint() const {
  OP_LOGD(context, "M: %d.", tilingData.M);
  OP_LOGD(context, "N: %d.", tilingData.N);
  OP_LOGD(context, "K: %d.", tilingData.K);
  OP_LOGD(context, "NAligned: %d.", tilingData.NAligned);
  OP_LOGD(context, "frontCoreNum: %d.", tilingData.frontCoreNum);
  OP_LOGD(context, "tailCoreNum: %d.", tilingData.tailCoreNum);
  OP_LOGD(context, "frontCoreIndicesNum: %d.", tilingData.frontCoreIndicesNum);
  OP_LOGD(context, "tailCoreIndicesNum: %d.", tilingData.tailCoreIndicesNum);
  OP_LOGD(context, "ubSize: %d * sizeof(float).", tilingData.ubSize);
  OP_LOGD(context, "BlockDim: %d.", tilingData.frontCoreNum + tilingData.tailCoreNum);
  OP_LOGD(context, "tilingKey: %d.", tilingData.tilingKey);
}

void InplaceScatterAddTilingOp::SetKernelTiling() {
  InplaceScatterAddTilingData kernelTiling;
  kernelTiling.set_M(tilingData.M);
  kernelTiling.set_N(tilingData.N);
  kernelTiling.set_K(tilingData.K);
  kernelTiling.set_NAligned(tilingData.NAligned);
  kernelTiling.set_frontCoreNum(tilingData.frontCoreNum);
  kernelTiling.set_tailCoreNum(tilingData.tailCoreNum);
  kernelTiling.set_frontCoreIndicesNum(tilingData.frontCoreIndicesNum);
  kernelTiling.set_tailCoreIndicesNum(tilingData.tailCoreIndicesNum);
  kernelTiling.set_ubSize(tilingData.ubSize);

  context->SetBlockDim(tilingData.frontCoreNum + tilingData.tailCoreNum);
  context->SetTilingKey(tilingData.tilingKey);

  kernelTiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(kernelTiling.GetDataSize());
}

void InplaceScatterAddTilingOp::Init() {
  const gert::StorageShape* var_shape = context->GetInputShape(0);
  tilingData.M = var_shape->GetStorageShape().GetDim(0);
  tilingData.N = 1;
  for (uint32_t i = 1; i < var_shape->GetStorageShape().GetDimNum(); i++)
    tilingData.N = tilingData.N * var_shape->GetStorageShape().GetDim(i);
  const gert::StorageShape* indices_shape = context->GetInputShape(1);
  tilingData.K = indices_shape->GetStorageShape().GetDim(0);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  OP_LOGD(context, "ascendcPlatform CoreNum: %d.", coreNum);

  if (coreNum == 0){
    OP_LOGE(context, "CoreNum got is 0!");
  } else{
    tilingData.tailCoreIndicesNum = tilingData.K / coreNum;
  }
  if (tilingData.tailCoreIndicesNum == 0) {
    tilingData.frontCoreIndicesNum = 1;
    tilingData.frontCoreNum = tilingData.K;
  } else {
    tilingData.frontCoreIndicesNum = tilingData.tailCoreIndicesNum + 1;
    tilingData.frontCoreNum = tilingData.K - tilingData.tailCoreIndicesNum * coreNum;
    tilingData.tailCoreNum = coreNum - tilingData.frontCoreNum;
  }

  uint32_t alignedNum = BYTE_BLOCK / GetDTypeSize(context->GetInputDesc(0)->GetDataType());
  tilingData.NAligned = ((tilingData.N + alignedNum - 1) / alignedNum) * alignedNum;

  if (tilingData.N > SMALL_TAIL) {
    tilingData.tilingKey = tilingData.tilingKey + 1;
  }

  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  tilingData.ubSize = static_cast<uint32_t>(ubSizePlatForm) / GetDTypeSize(context->GetInputDesc(0)->GetDataType());

  size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
  currentWorkSpace[0] = SYS_WORK_SPACE_SIZE;

  SetKernelTiling();
  TilingDataPrint();
}

static ge::graphStatus InplaceScatterAddTilingFunc(gert::TilingContext* context)
{
  OP_LOGD(context, "Tiling for InplaceScatterAdd start.");
  InplaceScatterAddTilingOp tilingOp(context);
  tilingOp.Init();
  OP_LOGD(context, "Tiling for InplaceScatterAdd end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForInplaceScatterAdd(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For InplaceScatterAdd start.");
    auto compileInfo = context -> GetCompiledInfo<InplaceScatterAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
                    OP_LOGE(context, "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For InplaceScatterAdd end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplaceScatterAdd)
    .Tiling(InplaceScatterAddTilingFunc)
    .TilingParse<InplaceScatterAddCompileInfo>(TilingPrepareForInplaceScatterAdd);
}
