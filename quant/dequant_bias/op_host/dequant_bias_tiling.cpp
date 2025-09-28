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
 * \file dequant_bias_tiling.cpp
 * \brief dequant_bias_tiling source file
 */

#include "dequant_bias_tiling.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
const static int64_t MAX_BINARY_ADD_BUFFER_CNT = 64;
const static int64_t MAX_BINARY_ADD_BUFFER_CNT_WITHOUT_DROPLESS = 8;
const static int64_t MAX_EXPONENT_OF_BINARY = 6;
const static int64_t MAX_EXPONENT_OF_BINARY_WITHOUT_DROPLESS = 3;
const static int64_t BOUND_K_FOR_BINARY = 128;
const static int64_t BOUND_K_FOR_BINARY_WITHOUT_DROPLESS = 16;
const static int64_t ELEMENT_ALIGN_SIZE = 32;
const static int64_t MAX_DATACOPY_SIZE = 65536;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static int32_t SIZE_16 = 16;
const static int32_t ALIGN_SIZE_32 = 32;
const static int32_t LENGTH_1024 = 1024;
const static int32_t BUFFER_NUM = 2;
const static int32_t SIZE_1 = 1;
const static int32_t SIZE_2 = 2;
const static int32_t SIZE_3 = 3;
const static int32_t SIZE_27 = 27;
const static int32_t SIZE_25000 = 25000;
const static int32_t PART_LEN = 8192;

inline uint32_t CeilDiv(uint32_t value, uint32_t factor) {
  if (factor == 0) {
    return value;
  }
  return (value + factor - 1) / factor;
}

class DequantBiasTiling : public TilingBaseClass {
 public:
  explicit DequantBiasTiling(gert::TilingContext* context) : TilingBaseClass(context) {
    Reset();
  }
  ~DequantBiasTiling() override = default;

  void Reset(gert::TilingContext* context) override {
    TilingBaseClass::Reset(context);
    Reset();
  }

 protected:
  bool IsCapable() override {
    return true;
  }

  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
  ge::graphStatus GetPlatformInfo() override;
  // 2、获取INPUT/OUTPUT/ATTR信息
  ge::graphStatus GetShapeAttrsInfo() override;
  // 3、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;
  // 4、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override;
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 6、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;
  void Reset();

 private:
  ge::graphStatus CheckParamsValidity1(const gert::Shape& xShape, const gert::Shape& weightScaleShape,
                                      const gert::StorageShape* activateScaleShape, const gert::StorageShape* biasShape,
                                      const gert::Shape& yShape);
  ge::graphStatus CheckParamsValidity2(const gert::Shape& xShape, const gert::Shape& weightScaleShape,
                                      const gert::StorageShape* activateScaleShape, const gert::StorageShape* biasShape,
                                      const gert::Shape& yShape);
  ge::graphStatus CheckParamsValidity3();
  ge::graphStatus TilingGradCompute();
  void TilingSplitCore();
  void ShowTilingData();

  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t nAlign = 0;
  uint32_t aivNum = 0;
  uint32_t outputDtype = 1;

  ge::DataType biasDtype = ge::DT_FLOAT;
  ge::DataType asDtype = ge::DT_FLOAT;

  uint32_t asExist = 0;
  uint32_t biasExist = 0;

  const char* opName = "";
  DequantBiasTilingData dequantBiasTilingData;
};

void DequantBiasTiling::Reset() {
  opName = nullptr;
  return;
}

ge::graphStatus DequantBiasTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"),
                  return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  aivNum = ascendcPlatform.GetCoreNumAiv();
  aicoreParams_.blockDim = aivNum;
  uint64_t ubSizePlatForm;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
  aicoreParams_.ubSize = ubSizePlatForm;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantBiasTiling::CheckParamsValidity1(const gert::Shape& xShape, const gert::Shape& weightScaleShape,
                                                       const gert::StorageShape* activateScaleShape,
                                                       const gert::StorageShape* biasShape, const gert::Shape& yShape) {
  // attr属性校验
  if (outputDtype != SIZE_1 && outputDtype != SIZE_27) {
    OP_LOGE(context_->GetNodeName(), "Attr outputDtype should be 1 or 27.");
    return ge::GRAPH_FAILED;
  }

  // shape校验
  size_t xDimNum = xShape.GetDimNum();
  if (xDimNum != DIM_TWO) {
    OP_LOGE(context_->GetNodeName(), "The dim number of x should be 2.");
    return ge::GRAPH_FAILED;
  }

  size_t weightScaleDimNum = weightScaleShape.GetDimNum();
  if (weightScaleDimNum != DIM_ONE) {
    OP_LOGE(context_->GetNodeName(), "The dim number of weightScale should be 1.");
    return ge::GRAPH_FAILED;
  }

  if (activateScaleShape != nullptr && activateScaleShape->GetStorageShape().GetDimNum() != DIM_ONE) {
    OP_LOGE(context_->GetNodeName(), "The dim number of activateScale should be 1.");
    return ge::GRAPH_FAILED;
  }

  if (biasShape != nullptr && biasShape->GetStorageShape().GetDimNum() != DIM_ONE) {
    OP_LOGE(context_->GetNodeName(), "The dim number of bias should be 1.");
    return ge::GRAPH_FAILED;
  }

  size_t outDimNum = yShape.GetDimNum();
  if (outDimNum != DIM_TWO) {
    OP_LOGE(context_->GetNodeName(), "The dim number of y should be 2.");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantBiasTiling::CheckParamsValidity2(const gert::Shape& xShape, const gert::Shape& weightScaleShape,
                                                       const gert::StorageShape* activateScaleShape,
                                                       const gert::StorageShape* biasShape, const gert::Shape& yShape) {
    // shape 约束
  if (weightScaleShape.GetDim(0) != xShape.GetDim(1)) {
    OP_LOGE(context_->GetNodeName(), "dim 1 of x and weightScale should be same.");
    return ge::GRAPH_FAILED;
  }

  if (activateScaleShape != nullptr && activateScaleShape->GetStorageShape().GetShapeSize() > 0 &&
      activateScaleShape->GetStorageShape().GetDim(0) != xShape.GetDim(0)) {
    OP_LOGE(context_->GetNodeName(), "dim 0 of x and activateScale %zu and %zu should be same.",
            activateScaleShape->GetStorageShape().GetDim(0), xShape.GetDim(0));
    return ge::GRAPH_FAILED;
  }

  if (biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() > 0 &&
      biasShape->GetStorageShape().GetDim(0) != xShape.GetDim(1)) {
    OP_LOGE(context_->GetNodeName(), "dim 1 of x %zu and dim 0 of bias %zu should be same.",
            biasShape->GetStorageShape().GetDim(0), xShape.GetDim(1));
    return ge::GRAPH_FAILED;
  }

  if (xShape.GetDim(0) != yShape.GetDim(0)) {
    OP_LOGE(context_->GetNodeName(), "dim 0 of input and output should be same.");
    return ge::GRAPH_FAILED;
  }

  if (xShape.GetDim(1) != yShape.GetDim(1)) {
    OP_LOGE(context_->GetNodeName(), "dim 1 of input and output should be same.");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantBiasTiling::CheckParamsValidity3(){
  if(asExist == 1){
    auto asDesc = context_->GetOptionalInputDesc(2);  // x idx
    OP_CHECK_NULL_WITH_CONTEXT(context_, asDesc);
    asDtype = asDesc->GetDataType();
    if (asDtype != ge::DT_FLOAT) {
      OP_LOGE(context_->GetNodeName(), "activate_scale's dtype should be float");
      return ge::GRAPH_FAILED;
    }
  }

  if (biasExist == 1){
    if ((biasDtype != ge::DT_BF16) && (biasDtype != ge::DT_FLOAT) && (biasDtype != ge::DT_FLOAT16) && (biasDtype != ge::DT_INT32)) {
      OP_LOGE(context_->GetNodeName(), "bias's dtype should be bf16 or float or fp16 or int32");
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantBiasTiling::GetShapeAttrsInfo() {
  opName = context_->GetNodeName();
  OP_LOGD(context_);

  // 获取输入shape
  auto xShapePtr = context_->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
  const gert::Shape xShape = xShapePtr->GetStorageShape();
  auto weightScaleShapePtr = context_->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, weightScaleShapePtr);
  const gert::Shape weightScaleShape = weightScaleShapePtr->GetStorageShape();

  auto activateScaleShape = context_->GetOptionalInputShape(2);  // 2: activate_scale idx
  auto biasShape = context_->GetOptionalInputShape(3);           // 3: bias idx

  auto yShapePtr = context_->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
  const gert::Shape yShape = yShapePtr->GetStorageShape();

  // 获取输入属性
  auto attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const int64_t* outpurDtypePtr = attrs->GetAttrPointer<int64_t>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, outpurDtypePtr);
  outputDtype = *outpurDtypePtr;

  // 参数校验
  ge::graphStatus res = CheckParamsValidity1(xShape, weightScaleShape, activateScaleShape, biasShape, yShape);
  if (res != ge::GRAPH_SUCCESS) {
    return res;
  }

  res = CheckParamsValidity2(xShape, weightScaleShape, activateScaleShape, biasShape, yShape);
  if (res != ge::GRAPH_SUCCESS) {
    return res;
  }

  // 设置tilingData基本参数
  M = xShape.GetDim(0);
  N = xShape.GetDim(1);
  nAlign = (N * sizeof(float) + 63) / 64 * 64 / sizeof(float);  // 63, 64 , align with 64 bytes

  asExist = (activateScaleShape == nullptr || activateScaleShape->GetStorageShape().GetShapeSize() == 0) ? 0 : 1;
  biasExist = (biasShape == nullptr || biasShape->GetStorageShape().GetShapeSize() == 0) ? 0 : 1;
  if (biasExist == 1) {
    auto biasDesc = context_->GetOptionalInputDesc(3);  // 3: bias idx
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
    biasDtype = biasDesc->GetDataType();
  }

  res = CheckParamsValidity3();
  if (res != ge::GRAPH_SUCCESS) {
    return res;
  }

  // 设置tilingdata参数
  dequantBiasTilingData.set_M(M);
  dequantBiasTilingData.set_N(N);
  dequantBiasTilingData.set_nAlign(nAlign);
  dequantBiasTilingData.set_asExist(asExist);

  return ge::GRAPH_SUCCESS;
}

void DequantBiasTiling::ShowTilingData() {
  OP_LOGI(opName,
          "DequantBiasTilingData is M:%u, N:%u, nAlign:%u, asExist:%u, needCoreNum:%u, "
          "perCoreRow:%u, tailCoreRow:%u, inBufferSize:%u, wsBufferSize:%u, asBufferSize:%u, "
          "biasBufferSize:%u, perCoreLoopRow:%u, perCoreTailLoopRow:%u, perCoreLoops:%u, "
          "tailCoreLoopRow:%u, tailCoreTailLoopRow:%u, tailCoreLoops:%u",
          dequantBiasTilingData.get_M(), dequantBiasTilingData.get_N(), dequantBiasTilingData.get_nAlign(),
          dequantBiasTilingData.get_asExist(), dequantBiasTilingData.get_needCoreNum(),
          dequantBiasTilingData.get_perCoreRow(), dequantBiasTilingData.get_tailCoreRow(),
          dequantBiasTilingData.get_inBufferSize(), dequantBiasTilingData.get_wsBufferSize(),
          dequantBiasTilingData.get_asBufferSize(), dequantBiasTilingData.get_biasBufferSize(),
          dequantBiasTilingData.get_perCoreLoopRow(), dequantBiasTilingData.get_perCoreTailLoopRow(),
          dequantBiasTilingData.get_perCoreLoops(), dequantBiasTilingData.get_tailCoreLoopRow(),
          dequantBiasTilingData.get_tailCoreTailLoopRow(), dequantBiasTilingData.get_tailCoreLoops());
}

ge::graphStatus DequantBiasTiling::DoOpTiling() {
  TilingSplitCore();
  auto ret = TilingGradCompute();
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }
  ShowTilingData();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantBiasTiling::DoLibApiTiling() {
  return ge::GRAPH_SUCCESS;
}

uint64_t DequantBiasTiling::GetTilingKey() const {
  uint64_t tilingKey = tilingKey_;
  // activate_scale, 不存在或者存在时shape为[1]，默认0；存在shape为[M]时，配置为1
  // TD：当前不支持[1]场景，所以先将剩余两个场景固定为1
  tilingKey += 100;  // 100: 01 00

  if (biasExist == 1) {
    tilingKey += 10;  // 10: 0 1

    if (biasDtype == ge::DT_FLOAT) {
      tilingKey += SIZE_1;
    } else if (biasDtype == ge::DT_FLOAT16) {
      tilingKey += SIZE_2;
    } else if (biasDtype == ge::DT_BF16) {
      tilingKey += SIZE_3;
    }
  }

  return tilingKey;
}

ge::graphStatus DequantBiasTiling::GetWorkspaceSize() {
  // 计算workspace大小，无需workspace临时空间，不存在多核同步，预留固定大小即可
  workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantBiasTiling::PostTiling() {
  context_->SetBlockDim(aivNum);
  size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
  currentWorkspace[0] = workspaceSize_;
  auto tilingData = context_->GetRawTilingData();
  OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
  dequantBiasTilingData.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
  tilingData->SetDataSize(dequantBiasTilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

void DequantBiasTiling::TilingSplitCore() {
  tilingKey_ = 10000UL;

  uint32_t M_ = dequantBiasTilingData.get_M();
  uint32_t perCoreRow = Ops::Base::CeilDiv(M_, aivNum);                       // 单核处理最大行
  uint32_t needCoreNum = Ops::Base::CeilDiv(M_, perCoreRow);                  // 实际使用核数
  uint32_t set_tailCoreRow = M_ - (needCoreNum - 1) * perCoreRow;  // 尾核处理行

  dequantBiasTilingData.set_needCoreNum(needCoreNum);
  dequantBiasTilingData.set_perCoreRow(perCoreRow);
  dequantBiasTilingData.set_tailCoreRow(set_tailCoreRow);
}

ge::graphStatus DequantBiasTiling::TilingGradCompute() {
  uint32_t perCoreRow = dequantBiasTilingData.get_perCoreRow();
  uint32_t tailCoreRow = dequantBiasTilingData.get_tailCoreRow();

  // in buffer
  uint32_t inBuffSize = nAlign * sizeof(float) * BUFFER_NUM;  // DB

  // weight_scale buffer
  uint32_t weightScaleBuffSize = nAlign * sizeof(float);

  // activate_scale buffer
  uint32_t asBuffSize = 0;
  if (asExist) {
    asBuffSize = (perCoreRow * sizeof(float) + 31) / 32 * 32;  // 31,32 algin with 32 Bytes
  }

  uint32_t biasBuffSize = biasExist ? weightScaleBuffSize : 0;

  uint32_t perCoreLoopRow = 1;  // 一次处理一行
  uint32_t tailCoreLoopRow = 1;
  uint32_t perCoreLoops = perCoreRow;
  uint32_t tailCoreLoops = tailCoreRow;
  uint32_t perCoreTailLoopRow = 1;   // 每次最后处理一行
  uint32_t tailCoreTailLoopRow = 1;  // 每次最后处理一行
  if (N > PART_LEN) {
    inBuffSize = PART_LEN * sizeof(float) * BUFFER_NUM ;
    weightScaleBuffSize = PART_LEN * sizeof(float);
    biasBuffSize = PART_LEN * sizeof(float);
    perCoreLoopRow = 1;
    tailCoreLoopRow = 1;
  }

  uint32_t totalBuffSize = inBuffSize + weightScaleBuffSize + asBuffSize + biasBuffSize;
  if (totalBuffSize >= aicoreParams_.ubSize || M > SIZE_25000) {
    OP_LOGE(context_->GetNodeName(), "input x size is too large.");
    return ge::GRAPH_FAILED;
  }

  uint32_t releaseUbSizeForX = aicoreParams_.ubSize - (weightScaleBuffSize + asBuffSize + biasBuffSize);
  perCoreLoopRow = std::min(releaseUbSizeForX / inBuffSize, perCoreRow);
  tailCoreLoopRow = std::min(releaseUbSizeForX / inBuffSize, tailCoreRow);
  perCoreLoops = Ops::Base::CeilDiv(perCoreRow, perCoreLoopRow);
  perCoreTailLoopRow = perCoreRow - perCoreLoopRow * (perCoreLoops - 1);
  tailCoreLoops = Ops::Base::CeilDiv(tailCoreRow, tailCoreLoopRow);
  tailCoreTailLoopRow = tailCoreRow - tailCoreLoopRow * (tailCoreLoops - 1);

  inBuffSize *= perCoreLoopRow;

  // 设置tilingData
  dequantBiasTilingData.set_inBufferSize(inBuffSize);
  dequantBiasTilingData.set_wsBufferSize(weightScaleBuffSize);
  dequantBiasTilingData.set_asBufferSize(asBuffSize);
  dequantBiasTilingData.set_biasBufferSize(biasBuffSize);
  dequantBiasTilingData.set_perCoreLoopRow(perCoreLoopRow);
  dequantBiasTilingData.set_perCoreTailLoopRow(perCoreTailLoopRow);
  dequantBiasTilingData.set_perCoreLoops(perCoreLoops);
  dequantBiasTilingData.set_tailCoreLoopRow(tailCoreLoopRow);
  dequantBiasTilingData.set_tailCoreTailLoopRow(tailCoreTailLoopRow);
  dequantBiasTilingData.set_tailCoreLoops(tailCoreLoops);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForDequantBias(gert::TilingContext* context) {
  DequantBiasTiling tiling(context);
  return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForDequantBias(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DequantBias)
    .Tiling(TilingForDequantBias)
    .TilingParse<DequantBiasCompileInfo>(TilingPrepareForDequantBias);
}  // namespace optiling