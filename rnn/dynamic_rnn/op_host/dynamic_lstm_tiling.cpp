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
 * \file dynamic_lstm.cc
 * \brief
 */
#include "dynamic_lstm_tiling.h"

using namespace AscendC;

namespace {
  bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  *workspace_size = workspace;
  return true;
}
}
namespace optiling {

void LstmBaseTiling::GetAICoreIntrinsicDtype(fe::PlatFormInfos& platform_info, const std::string& intrinsic_name, bool& value) {
  std::string val;
  (void)platform_info.GetPlatformRes("AICoreintrinsicDtypeMap", intrinsic_name, val);

  if (!val.empty()) {
    OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s : %s", intrinsic_name.c_str(), val.c_str());
    value = true;
  } else {
    value = false;
    OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", intrinsic_name.c_str());
  }

  return;
}

ge::graphStatus LstmBaseTiling::TilingWithAscendC(gert::TilingContext* context) {
  OP_TILING_CHECK(!CheckParamsShape(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check shape fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!CheckParamsDtype(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check dtype fail."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!CheckAttr(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check attr fail."),
                  return ge::GRAPH_FAILED);

  // get op info
  OP_TILING_CHECK(GetOpInfo(context, rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get opinfo fail."),
                  return ge::GRAPH_FAILED);
  // get attr
  OP_TILING_CHECK(GetAttr(context, rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get attr fail."),
                  return ge::GRAPH_FAILED);

  // get UB L1 size
  auto platformInfo = context->GetPlatformInfo();

  int32_t doubleNum = 2;
  rnnParams.sysAicCoreNum = context->GetPlatformInfo()->GetCoreNumByType("AiCore");
  rnnParams.sysAivCoreNum = rnnParams.sysAicCoreNum * doubleNum;
  platformInfo->GetLocalMemSize(fe::LocalMemType::UB, rnnParams.ubSize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L1, rnnParams.l1Size);

  // get matmul tiling data
  OP_TILING_CHECK(
      GetMMTilingData(context, tilingData, rnnParams) != ge::GRAPH_SUCCESS,
      VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get matmul tiling data fail."),
      return ge::GRAPH_FAILED);

  OP_TILING_CHECK(CalcTilingKey(rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling key fail."),
                  return ge::GRAPH_FAILED);

  OP_TILING_CHECK(SetTilingData(context, tilingData, rnnParams) != ge::GRAPH_SUCCESS,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tiling data fail."),
                  return ge::GRAPH_FAILED);

  int64_t workspaceSize = rnnParams.timeStep * rnnParams.batch * 4 * rnnParams.hiddenSize * 4 + 20 * 1024 * 1024;
  auto launchCore = (rnnParams.usedCoreNum + DEFAULT_INDEX_TWO - 1) / DEFAULT_INDEX_TWO;
  context->SetBlockDim(launchCore);  // 24上限
  context->SetTilingKey(rnnParams.tilingKey);
  AddWorkspace(context, workspaceSize);

  return ge::GRAPH_SUCCESS;
}

 bool LstmBaseTiling::CheckParamsDtype(const gert::TilingContext* context) {
  // dtype support list
  std::vector<ge::DataType> supportDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};

  // all params dtype list
  std::vector<ge::DataType> paramsDtype;
  int32_t inputDesc = 2;
  paramsDtype.push_back(context->GetInputDesc(0)->GetDataType());
  paramsDtype.push_back(context->GetInputDesc(1)->GetDataType());
  paramsDtype.push_back(context->GetInputDesc(inputDesc)->GetDataType());

  int32_t inputShapeSize = 4;
  auto inithShape = context->GetOptionalInputShape(inputShapeSize);
  if (inithShape != nullptr) {
    paramsDtype.push_back(context->GetInputDesc(inputShapeSize)->GetDataType());
  }

  for (auto dtype : paramsDtype) {
    OP_TILING_CHECK(
        std::find(supportDtype.begin(), supportDtype.end(), dtype) == supportDtype.end(),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "input dtype error, please check."),
        return false);
  }

  return true;
}

bool LstmBaseTiling::CheckAttr(gert::TilingContext* context) {
  bool ret = CheckAttrOps(context);
  if (ret) {
    ret = CheckAttrTiling(context);
  }

  return ret;
}

ge::graphStatus LstmBaseTiling::GetMMTilingData(gert::TilingContext* context, DynamicRNNTilingData& dynamicTilingData,
                                          DynamicRnnTiling& dynamicRnnParams) {
  int32_t ubSeq = 10;
  int32_t ubNoSeq = 8;
  vector<int64_t> dims = {MIN_BASE_SHAPE};
  uint32_t sigmoidMaxSize = 0;
  uint32_t sigmoidMinSize = 0;
  uint32_t tanhMaxSize = 0;
  uint32_t tanhMinSize = 0;
  GetSigmoidMaxMinTmpSize(ge::Shape(dims), CONST_TWO, false, sigmoidMaxSize, sigmoidMinSize);
  GetTanhMaxMinTmpSize(ge::Shape(dims), CONST_TWO, false, tanhMaxSize, tanhMinSize);
  uint32_t apiUbSize = sigmoidMinSize > tanhMinSize ? sigmoidMinSize : tanhMinSize;
  uint32_t multiple = (apiUbSize + MIN_BASE_BUFFER - 1) / MIN_BASE_BUFFER;
  if (dynamicRnnParams.isSeqLength == 1) {
    dynamicRnnParams.maxUbSize = dynamicRnnParams.ubSize / (ubSeq + multiple);
  } else {
    dynamicRnnParams.maxUbSize = dynamicRnnParams.ubSize / (ubNoSeq + multiple);
  }
  auto dataType = context->GetInputDesc(0)->GetDataType();

  dynamicRnnParams.dataType = dataType;
  dynamicRnnParams.isUseMerged = false;
  dynamicRnnParams.isFullLoad = false;

  auto ret = GetMMTilingDataSplit(context, dynamicTilingData, dynamicRnnParams,
                                  static_cast<matmul_tiling::DataType>(dataType));

  return ret;
}

ge::graphStatus LstmBaseTiling::GetMMTilingDataSplit(const gert::TilingContext* context,
  DynamicRNNTilingData& dynamicTilingData, DynamicRnnTiling& dynamicRnnParams, matmul_tiling::DataType dataType) {
  int32_t hiddenBlock = 4;
  int64_t aivDouble = 2;
  matmul_tiling::MultiCoreMatmulTiling rnnMatmul1;
  dynamicRnnParams.usedCoreNum = context->GetPlatformInfo()->GetCoreNumByType("AiCore") * aivDouble;
  auto ret = rnnMatmul1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetAType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetBType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetCType fail."),
                  return ge::GRAPH_FAILED);

  if (dynamicRnnParams.isBias) {
    rnnMatmul1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    ret = rnnMatmul1.SetBias(true);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetBias fail."),
                    return ge::GRAPH_FAILED);
  }

  ret = rnnMatmul1.SetDim(dynamicRnnParams.sysAivCoreNum);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetDim fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetOrgShape(dynamicRnnParams.timeStep * dynamicRnnParams.batch,
                               dynamicRnnParams.hiddenSize * hiddenBlock, dynamicRnnParams.inputSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetOrgShape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetShape(dynamicRnnParams.timeStep * dynamicRnnParams.batch,
                            dynamicRnnParams.hiddenSize * hiddenBlock, dynamicRnnParams.inputSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 Set single shape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul1.SetBufferSpace(-1, -1, -1);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 SetBufferSpace fail."),
                  return ge::GRAPH_FAILED);

  ret = rnnMatmul1.GetTiling(dynamicTilingData.inputMMParam);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm1 GetTiling fail."),
                  return ge::GRAPH_FAILED);

  matmul_tiling::MultiCoreMatmulTiling rnnMatmul2;
  ret = rnnMatmul2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetAType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetBType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetCType fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetDim(dynamicRnnParams.sysAivCoreNum);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetDim fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetOrgShape(dynamicRnnParams.batch, dynamicRnnParams.hiddenSize * hiddenBlock,
                               dynamicRnnParams.hiddenSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetOrgShape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetShape(dynamicRnnParams.batch, dynamicRnnParams.hiddenSize * hiddenBlock,
                            dynamicRnnParams.hiddenSize);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 Set single shape fail."),
                  return ge::GRAPH_FAILED);
  ret = rnnMatmul2.SetBufferSpace(-1, -1, -1);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 SetBufferSpace fail."),
                  return ge::GRAPH_FAILED);

  ret = rnnMatmul2.GetTiling(dynamicTilingData.hiddenMMParam);
  OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "mm2 GetTiling fail."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LstmBaseTiling::CalcTilingKey(DynamicRnnTiling& dynamicRnnParams) {
  // 判断是否需要切分L0c输出，分次搬入UB
  int64_t tilingKey = 0;

  if (dynamicRnnParams.dataType == 1) {
    tilingKey = static_cast<int64_t>(RNNTilingKey::MM_FP16_SPLIT);
  } else if (dynamicRnnParams.dataType == 0) {
    tilingKey = static_cast<int64_t>(RNNTilingKey::MM_FP32_SPLIT);
  }

  dynamicRnnParams.tilingKey = tilingKey;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LstmBaseTiling::SetTilingData(gert::TilingContext* context, DynamicRNNTilingData& dynamicTilingData,
                                        DynamicRnnTiling& dynamicRnnParams) {
  // 无效数据附初值
  dynamicRnnParams.isHF32 = 0;
  dynamicRnnParams.isCached = 0;
  dynamicRnnParams.cacheLength = 0;

  dynamicTilingData.set_tilingKey(dynamicRnnParams.tilingKey);
  dynamicTilingData.set_usedCoreNum(dynamicRnnParams.usedCoreNum);
  dynamicTilingData.set_timeStep(dynamicRnnParams.timeStep);
  dynamicTilingData.set_batch(dynamicRnnParams.batch);
  dynamicTilingData.set_inputSize(dynamicRnnParams.inputSize);
  dynamicTilingData.set_hiddenSize(dynamicRnnParams.hiddenSize);
  dynamicTilingData.set_isBias(dynamicRnnParams.isBias);
  dynamicTilingData.set_isInithc(dynamicRnnParams.isInithc);
  dynamicTilingData.set_isSeqLength(dynamicRnnParams.isSeqLength);
  dynamicTilingData.set_isHF32(dynamicRnnParams.isHF32);

  dynamicTilingData.set_isCached(dynamicRnnParams.isCached);
  dynamicTilingData.set_cacheLength(dynamicRnnParams.cacheLength);

  dynamicTilingData.set_gateOrder(dynamicRnnParams.gateOrder);
  dynamicTilingData.set_direction(dynamicRnnParams.direction);
  dynamicTilingData.set_isTraining(dynamicRnnParams.isTraining);
  dynamicTilingData.set_cellClip(dynamicRnnParams.cellClip);
  dynamicTilingData.set_forgetBias(dynamicRnnParams.forgetBias);

  gert::TilingData* rnnRawTilingData = context->GetRawTilingData();
  OP_LOGE_IF(rnnRawTilingData == nullptr, ge::GRAPH_FAILED, context->GetNodeType(), "GetRawTilingData failed.");
  OP_TILING_CHECK(dynamicTilingData.GetDataSize() > rnnRawTilingData->GetCapacity(),
                  VECTOR_INNER_ERR_REPORT_TILIING(context, "actual tiling data size %zu > context tiling data size %zu",
                                                  dynamicTilingData.GetDataSize(), rnnRawTilingData->GetCapacity()),
                  return ge::GRAPH_FAILED);
  dynamicTilingData.SaveToBuffer(rnnRawTilingData->GetData(), rnnRawTilingData->GetCapacity());
  rnnRawTilingData->SetDataSize(dynamicTilingData.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling