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
 * \file dynamic_rnn_tiling.cpp
 * \brief
 */
#include "dynamic_rnn_tiling.h"

using namespace AscendC;
namespace {
bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  *workspace_size = workspace;
  return true;
}
}
namespace optiling {
static int32_t GetRnnLibItem(const DynamicRnnCompileInfo* compileInfo, const gert::Shape xStorageShape) {
  for (const auto& tuneShape : compileInfo->tuneShapeList) {
    if (tuneShape.size() < DEFAULT_SHAPE_LIST_SIZE) {
      OP_LOGE("DynamicRnn", "tune_shape_list's size is illegal. it's %lu.", tuneShape.size());
      return DEFAULT_RETURN;
    }
    if ((tuneShape[0] == -1) && (tuneShape[1] == -1)) {
      return static_cast<int32_t>(tuneShape[DEFAULT_INDEX_TWO]);
    }

    if (((tuneShape[1] + CEIL_LENTH16 - 1) / CEIL_LENTH16) == xStorageShape.GetDim(DEFAULT_INDEX_TWO)) {
      return static_cast<int32_t>(tuneShape[DEFAULT_INDEX_TWO]);
    }
  }
  return DEFAULT_RETURN;
}

ge::graphStatus DynamicRNNTiling::GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams) {
  OP_CHECK_IF(context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE,
                  OP_LOGE(context->GetNodeName(), "input shape error."),
                  return ge::GRAPH_FAILED);

  // get x shape
  auto xTensor = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xTensor);
  auto xShape = xTensor->GetStorageShape();

  // get w shape
  auto weightTensor = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, weightTensor);
  auto weightShape = weightTensor->GetStorageShape();

  // get seq_length, init_h, init_C
  auto biasShape = context->GetInputShape(2);
  auto seqShape = context->GetOptionalInputShape(3);
  auto inithShape = context->GetOptionalInputShape(4);
  auto initcShape = context->GetOptionalInputShape(5);

  biasShape != nullptr ? dynamicRnnParams.isBias = 1 : dynamicRnnParams.isBias = 0;
  seqShape != nullptr ? dynamicRnnParams.isSeqLength = 1 : dynamicRnnParams.isSeqLength = 0;
  (inithShape != nullptr && initcShape != nullptr) ? dynamicRnnParams.isInithc = 1 : dynamicRnnParams.isInithc = 0;

  int32_t dim = 2;
  OP_CHECK_IF(xShape.GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(),
                "Dynamicrnn get x shape dim is not 3, please check."), return false);
  OP_CHECK_IF(xShape.GetDim(CONST_TWO) == 0,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn input_size not support 0, please check."), return false);
  OP_CHECK_IF(weightShape.GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(),
                "Dynamicrnn get weight shape dim is not 2, please check."), return false);
  dynamicRnnParams.timeStep = static_cast<int64_t>(xShape.GetDim(0));
  dynamicRnnParams.batch = static_cast<int64_t>(xShape.GetDim(1));
  dynamicRnnParams.inputSize = static_cast<int64_t>(xShape.GetDim(dim));
  dynamicRnnParams.hiddenSize = static_cast<int64_t>(weightShape.GetDim(0)) - dynamicRnnParams.inputSize;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicRNNTiling::GetAttr(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams) {
  // get attr
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  // get gate_order
  const char* gateOrder = attrs->GetAttrPointer<char>(10);
  OPS_CHECK_NULL_WITH_CONTEXT(context, gateOrder);
  if (strcmp(gateOrder, "ijfo") == 0) {
    dynamicRnnParams.gateOrder = static_cast<int64_t>(GateOrder::IJFO);
  } else {
    dynamicRnnParams.gateOrder = static_cast<int64_t>(GateOrder::IFJO);
  }

  // get direction
  const char* direction = attrs->GetAttrPointer<char>(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, direction);
  if (strcmp(direction, "UNIDIRECTIONAL") == 0) {
    dynamicRnnParams.direction = 0;
  } else {
    dynamicRnnParams.direction = 1;
  }

  // get cell_clip
  const float* cellClip = attrs->GetAttrPointer<float>(5);
  OPS_CHECK_NULL_WITH_CONTEXT(context, cellClip);
  dynamicRnnParams.cellClip = *cellClip;

  // get forget_bias
  const float* forgetBias = attrs->GetAttrPointer<float>(9);
  OPS_CHECK_NULL_WITH_CONTEXT(context, forgetBias);
  dynamicRnnParams.forgetBias = *forgetBias;

  // get is_training
  const bool* isTraining = attrs->GetAttrPointer<bool>(11);
  OPS_CHECK_NULL_WITH_CONTEXT(context, isTraining);
  dynamicRnnParams.isTraining = *isTraining;

  return ge::GRAPH_SUCCESS;
}

bool DynamicRNNTiling::CheckParamsShape(gert::TilingContext* context) {
  // get input shape
  auto xInput = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, xInput, false);
  auto xShape = xInput->GetStorageShape();
  // get wight shape
  auto wInput = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, wInput, false);
  auto wShape = wInput->GetStorageShape();

  // get bias shape
  auto bInput = context->GetInputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, bInput, false);
  auto bShape = bInput->GetStorageShape();

  // get output y shape
  auto outputY = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, outputY, false);
  auto outputShape = outputY->GetStorageShape();

  // check dim num
  OP_CHECK_IF(xShape.GetDimNum() != 3,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get x shape dim is not 3, please check."), return false);
  OP_CHECK_IF(wShape.GetDimNum() != 2,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get w shape dim is not 2, please check."), return false);
  OP_CHECK_IF(bShape.GetDimNum() != 1,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get b shape dim is not 1, please check."), return false);

  OP_CHECK_IF(outputShape.GetDimNum() != 3,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get output shape dim is not 3, please check."), return false);

  // check batch dim
  OP_CHECK_IF(xShape.GetDim(1) != outputShape.GetDim(1),
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get x, output batch is not equal, please check."), return false);

  // check x/w input_size dim
  OP_CHECK_IF(wShape.GetDim(0) != xShape.GetDim(2) + outputShape.GetDim(2),
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get w shape dim0 is wrong, please check."), return false);

  // check hidden dim
  OP_CHECK_IF(wShape.GetDim(1) != 4 * outputShape.GetDim(2),
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get w shape dim1 is wrong, please check."), return false);
  OP_CHECK_IF(bShape.GetDim(0) != 4 * outputShape.GetDim(2),
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get b shape dim0 is wrong, please check."), return false);
  bool ret = CheckInitParamsShape(context);

  return ret;
}

bool DynamicRNNTiling::CheckInitParamsShape(gert::TilingContext* context) {
  auto seqInput = context->GetOptionalInputShape(3);
  auto inithInput = context->GetOptionalInputShape(4);
  auto initcInput = context->GetOptionalInputShape(5);

  // get output y shape
  auto outputY = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, outputY, false);
  auto outputShape = outputY->GetStorageShape();

  OP_CHECK_IF(outputShape.GetDimNum() != 3,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn get output shape dim is not 3, please check."), return false);

  if (inithInput != nullptr && initcInput != nullptr) {
    auto inithShape = inithInput->GetStorageShape();
    auto initcShape = initcInput->GetStorageShape();
    OP_CHECK_IF(inithShape.GetDimNum() != 3,
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn get init_h shape dim is not 3, please check."), return false);
    OP_CHECK_IF(initcShape.GetDimNum() != 3,
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn get init_c shape dim is not 3, please check."), return false);
    OP_CHECK_IF(inithShape.GetDim(1) != outputShape.GetDim(1),
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn init_h output batch is not equal, please check."), return false);
    OP_CHECK_IF(initcShape.GetDim(1) != outputShape.GetDim(1),
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn init_c output batch is not equal, please check."), return false);
    OP_CHECK_IF(inithShape.GetDim(2) != outputShape.GetDim(2),
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn init_h output hidden is not equal, please check."), return false);
    OP_CHECK_IF(initcShape.GetDim(2) != outputShape.GetDim(2),
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn init_c output hidden is not equal, please check."), return false);
  }

  if (seqInput != nullptr) {
    auto seqShape = seqInput->GetStorageShape();
    OP_CHECK_IF(seqShape.GetDimNum() != 3,
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn get seq shape dim is not 3, please check."), return false);
    OP_CHECK_IF(seqShape.GetDim(0) != outputShape.GetDim(0),
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn seq and x T is not equal, please check."), return false);
    OP_CHECK_IF(seqShape.GetDim(1) != outputShape.GetDim(1),
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn seq output batch is not equal, please check."), return false);
    OP_CHECK_IF(seqShape.GetDim(2) != outputShape.GetDim(2),
                    OP_LOGE(context->GetNodeName(),
                    "Dynamicrnn seq output hidden is not equal, please check."), return false);
  }

  return true;
}

bool DynamicRNNTiling::CheckAttrTiling(gert::TilingContext* context) {
  // get attr
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
  const int* numProj = attrs->GetAttrPointer<int>(6);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, numProj, false);
  const bool* timeMajor = attrs->GetAttrPointer<bool>(7);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, timeMajor, false);
  const char* activation = attrs->GetAttrPointer<char>(8);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, activation, false);
  const float* forgetBias = attrs->GetAttrPointer<float>(9);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, forgetBias, false);
  const char* gateOrder = attrs->GetAttrPointer<char>(10);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, gateOrder, false);
  const bool* isTraining = attrs->GetAttrPointer<bool>(11);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, isTraining, false);

  OP_CHECK_IF(*numProj != 0,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn attr num_proj only support 0, please check."), return false);
  OP_CHECK_IF(!(*timeMajor),
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn attr time_major only support true, please check."), return false);

  std::vector<std::string> supportGateOrder = {"ifjo", "ijfo"};
  OP_CHECK_IF(std::find(supportGateOrder.begin(), supportGateOrder.end(), gateOrder) == supportGateOrder.end(),
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn attr gate_order is not support, please check."), return false);

  OP_CHECK_IF(strcmp(activation, "tanh") != 0,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn attr activation is not support, please check."), return false);

  return true;
}

bool DynamicRNNTiling::CheckAttrOps(gert::TilingContext* context) {
  // get attr
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
  const char* cellType = attrs->GetAttrPointer<char>(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellType, false);
  const char* direction = attrs->GetAttrPointer<char>(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, direction, false);
  const int* cellDepth = attrs->GetAttrPointer<int>(2);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellDepth, false);
  const bool* usePeephole = attrs->GetAttrPointer<bool>(3);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, usePeephole, false);
  const float* keepProb = attrs->GetAttrPointer<float>(4);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, keepProb, false);
  const float* cellClip = attrs->GetAttrPointer<float>(5);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellClip, false);

  OP_CHECK_IF(*cellDepth != 1,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn attr cell_depth only support 1, please check."), return false);
  OP_CHECK_IF(*usePeephole,
                  OP_LOGE( context->GetNodeName(),
                  "Dynamicrnn attr use_peephole only support false, please check."), return false);
  std::vector<std::string> supportDirection = {"UNIDIRECTIONAL", "REDIRECTIONAL"};
  OP_CHECK_IF(std::find(supportDirection.begin(), supportDirection.end(), direction) == supportDirection.end(),
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn attr direction is not support, please check."), return false);
  OP_CHECK_IF(strcmp(cellType, "LSTM") != 0,
                  OP_LOGE(context->GetNodeName(),
                  "Dynamicrnn attr cell_type is not support, please check."), return false);

  return true;
}

ge::graphStatus TilingForDynamicRNN(gert::TilingContext* context) {
  // set sync op batchmode
  context->SetScheduleMode(SCHEDULE_MODE);
  // 910B/C AscendC
  bool supportL0c2out = false;
  DynamicRNNTiling rnnTiling;
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  rnnTiling.GetAICoreIntrinsicDtype(*platformInfo, "Intrinsic_fix_pipe_l0c2out", supportL0c2out);
  if (supportL0c2out) {
    OP_CHECK_IF(rnnTiling.TilingWithAscendC(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "tiling with ascendc have error."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
  }

  OP_CHECK_IF(context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE,
                  OP_LOGE(context->GetNodeName(), "input shape error."),
                  return ge::GRAPH_FAILED);

  auto xShape = context->GetInputShape(0);
  OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "input_x is null."),
                  return ge::GRAPH_FAILED);

  const auto& xStorageShape = optiling::EnsureNotScalar(xShape->GetStorageShape());
  OP_CHECK_IF(xStorageShape.GetDimNum() < DEFAULT_XSHAPE_SIZE,
                  OP_LOGE(context->GetNodeName(), "xStorageShape error."),
                  return ge::GRAPH_FAILED);

  auto compileInfo = context->GetCompileInfo<DynamicRnnCompileInfo>();
  OP_CHECK_IF(compileInfo == nullptr,
                  OP_LOGE(context->GetNodeName(), "compileInfo is null."),
                  return ge::GRAPH_FAILED);

  auto runParams = context->GetTilingData<DynamicRnnTilingData>();
  OP_CHECK_IF(runParams == nullptr, OP_LOGE(context->GetNodeName(), "runParams is null."),
                  return ge::GRAPH_FAILED);

  runParams->sequenceLength = static_cast<int32_t>(xStorageShape.GetDim(0));
  runParams->dynamicRnnBatch = static_cast<int32_t>(xStorageShape.GetDim(DEFAULT_INDEX_TWO));
  runParams->chequeIndex = GetRnnLibItem(compileInfo, xStorageShape);
  OP_CHECK_IF(runParams->chequeIndex == DEFAULT_RETURN,
                  OP_LOGE(context->GetNodeName(), "DynamicRnnTiling has no matched schedule."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetTilingKey(runParams->chequeIndex) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "DynamicRnnTiling set tiling fail."),
                  return ge::GRAPH_FAILED);

  context->SetBlockDim(BLOCK_DIM);
  AddWorkspace(context, WORKSPACE_SIZE);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDynamicRNN(gert::TilingParseContext* context) {
  // 910B/C AscendC
  bool supportL0c2out = false;
  DynamicRNNTiling rnnTiling;
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  rnnTiling.GetAICoreIntrinsicDtype(*platformInfo, "Intrinsic_fix_pipe_l0c2out", supportL0c2out);
  if (supportL0c2out) {
    return ge::GRAPH_SUCCESS;
  }

  auto compileInfo = context->GetCompiledInfo<DynamicRnnCompileInfo>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

  std::unique_ptr<nlohmann::json> parsedObjectCinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsedObjectCinfo);
  const nlohmann::json& vars = (*parsedObjectCinfo)["vars"];
  OP_CHECK_IF(vars.empty(),
                  OP_LOGE(context->GetNodeName(), "Failed to get vars in parsedObjectCinfo."),
                  return ge::GRAPH_FAILED);

  OP_CHECK_IF(!ReadCompileItem(vars, "tune_shape_list", compileInfo->tuneShapeList),
                  OP_LOGE(context->GetNodeName(), "DynamicRNN, get tune_shape_list failed."),
                  return ge::GRAPH_FAILED);

  OP_CHECK_IF(compileInfo->tuneShapeList.empty(),
                  OP_LOGE(context->GetNodeName(), "DynamicRNN, tune_shape_list is empty."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicRNN).Tiling(TilingForDynamicRNN).TilingParse<DynamicRnnCompileInfo>(TilingPrepareForDynamicRNN);
}  // namespace optiling
