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
 * \file dynamic_rnnv2_tiling.cpp
 * \brief
 */
#include "dynamic_rnnv2_tiling.h"


using namespace ge;
using namespace AscendC;
namespace {
template <typename T>

bool AddWorkspace(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  *workspace_size = workspace;
  return true;
}
}

namespace optiling {
static int32_t GetRnnV2LibItem(const gert::TilingContext* context, const DynamicRNNV2CompileInfo* compileInfo,
                               const gert::Shape xShape) {
  for (const auto& tuneShape : compileInfo->tuneShapeList) {
    OP_CHECK_IF(tuneShape.size() < DST_SHAPE_SIZE,
                    OP_LOGE(context->GetNodeName(),
                                                    "tuneShapeList's size is illegal. it's %lu.", tuneShape.size()),
                    return DEFAULT_RETURN);

    if ((tuneShape[0] == -1) && (tuneShape[1] == -1)) {
      OP_LOGI(context->GetNodeName(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, The corresponding schedule is %lu",
              tuneShape[DEFAULT_INDEX_TWO]);
      return static_cast<int32_t>(tuneShape[DEFAULT_INDEX_TWO]);
    }

    if ((tuneShape[0] == xShape.GetDim(0)) && ((tuneShape[1] / NUM_SIXTEEN) == xShape.GetDim(DEFAULT_INDEX_TWO))) {
      OP_LOGI(context->GetNodeName(), "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, The corresponding schedule is %lu",
              tuneShape[DEFAULT_INDEX_TWO]);
      return static_cast<int32_t>(tuneShape[DEFAULT_INDEX_TWO]);
    }
  }
  return DEFAULT_RETURN;
}

bool DynamicRNNV2Tiling::CheckInitParamsShape(gert::TilingContext* context) {
  auto bInput = context->GetOptionalInputShape(3);
  auto seqInput = context->GetOptionalInputShape(4);
  auto inithInput = context->GetOptionalInputShape(5);
  auto initcInput = context->GetOptionalInputShape(6);
  auto outputShape = context->GetOutputShape(0)->GetStorageShape();

  if (bInput != nullptr) {
    auto bShape = bInput->GetStorageShape();
    OP_CHECK_IF(bShape.GetDimNum() != 1,
                    OP_LOGE(context->GetNodeName(),
                    "DynamicrnnV2 get bias shape dim is not 1, please check."), return false);

    OP_CHECK_IF(bShape.GetDim(0) != 4 * outputShape.GetDim(2),
                    OP_LOGE(context->GetNodeName(),
                    "DynamicrnnV2 bias and output is not equal, please check."), return false);
  }

  OP_CHECK_IF(seqInput != nullptr,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 do not support seqlength, please check."), return false);

  OPS_CHECK_NULL_WITH_CONTEXT(context, inithInput);
  OPS_CHECK_NULL_WITH_CONTEXT(context, initcInput);
  auto inithShape = inithInput->GetStorageShape();
  auto initcShape = initcInput->GetStorageShape();

  OP_CHECK_IF(inithShape.GetDimNum() != 3,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get init_h shape dim is not 3, please check."), return false);

  OP_CHECK_IF(initcShape.GetDimNum() != 3,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get init_c shape dim is not 3, please check."), return false);

  OP_CHECK_IF(inithShape.GetDim(1) != outputShape.GetDim(1),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 init_h and output batch is unequal, please check."), return false);

  OP_CHECK_IF(initcShape.GetDim(1) != outputShape.GetDim(1),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 init_c and output batch is unequal, please check."), return false);

  OP_CHECK_IF(inithShape.GetDim(2) != outputShape.GetDim(2),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 init_h and output hidden is unequal, please check."), return false);

  OP_CHECK_IF(initcShape.GetDim(2) != outputShape.GetDim(2),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 init_c and output hidden is unequal, please check."), return false);

  return true;
}

bool DynamicRNNV2Tiling::CheckAttrOps(gert::TilingContext* context) {
  // get attr
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
  const float* cellClip = attrs->GetAttrPointer<float>(5);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellClip, false);
  const float* keepProb = attrs->GetAttrPointer<float>(4);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, keepProb, false);
  const bool* usePeephole = attrs->GetAttrPointer<bool>(3);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, usePeephole, false);
  const int* cellDepth = attrs->GetAttrPointer<int>(2);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellDepth, false);
  const char* direction = attrs->GetAttrPointer<char>(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, direction, false);
  const char* cellType = attrs->GetAttrPointer<char>(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, cellType, false);

  OP_CHECK_IF(*cellDepth != 1,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr cell_depth only support 1, please check."), return false);

  OP_CHECK_IF(std::abs(*keepProb - 1) >= std::numeric_limits<float>::epsilon(),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr keep_prob only support 1.0, please check."), return false);

  OP_CHECK_IF(std::abs(*cellClip + 1) >= std::numeric_limits<float>::epsilon(),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr cell_clip only support -1.0, please check."), return false);

  OP_CHECK_IF(*usePeephole,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr use_peephole only support false, please check."), return false);

  std::vector<std::string> supportDirection = {"UNIDIRECTIONAL", "REDIRECTIONAL"};
  OP_CHECK_IF(std::find(supportDirection.begin(), supportDirection.end(), direction) == supportDirection.end(),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr direction is not support, please check."), return false);

  OP_CHECK_IF(strcmp(cellType, "LSTM") != 0,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr cell_type is not support, please check."), return false);
  return true;
}

bool DynamicRNNV2Tiling::CheckAttrTiling(gert::TilingContext* context) {
  // get attr
  auto attrs = context->GetAttrs();
  const int* numProj = attrs->GetAttrPointer<int>(6);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, numProj, false);
  const bool* timeMajor = attrs->GetAttrPointer<bool>(7);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, timeMajor, false);
  const char* activation = attrs->GetAttrPointer<char>(8);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, activation, false);
  const char* recurrentActivation = attrs->GetAttrPointer<char>(9);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, recurrentActivation, false);
  const float* forgetBias = attrs->GetAttrPointer<float>(10);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, forgetBias, false);
  const char* gateOrder = attrs->GetAttrPointer<char>(11);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, gateOrder, false);
  const bool* stateful = attrs->GetAttrPointer<bool>(12);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, stateful, false);
  const char* mergeMode = attrs->GetAttrPointer<char>(13);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, mergeMode, false);
  const bool* isTraining = attrs->GetAttrPointer<bool>(14);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, isTraining, false);

  OP_CHECK_IF(*numProj != 0,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr num_proj only support 0, please check."), return false);
  OP_CHECK_IF(!(*timeMajor),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr time_major only support true, please check."), return false);

  OP_CHECK_IF(*stateful,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr stateful only support false, please check."), return false);

  std::vector<std::string> supportGateOrder = {"ifco", "ijfo"};
  OP_CHECK_IF(std::find(supportGateOrder.begin(), supportGateOrder.end(), gateOrder) == supportGateOrder.end(),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr gate_order is not support, please check."), return false);

  OP_CHECK_IF(strcmp(activation, "tanh") != 0,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr activation is not support, please check."), return false);

  OP_CHECK_IF(strcmp(recurrentActivation, "sigmoid") != 0,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr recurrent_activation is not support, please check."), return false);

  std::vector<std::string> supportMergeMode = {"concat", "add"};
  OP_CHECK_IF(std::find(supportMergeMode.begin(), supportMergeMode.end(), mergeMode) == supportMergeMode.end(),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 attr merge_mode is not support, please check."), return false);
  return true;
}

ge::graphStatus DynamicRNNV2Tiling::GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams) {
  OP_CHECK_IF(context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE,
                  OP_LOGE(context->GetNodeName(), "DynamicrnnV2 input shape error."),
                  return ge::GRAPH_FAILED);

  // get x shape
  auto xTensor = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, xTensor);
  auto xShape = xTensor->GetStorageShape();

  // get w shape
  auto weightHiddenTensor = context->GetInputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT(context, weightHiddenTensor);
  auto weightHiddenShape = weightHiddenTensor->GetStorageShape();

  // get bias seq_length, init_h, init_C
  auto biasShape = context->GetOptionalInputShape(3);
  auto seqShape = context->GetOptionalInputShape(4);
  auto inithShape = context->GetOptionalInputShape(5);
  auto initcShape = context->GetOptionalInputShape(6);

  biasShape != nullptr ? dynamicRnnParams.isBias = 1 : dynamicRnnParams.isBias = 0;
  seqShape != nullptr ? dynamicRnnParams.isSeqLength = 1 : dynamicRnnParams.isSeqLength = 0;
  (inithShape != nullptr && initcShape != nullptr) ? dynamicRnnParams.isInithc = 1 : dynamicRnnParams.isInithc = 0;

  int32_t dim = 2;
  OP_CHECK_IF(xShape.GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(),
                "DynamicrnnV2 get x shape dim is not 3, please check."), return false);
  OP_CHECK_IF(weightHiddenShape.GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(),
                "DynamicrnnV2 get weight_hidden shape dim is not 2, please check."), return false);
  dynamicRnnParams.timeStep = static_cast<int64_t>(xShape.GetDim(0));
  dynamicRnnParams.batch = static_cast<int64_t>(xShape.GetDim(1));
  dynamicRnnParams.inputSize = static_cast<int64_t>(xShape.GetDim(dim));
  dynamicRnnParams.hiddenSize = static_cast<int64_t>(weightHiddenShape.GetDim(0));

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicRNNV2Tiling::GetAttr(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams) {
  // get attr
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  // get gate_order
  const char* gateOrder = attrs->GetAttrPointer<char>(11);
  OPS_CHECK_NULL_WITH_CONTEXT(context, gateOrder);
  if (strcmp(gateOrder, "ijfo") == 0) {
    dynamicRnnParams.gateOrder = static_cast<int64_t>(GateOrder::IJFO);
  } else {
    dynamicRnnParams.gateOrder = static_cast<int64_t>(GateOrder::IFJO);
  }

  // set direction
  dynamicRnnParams.direction = 0;

  // get cell_clip
  const float* cellClip = attrs->GetAttrPointer<float>(5);
  OPS_CHECK_NULL_WITH_CONTEXT(context, cellClip);
  dynamicRnnParams.cellClip = *cellClip;

  // get forget_bias
  const float* forgetBias = attrs->GetAttrPointer<float>(10);
  OPS_CHECK_NULL_WITH_CONTEXT(context, forgetBias);
  dynamicRnnParams.forgetBias = *forgetBias;

  // get is_training
  const bool* isTraining = attrs->GetAttrPointer<bool>(14);
  OPS_CHECK_NULL_WITH_CONTEXT(context, isTraining);
  dynamicRnnParams.isTraining = *isTraining;

  return ge::GRAPH_SUCCESS;
}

bool DynamicRNNV2Tiling::CheckParamsShape(gert::TilingContext* context) {
  // get input shape
  auto xInput = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, xInput, false);
  auto xShape = xInput->GetStorageShape();
  // get weight input shape
  auto wInput = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, wInput, false);
  auto wInputShape = wInput->GetStorageShape();

  // get weight hidden shape
  auto wHidden = context->GetInputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, wHidden, false);
  auto wHiddenShape = wHidden->GetStorageShape();

  // get output y shape
  auto outputY = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, outputY, false);
  auto outputShape = outputY->GetStorageShape();

  // check dim num
  OP_CHECK_IF(xShape.GetDimNum() != 3,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get x shape dim is not 3, please check."), return false);
  OP_CHECK_IF(xShape.GetDim(CONST_TWO) == 0,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 input_size not support 0, please check."), return false);
  OP_CHECK_IF(wInputShape.GetDimNum() != 2,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get weight_input shape dim is not 2, please check."), return false);

  OP_CHECK_IF(wHiddenShape.GetDimNum() != 2,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get weight_hidden shape dim is not 2, please check."), return false);
  OP_CHECK_IF(outputShape.GetDimNum() != 3,
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get output shape dim is not 3, please check."), return false);

  // check batch dim
  OP_CHECK_IF(xShape.GetDim(1) != outputShape.GetDim(1),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get x, output batch is not equal, please check."), return false);

  // check x/w input_size dim
  OP_CHECK_IF(wInputShape.GetDim(0) != xShape.GetDim(2),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get weight_input shape dim0 is wrong, please check."), return false);

  // check hidden dim
  OP_CHECK_IF(wHiddenShape.GetDim(1) != 4 * outputShape.GetDim(2),
                  OP_LOGE(context->GetNodeName(),
                  "DynamicrnnV2 get weight_hidden shape dim1 is wrong, please check."), return false);
  bool ret = CheckInitParamsShape(context);

  return ret;
}

ge::graphStatus Tiling4DynamicRNNV2(gert::TilingContext* context) {
  OP_LOGD(context->GetNodeName(), "DynamicRNNV2Tiling start.");
  // set sync op batchmode
  context->SetScheduleMode(SCHEDULE_MODE);
  // 910B/C AscendC
  bool supportL0c2out = false;
  DynamicRNNV2Tiling rnnv2Tiling;
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  rnnv2Tiling.GetAICoreIntrinsicDtype(*platformInfo, "Intrinsic_fix_pipe_l0c2out", supportL0c2out);
  if (supportL0c2out) {
    OP_CHECK_IF(rnnv2Tiling.TilingWithAscendC(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(),
                                                    "DynamicrnnV2 with ascendc have error."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
  }

  if (context->GetComputeNodeInputNum() < DST_INPUT_SIZE) {
    OP_LOGE(context->GetNodeName(), "DynamicrnnV2 input shape error.");
    return ge::GRAPH_FAILED;
  }
  const gert::StorageShape* xShape = context->GetInputShape(0);
  OP_CHECK_IF(xShape == nullptr,
                  OP_LOGE(context->GetNodeName(), "RNNV2 input_x is null."),
                  return ge::GRAPH_FAILED);

  const gert::Shape& xStorageShape = optiling::EnsureNotScalar(xShape->GetStorageShape());
  OP_CHECK_IF(xStorageShape.GetDimNum() < DST_SHAPE_SIZE,
                  OP_LOGE(context->GetNodeName(), "DynamicrnnV2 xStorageShape error."),
                  return ge::GRAPH_FAILED);

  auto compileInfo = context->GetCompileInfo<DynamicRNNV2CompileInfo>();
  OP_CHECK_IF(compileInfo == nullptr,
                  OP_LOGE(context->GetNodeName(), "DynamicrnnV2 compileInfo is null."),
                  return ge::GRAPH_FAILED);

  auto runParams = context->GetTilingData<DynamicRNNV2TilingData>();
  OP_CHECK_IF(runParams == nullptr,
                  OP_LOGE(context->GetNodeName(), "runParams is null."),
                  return ge::GRAPH_FAILED);

  runParams->sequenceLength = xStorageShape.GetDim(0);
  runParams->dynamicrnnBatch = xStorageShape.GetDim(DEFAULT_INDEX_TWO);
  runParams->chequeIndex = GetRnnV2LibItem(context, compileInfo, xStorageShape);
  OP_CHECK_IF(runParams->chequeIndex == DEFAULT_RETURN,
                  OP_LOGE(context->GetNodeName(), "DynamicrnnV2 get index fail."),
                  return ge::GRAPH_FAILED);
  OP_CHECK_IF(context->SetTilingKey(runParams->chequeIndex) != ge::GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "DynamicrnnV2 set tiling fail."),
                  return ge::GRAPH_FAILED);

  context->SetBlockDim(MAX_BLOCK_DIM);
  size_t* workspaceSize = context->GetWorkspaceSizes(WORKSPACE_SIZES.size());
  OPS_CHECK_NULL_WITH_CONTEXT(context, workspaceSize);
  for (size_t i = 0; i < WORKSPACE_SIZES.size(); i++) {
    workspaceSize[i] = WORKSPACE_SIZES[i];
  }

  OP_LOGD(context->GetNodeName(), "DynamicRNNV2Tiling end.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4DynamicRNNV2(gert::TilingParseContext* context) {
  OP_LOGD(context->GetNodeName(), "DynamicRNNV2Tiling TilingPrepare4DynamicRNNV2 start.");
  // 910B/C AscendC
  bool supportL0c2out = false;
  DynamicRNNV2Tiling rnnv2Tiling;
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
  rnnv2Tiling.GetAICoreIntrinsicDtype(*platformInfo, "Intrinsic_fix_pipe_l0c2out", supportL0c2out);
  if (supportL0c2out) {
    return ge::GRAPH_SUCCESS;
  }

  auto compileInfo = context->GetCompiledInfo<DynamicRNNV2CompileInfo>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

  std::unique_ptr<nlohmann::json> parsedObjectCinfo = GetCompileInfoJson(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsedObjectCinfo);
  const nlohmann::json& vars = (*parsedObjectCinfo)["vars"];
  OP_CHECK_IF(vars.empty(),
                  OP_LOGE(context->GetNodeName(),
                                                  "Failed to get vars in parsed_object_cinfo."),
                  return ge::GRAPH_FAILED);

  OP_CHECK_IF(!ReadCompileItem(vars, "tune_shape_list", compileInfo->tuneShapeList),
                  OP_LOGE(context->GetNodeName(),
                                                  "DynamicRNNV2, get tune_shape_list failed."),
                  return ge::GRAPH_FAILED);

  OP_CHECK_IF(compileInfo->tuneShapeList.empty(),
                  OP_LOGE(context->GetNodeName(),
                                                  "DynamicRNNV2, tune_shape_list is empty."),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicRNNV2).Tiling(Tiling4DynamicRNNV2).TilingParse<DynamicRNNV2CompileInfo>(TilingPrepare4DynamicRNNV2);
}  // namespace optiling
