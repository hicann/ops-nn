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

bool AddWorkspace(gert::TilingContext* context, const size_t workspace)
{
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    *workspace_size = workspace;
    return true;
}

// 合法输入下，tiling 内部失败统一上报正式码 EZ1008（Execution_Error），携带 opName 与 reason。
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"op_name", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace

namespace optiling {
static int32_t GetRnnV2LibItem(const gert::TilingContext* context, const DynamicRNNV2CompileInfo* compileInfo,
                               const gert::Shape xShape)
{
    for (const auto& tuneShape : compileInfo->tuneShapeList) {
        OP_CHECK_IF(tuneShape.size() < DST_SHAPE_SIZE,
                    ReportTilingFailure("DynamicRNNV2", "tune_shape_list's size is illegal. it's " +
                                                            std::to_string(tuneShape.size()) + "."),
                    return DEFAULT_RETURN);

        if ((tuneShape[0] == -1) && (tuneShape[1] == -1)) {
            OP_LOGI(context->GetNodeName(),
                    "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, The corresponding schedule is %lu",
                    tuneShape[DEFAULT_INDEX_TWO]);
            return static_cast<int32_t>(tuneShape[DEFAULT_INDEX_TWO]);
        }

        if ((tuneShape[0] == xShape.GetDim(0)) && ((tuneShape[1] / NUM_SIXTEEN) == xShape.GetDim(DEFAULT_INDEX_TWO))) {
            OP_LOGI(context->GetNodeName(),
                    "op [DynamicRnnV2Tiling] : GetRnnV2LibItem, The corresponding schedule is %lu",
                    tuneShape[DEFAULT_INDEX_TWO]);
            return static_cast<int32_t>(tuneShape[DEFAULT_INDEX_TWO]);
        }
    }
    return DEFAULT_RETURN;
}

bool DynamicRNNV2Tiling::CheckInitParamsShape(gert::TilingContext* context)
{
    auto bInput = context->GetOptionalInputShape(3);
    auto seqInput = context->GetOptionalInputShape(4);
    auto inithInput = context->GetOptionalInputShape(5);
    auto initcInput = context->GetOptionalInputShape(6);
    auto outputShape = context->GetOutputShape(0)->GetStorageShape();

    if (bInput != nullptr) {
        auto bShape = bInput->GetStorageShape();
        OP_CHECK_IF(bShape.GetDimNum() != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input b",
                                                             std::to_string(bShape.GetDimNum()) + "D",
                                                             "the shape dim of input b must be 1"),
                    return false);

        OP_CHECK_IF(
            bShape.GetDim(0) != 4 * outputShape.GetDim(2),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "input b, output y", Shape2String(bShape) + " and " + Shape2String(outputShape),
                "the 1st dim of input b must equal 4 times the last dim of output y"),
            return false);
    }

    OP_CHECK_IF(seqInput != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input seq_length",
                                                         "seq_length is not supported by DynamicRNNV2"),
                return false);

    OPS_CHECK_NULL_WITH_CONTEXT(context, inithInput);
    OPS_CHECK_NULL_WITH_CONTEXT(context, initcInput);
    auto inithShape = inithInput->GetStorageShape();
    auto initcShape = initcInput->GetStorageShape();

    OP_CHECK_IF(inithShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input init_h",
                                                         std::to_string(inithShape.GetDimNum()) + "D",
                                                         "the shape dim of input init_h must be 3"),
                return false);

    OP_CHECK_IF(initcShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input init_c",
                                                         std::to_string(initcShape.GetDimNum()) + "D",
                                                         "the shape dim of input init_c must be 3"),
                return false);

    OP_CHECK_IF(inithShape.GetDim(1) != outputShape.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_h, output y",
                                                       Shape2String(inithShape) + " and " + Shape2String(outputShape),
                                                       "the batch dim of input init_h must equal that of output y"),
                return false);

    OP_CHECK_IF(initcShape.GetDim(1) != outputShape.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_c, output y",
                                                       Shape2String(initcShape) + " and " + Shape2String(outputShape),
                                                       "the batch dim of input init_c must equal that of output y"),
                return false);

    OP_CHECK_IF(inithShape.GetDim(2) != outputShape.GetDim(2),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_h, output y",
                                                       Shape2String(inithShape) + " and " + Shape2String(outputShape),
                                                       "the hidden dim of input init_h must equal that of output y"),
                return false);

    OP_CHECK_IF(initcShape.GetDim(2) != outputShape.GetDim(2),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_c, output y",
                                                       Shape2String(initcShape) + " and " + Shape2String(outputShape),
                                                       "the hidden dim of input init_c must equal that of output y"),
                return false);

    return true;
}

bool DynamicRNNV2Tiling::CheckAttrOps(gert::TilingContext* context)
{
    // get attr
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportNullPtrWithContext(context, "attrs"), return false);
    const float* cellClip = attrs->GetAttrPointer<float>(5);
    OP_CHECK_IF(cellClip == nullptr, ReportNullPtrWithContext(context, "cellClip"), return false);
    const float* keepProb = attrs->GetAttrPointer<float>(4);
    OP_CHECK_IF(keepProb == nullptr, ReportNullPtrWithContext(context, "keepProb"), return false);
    const bool* usePeephole = attrs->GetAttrPointer<bool>(3);
    OP_CHECK_IF(usePeephole == nullptr, ReportNullPtrWithContext(context, "usePeephole"), return false);
    const int* cellDepth = attrs->GetAttrPointer<int>(2);
    OP_CHECK_IF(cellDepth == nullptr, ReportNullPtrWithContext(context, "cellDepth"), return false);
    const char* direction = attrs->GetAttrPointer<char>(1);
    OP_CHECK_IF(direction == nullptr, ReportNullPtrWithContext(context, "direction"), return false);
    const char* cellType = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(cellType == nullptr, ReportNullPtrWithContext(context, "cellType"), return false);

    OP_CHECK_IF(*cellDepth != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr cell_depth",
                                                      std::to_string(*cellDepth), "cell_depth only supports 1"),
                return false);

    OP_CHECK_IF(std::abs(*keepProb - 1) >= std::numeric_limits<float>::epsilon(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr keep_prob",
                                                      std::to_string(*keepProb), "keep_prob only supports 1.0"),
                return false);

    OP_CHECK_IF(std::abs(*cellClip + 1) >= std::numeric_limits<float>::epsilon(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr cell_clip",
                                                      std::to_string(*cellClip), "cell_clip only supports -1.0"),
                return false);

    OP_CHECK_IF(*usePeephole,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr use_peephole",
                                                      std::to_string(*usePeephole), "use_peephole only supports false"),
                return false);

    std::vector<std::string> supportDirection = {"UNIDIRECTIONAL", "REDIRECTIONAL"};
    OP_CHECK_IF(std::find(supportDirection.begin(), supportDirection.end(), direction) == supportDirection.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr direction", direction,
                                                      "direction only supports UNIDIRECTIONAL or REDIRECTIONAL"),
                return false);

    OP_CHECK_IF(strcmp(cellType, "LSTM") != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr cell_type", cellType,
                                                      "cell_type only supports LSTM"),
                return false);
    return true;
}

bool DynamicRNNV2Tiling::CheckAttrTiling(gert::TilingContext* context)
{
    // get attr
    auto attrs = context->GetAttrs();
    const int* numProj = attrs->GetAttrPointer<int>(6);
    OP_CHECK_IF(numProj == nullptr, ReportNullPtrWithContext(context, "numProj"), return false);
    const bool* timeMajor = attrs->GetAttrPointer<bool>(7);
    OP_CHECK_IF(timeMajor == nullptr, ReportNullPtrWithContext(context, "timeMajor"), return false);
    const char* activation = attrs->GetAttrPointer<char>(8);
    OP_CHECK_IF(activation == nullptr, ReportNullPtrWithContext(context, "activation"), return false);
    const char* recurrentActivation = attrs->GetAttrPointer<char>(9);
    OP_CHECK_IF(recurrentActivation == nullptr, ReportNullPtrWithContext(context, "recurrentActivation"), return false);
    const float* forgetBias = attrs->GetAttrPointer<float>(10);
    OP_CHECK_IF(forgetBias == nullptr, ReportNullPtrWithContext(context, "forgetBias"), return false);
    const char* gateOrder = attrs->GetAttrPointer<char>(11);
    OP_CHECK_IF(gateOrder == nullptr, ReportNullPtrWithContext(context, "gateOrder"), return false);
    const bool* stateful = attrs->GetAttrPointer<bool>(12);
    OP_CHECK_IF(stateful == nullptr, ReportNullPtrWithContext(context, "stateful"), return false);
    const char* mergeMode = attrs->GetAttrPointer<char>(13);
    OP_CHECK_IF(mergeMode == nullptr, ReportNullPtrWithContext(context, "mergeMode"), return false);
    const bool* isTraining = attrs->GetAttrPointer<bool>(14);
    OP_CHECK_IF(isTraining == nullptr, ReportNullPtrWithContext(context, "isTraining"), return false);

    OP_CHECK_IF(*numProj != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr num_proj", std::to_string(*numProj),
                                                      "num_proj only supports 0"),
                return false);
    OP_CHECK_IF(!(*timeMajor),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr time_major",
                                                      std::to_string(*timeMajor), "time_major only supports true"),
                return false);

    OP_CHECK_IF(*stateful,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr stateful",
                                                      std::to_string(*stateful), "stateful only supports false"),
                return false);

    std::vector<std::string> supportGateOrder = {"ifco", "ijfo"};
    OP_CHECK_IF(std::find(supportGateOrder.begin(), supportGateOrder.end(), gateOrder) == supportGateOrder.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr gate_order", gateOrder,
                                                      "gate_order only supports ijfo or ifco"),
                return false);

    OP_CHECK_IF(strcmp(activation, "tanh") != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr activation", activation,
                                                      "activation only supports tanh"),
                return false);

    OP_CHECK_IF(
        strcmp(recurrentActivation, "sigmoid") != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr recurrent_activation", recurrentActivation,
                                              "recurrent_activation only supports sigmoid"),
        return false);

    std::vector<std::string> supportMergeMode = {"concat", "add"};
    OP_CHECK_IF(std::find(supportMergeMode.begin(), supportMergeMode.end(), mergeMode) == supportMergeMode.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr merge_mode", mergeMode,
                                                      "merge_mode only supports concat or add"),
                return false);
    return true;
}

ge::graphStatus DynamicRNNV2Tiling::GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams)
{
    OP_CHECK_IF(
        context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input num",
                                                 "the input count of DynamicRNNV2 is less than the minimum required"),
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
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input x",
                                                         std::to_string(xShape.GetDimNum()) + "D",
                                                         "the shape dim of input x must be 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightHiddenShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input weight_hidden",
                                                         std::to_string(weightHiddenShape.GetDimNum()) + "D",
                                                         "the shape dim of input weight_hidden must be 2"),
                return ge::GRAPH_FAILED);
    dynamicRnnParams.timeStep = static_cast<int64_t>(xShape.GetDim(0));
    dynamicRnnParams.batch = static_cast<int64_t>(xShape.GetDim(1));
    dynamicRnnParams.inputSize = static_cast<int64_t>(xShape.GetDim(dim));
    dynamicRnnParams.hiddenSize = static_cast<int64_t>(weightHiddenShape.GetDim(0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicRNNV2Tiling::GetAttr(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams)
{
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

bool DynamicRNNV2Tiling::CheckParamsShape(gert::TilingContext* context)
{
    // get input shape
    auto xInput = context->GetInputShape(0);
    OP_CHECK_IF(xInput == nullptr, ReportNullPtrWithContext(context, "xInput"), return false);
    auto xShape = xInput->GetStorageShape();
    // get weight input shape
    auto wInput = context->GetInputShape(1);
    OP_CHECK_IF(wInput == nullptr, ReportNullPtrWithContext(context, "wInput"), return false);
    auto wInputShape = wInput->GetStorageShape();

    // get weight hidden shape
    auto wHidden = context->GetInputShape(2);
    OP_CHECK_IF(wHidden == nullptr, ReportNullPtrWithContext(context, "wHidden"), return false);
    auto wHiddenShape = wHidden->GetStorageShape();

    // get output y shape
    auto outputY = context->GetOutputShape(0);
    OP_CHECK_IF(outputY == nullptr, ReportNullPtrWithContext(context, "outputY"), return false);
    auto outputShape = outputY->GetStorageShape();

    // check dim num
    OP_CHECK_IF(xShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input x",
                                                         std::to_string(xShape.GetDimNum()) + "D",
                                                         "the shape dim of input x must be 3"),
                return false);
    OP_CHECK_IF(xShape.GetDim(CONST_TWO) == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input x",
                                                      std::to_string(xShape.GetDim(CONST_TWO)),
                                                      "the input_size of input x must not be 0"),
                return false);
    OP_CHECK_IF(wInputShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input weight_input",
                                                         std::to_string(wInputShape.GetDimNum()) + "D",
                                                         "the shape dim of input weight_input must be 2"),
                return false);

    OP_CHECK_IF(wHiddenShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input weight_hidden",
                                                         std::to_string(wHiddenShape.GetDimNum()) + "D",
                                                         "the shape dim of input weight_hidden must be 2"),
                return false);
    OP_CHECK_IF(outputShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "output y",
                                                         std::to_string(outputShape.GetDimNum()) + "D",
                                                         "the shape dim of output y must be 3"),
                return false);

    // check batch dim
    OP_CHECK_IF(xShape.GetDim(1) != outputShape.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input x, output y",
                                                       Shape2String(xShape) + " and " + Shape2String(outputShape),
                                                       "the batch dim of input x must equal that of output y"),
                return false);

    // check x/w input_size dim
    OP_CHECK_IF(wInputShape.GetDim(0) != xShape.GetDim(2),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "input weight_input, input x",
                    Shape2String(wInputShape) + " and " + Shape2String(xShape),
                    "the 1st dim of input weight_input must equal the input_size of input x"),
                return false);

    // check hidden dim
    OP_CHECK_IF(wHiddenShape.GetDim(1) != 4 * outputShape.GetDim(2),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "input weight_hidden, output y",
                    Shape2String(wHiddenShape) + " and " + Shape2String(outputShape),
                    "the 2nd dim of input weight_hidden must equal 4 times the last dim of output y"),
                return false);
    bool ret = CheckInitParamsShape(context);

    return ret;
}

ge::graphStatus Tiling4DynamicRNNV2(gert::TilingContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);
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
                    ReportTilingFailure("DynamicRNNV2", "TilingWithAscendC failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    if (context->GetComputeNodeInputNum() < DST_INPUT_SIZE) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input num",
                                                 "the input count of DynamicRNNV2 is less than the minimum required");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "RNNV2 input_x is null."), return ge::GRAPH_FAILED);

    const gert::Shape& xStorageShape = optiling::EnsureNotScalar(xShape->GetStorageShape());
    OP_CHECK_IF(xStorageShape.GetDimNum() < DST_SHAPE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input x",
                                                         std::to_string(xStorageShape.GetDimNum()) + "D",
                                                         "the shape dim of input x must be at least 3"),
                return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompileInfo<DynamicRNNV2CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "DynamicrnnV2 compileInfo is null."),
                return ge::GRAPH_FAILED);

    auto runParams = context->GetTilingData<DynamicRNNV2TilingData>();
    OP_CHECK_IF(runParams == nullptr, OP_LOGE(context->GetNodeName(), "runParams is null."), return ge::GRAPH_FAILED);

    runParams->sequenceLength = xStorageShape.GetDim(0);
    runParams->dynamicrnnBatch = xStorageShape.GetDim(DEFAULT_INDEX_TWO);
    runParams->chequeIndex = GetRnnV2LibItem(context, compileInfo, xStorageShape);
    OP_CHECK_IF(runParams->chequeIndex == DEFAULT_RETURN,
                ReportTilingFailure("DynamicRNNV2", "failed to get the tiling index from tune_shape_list."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetTilingKey(runParams->chequeIndex) != ge::GRAPH_SUCCESS,
                ReportTilingFailure("DynamicRNNV2", "failed to set the tiling key."), return ge::GRAPH_FAILED);

    context->SetBlockDim(MAX_BLOCK_DIM);
    size_t* workspaceSize = context->GetWorkspaceSizes(WORKSPACE_SIZES.size());
    OPS_CHECK_NULL_WITH_CONTEXT(context, workspaceSize);
    for (size_t i = 0; i < WORKSPACE_SIZES.size(); i++) {
        workspaceSize[i] = WORKSPACE_SIZES[i];
    }

    OP_LOGD(context->GetNodeName(), "DynamicRNNV2Tiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4DynamicRNNV2(gert::TilingParseContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);
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
    OP_CHECK_IF(vars.empty(), ReportTilingFailure("DynamicRNNV2", "failed to get vars in parsed_object_cinfo."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!ReadCompileItem(vars, "tune_shape_list", compileInfo->tuneShapeList),
                ReportTilingFailure("DynamicRNNV2", "get tune_shape_list failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(compileInfo->tuneShapeList.empty(), ReportTilingFailure("DynamicRNNV2", "tune_shape_list is empty."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicRNNV2)
    .Tiling(Tiling4DynamicRNNV2)
    .TilingParse<DynamicRNNV2CompileInfo>(TilingPrepare4DynamicRNNV2);
} // namespace optiling
