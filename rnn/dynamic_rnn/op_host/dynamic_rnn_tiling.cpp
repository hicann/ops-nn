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
#include <string>

using namespace AscendC;
namespace {
bool AddWorkspace(gert::TilingContext* context, const size_t workspace)
{
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    *workspace_size = workspace;
    return true;
}

// 合法输入下 tiling 内部失败统一上报正式码 EZ1008，携带 opName 与 reason。
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace
namespace optiling {
static int32_t GetRnnLibItem(const DynamicRnnCompileInfo* compileInfo, const gert::Shape xStorageShape)
{
    for (const auto& tuneShape : compileInfo->tuneShapeList) {
        if (tuneShape.size() < DEFAULT_SHAPE_LIST_SIZE) {
            ReportTilingFailure("DynamicRnn",
                                "tune_shape_list's size is illegal. it's " + std::to_string(tuneShape.size()) + ".");
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

ge::graphStatus DynamicRNNTiling::GetOpInfo(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams)
{
    OP_CHECK_IF(context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input"), return ge::GRAPH_FAILED);

    // get x shape
    auto xTensor = context->GetInputShape(0);
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input x"),
                return ge::GRAPH_FAILED);
    auto xShape = xTensor->GetStorageShape();

    // get w shape
    auto weightTensor = context->GetInputShape(1);
    OP_CHECK_IF(weightTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input weight"),
                return ge::GRAPH_FAILED);
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
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input x",
                                                         std::to_string(xShape.GetDimNum()) + "D",
                                                         "the shape dim of input x must be 3"),
                return false);
    OP_CHECK_IF(xShape.GetDim(CONST_TWO) == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input x",
                                                      std::to_string(xShape.GetDim(CONST_TWO)),
                                                      "the input_size of input x must not be 0"),
                return false);
    OP_CHECK_IF(weightShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input weight",
                                                         std::to_string(weightShape.GetDimNum()) + "D",
                                                         "the shape dim of input weight must be 2"),
                return false);
    dynamicRnnParams.timeStep = static_cast<int64_t>(xShape.GetDim(0));
    dynamicRnnParams.batch = static_cast<int64_t>(xShape.GetDim(1));
    dynamicRnnParams.inputSize = static_cast<int64_t>(xShape.GetDim(dim));
    dynamicRnnParams.hiddenSize = static_cast<int64_t>(weightShape.GetDim(0)) - dynamicRnnParams.inputSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicRNNTiling::GetAttr(const gert::TilingContext* context, DynamicRnnTiling& dynamicRnnParams)
{
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

bool DynamicRNNTiling::CheckParamsShape(gert::TilingContext* context)
{
    // get input shape
    auto xInput = context->GetInputShape(0);
    OP_CHECK_IF(xInput == nullptr, ReportNullPtrWithContext(context, "xInput"), return false);
    auto xShape = xInput->GetStorageShape();
    // get wight shape
    auto wInput = context->GetInputShape(1);
    OP_CHECK_IF(wInput == nullptr, ReportNullPtrWithContext(context, "wInput"), return false);
    auto wShape = wInput->GetStorageShape();

    // get bias shape
    auto bInput = context->GetInputShape(2);
    OP_CHECK_IF(bInput == nullptr, ReportNullPtrWithContext(context, "bInput"), return false);
    auto bShape = bInput->GetStorageShape();

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
    OP_CHECK_IF(wShape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input w",
                                                         std::to_string(wShape.GetDimNum()) + "D",
                                                         "the shape dim of input w must be 2"),
                return false);
    OP_CHECK_IF(bShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input b",
                                                         std::to_string(bShape.GetDimNum()) + "D",
                                                         "the shape dim of input b must be 1"),
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
    OP_CHECK_IF(wShape.GetDim(0) != xShape.GetDim(2) + outputShape.GetDim(2),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "input w, input x, output y",
                    Shape2String(wShape) + ", " + Shape2String(xShape) + " and " + Shape2String(outputShape),
                    "the 1st dim of input w must equal the last dim of input x plus "
                    "the last dim of output y"),
                return false);

    // check hidden dim
    OP_CHECK_IF(
        wShape.GetDim(1) != 4 * outputShape.GetDim(2),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input w, output y",
                                               Shape2String(wShape) + " and " + Shape2String(outputShape),
                                               "the 2nd dim of input w must equal 4 times the last dim of output y"),
        return false);
    OP_CHECK_IF(
        bShape.GetDim(0) != 4 * outputShape.GetDim(2),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input b, output y",
                                               Shape2String(bShape) + " and " + Shape2String(outputShape),
                                               "the 1st dim of input b must equal 4 times the last dim of output y"),
        return false);
    bool ret = CheckInitParamsShape(context);

    return ret;
}

bool DynamicRNNTiling::CheckInitParamsShape(gert::TilingContext* context)
{
    auto seqInput = context->GetOptionalInputShape(3);
    auto inithInput = context->GetOptionalInputShape(4);
    auto initcInput = context->GetOptionalInputShape(5);

    // get output y shape
    auto outputY = context->GetOutputShape(0);
    OP_CHECK_IF(outputY == nullptr, ReportNullPtrWithContext(context, "outputY"), return false);
    auto outputShape = outputY->GetStorageShape();

    OP_CHECK_IF(outputShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "output y",
                                                         std::to_string(outputShape.GetDimNum()) + "D",
                                                         "the shape dim of output y must be 3"),
                return false);

    if (inithInput != nullptr && initcInput != nullptr) {
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
        OP_CHECK_IF(
            inithShape.GetDim(1) != outputShape.GetDim(1),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_h, output y",
                                                   Shape2String(inithShape) + " and " + Shape2String(outputShape),
                                                   "the batch dim of input init_h must equal that of output y"),
            return false);
        OP_CHECK_IF(
            initcShape.GetDim(1) != outputShape.GetDim(1),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_c, output y",
                                                   Shape2String(initcShape) + " and " + Shape2String(outputShape),
                                                   "the batch dim of input init_c must equal that of output y"),
            return false);
        OP_CHECK_IF(
            inithShape.GetDim(2) != outputShape.GetDim(2),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_h, output y",
                                                   Shape2String(inithShape) + " and " + Shape2String(outputShape),
                                                   "the hidden dim of input init_h must equal that of output y"),
            return false);
        OP_CHECK_IF(
            initcShape.GetDim(2) != outputShape.GetDim(2),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input init_c, output y",
                                                   Shape2String(initcShape) + " and " + Shape2String(outputShape),
                                                   "the hidden dim of input init_c must equal that of output y"),
            return false);
    }

    if (seqInput != nullptr) {
        auto seqShape = seqInput->GetStorageShape();
        OP_CHECK_IF(seqShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input seq_length",
                                                             std::to_string(seqShape.GetDimNum()) + "D",
                                                             "the shape dim of input seq_length must be 3"),
                    return false);
        OP_CHECK_IF(
            seqShape.GetDim(0) != outputShape.GetDim(0),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input seq_length, output y",
                                                   Shape2String(seqShape) + " and " + Shape2String(outputShape),
                                                   "the time step of input seq_length must equal that of output y"),
            return false);
        OP_CHECK_IF(
            seqShape.GetDim(1) != outputShape.GetDim(1),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input seq_length, output y",
                                                   Shape2String(seqShape) + " and " + Shape2String(outputShape),
                                                   "the batch dim of input seq_length must equal that of output y"),
            return false);
        OP_CHECK_IF(
            seqShape.GetDim(2) != outputShape.GetDim(2),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input seq_length, output y",
                                                   Shape2String(seqShape) + " and " + Shape2String(outputShape),
                                                   "the hidden dim of input seq_length must equal that of output y"),
            return false);
    }

    return true;
}

bool DynamicRNNTiling::CheckAttrTiling(gert::TilingContext* context)
{
    // get attr
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportNullPtrWithContext(context, "attrs"), return false);
    const int* numProj = attrs->GetAttrPointer<int>(6);
    OP_CHECK_IF(numProj == nullptr, ReportNullPtrWithContext(context, "numProj"), return false);
    const bool* timeMajor = attrs->GetAttrPointer<bool>(7);
    OP_CHECK_IF(timeMajor == nullptr, ReportNullPtrWithContext(context, "timeMajor"), return false);
    const char* activation = attrs->GetAttrPointer<char>(8);
    OP_CHECK_IF(activation == nullptr, ReportNullPtrWithContext(context, "activation"), return false);
    const float* forgetBias = attrs->GetAttrPointer<float>(9);
    OP_CHECK_IF(forgetBias == nullptr, ReportNullPtrWithContext(context, "forgetBias"), return false);
    const char* gateOrder = attrs->GetAttrPointer<char>(10);
    OP_CHECK_IF(gateOrder == nullptr, ReportNullPtrWithContext(context, "gateOrder"), return false);
    const bool* isTraining = attrs->GetAttrPointer<bool>(11);
    OP_CHECK_IF(isTraining == nullptr, ReportNullPtrWithContext(context, "isTraining"), return false);

    OP_CHECK_IF(*numProj != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr num_proj", std::to_string(*numProj),
                                                      "num_proj only supports 0"),
                return false);
    OP_CHECK_IF(!(*timeMajor),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr time_major",
                                                      std::to_string(*timeMajor), "time_major only supports true"),
                return false);

    std::vector<std::string> supportGateOrder = {"ifjo", "ijfo"};
    OP_CHECK_IF(std::find(supportGateOrder.begin(), supportGateOrder.end(), gateOrder) == supportGateOrder.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr gate_order", gateOrder,
                                                      "gate_order only supports ijfo or ifjo"),
                return false);

    OP_CHECK_IF(strcmp(activation, "tanh") != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr activation", activation,
                                                      "activation only supports tanh"),
                return false);

    return true;
}

bool DynamicRNNTiling::CheckAttrOps(gert::TilingContext* context)
{
    // get attr
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportNullPtrWithContext(context, "attrs"), return false);
    const char* cellType = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(cellType == nullptr, ReportNullPtrWithContext(context, "cellType"), return false);
    const char* direction = attrs->GetAttrPointer<char>(1);
    OP_CHECK_IF(direction == nullptr, ReportNullPtrWithContext(context, "direction"), return false);
    const int* cellDepth = attrs->GetAttrPointer<int>(2);
    OP_CHECK_IF(cellDepth == nullptr, ReportNullPtrWithContext(context, "cellDepth"), return false);
    const bool* usePeephole = attrs->GetAttrPointer<bool>(3);
    OP_CHECK_IF(usePeephole == nullptr, ReportNullPtrWithContext(context, "usePeephole"), return false);
    const float* keepProb = attrs->GetAttrPointer<float>(4);
    OP_CHECK_IF(keepProb == nullptr, ReportNullPtrWithContext(context, "keepProb"), return false);
    const float* cellClip = attrs->GetAttrPointer<float>(5);
    OP_CHECK_IF(cellClip == nullptr, ReportNullPtrWithContext(context, "cellClip"), return false);

    OP_CHECK_IF(*cellDepth != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr cell_depth",
                                                      std::to_string(*cellDepth), "cell_depth only supports 1"),
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

ge::graphStatus TilingForDynamicRNN(gert::TilingContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);
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
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                           "tiling with ascendc failed; propagate the lower-layer first error."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input"), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input x"),
                return ge::GRAPH_FAILED);

    const auto& xStorageShape = optiling::EnsureNotScalar(xShape->GetStorageShape());
    OP_CHECK_IF(xStorageShape.GetDimNum() < DEFAULT_XSHAPE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input x",
                                                         std::to_string(xStorageShape.GetDimNum()) + "D",
                                                         "the shape dim of input x must be at least 3"),
                return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompileInfo<DynamicRnnCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "compileInfo is null."),
                return ge::GRAPH_FAILED);

    auto runParams = context->GetTilingData<DynamicRnnTilingData>();
    OP_CHECK_IF(runParams == nullptr, OP_LOGE(context->GetNodeName(), "runParams is null."), return ge::GRAPH_FAILED);

    runParams->sequenceLength = static_cast<int32_t>(xStorageShape.GetDim(0));
    runParams->dynamicRnnBatch = static_cast<int32_t>(xStorageShape.GetDim(DEFAULT_INDEX_TWO));
    runParams->chequeIndex = GetRnnLibItem(compileInfo, xStorageShape);
    OP_CHECK_IF(runParams->chequeIndex == DEFAULT_RETURN,
                ReportTilingFailure(context->GetNodeName(), "no matched schedule for the given input shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetTilingKey(runParams->chequeIndex) != ge::GRAPH_SUCCESS,
                ReportTilingFailure(context->GetNodeName(), "failed to set tiling key"), return ge::GRAPH_FAILED);

    context->SetBlockDim(BLOCK_DIM);
    AddWorkspace(context, WORKSPACE_SIZE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDynamicRNN(gert::TilingParseContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);
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
    OP_CHECK_IF(vars.empty(), ReportTilingFailure(context->GetNodeName(), "failed to get vars in parsedObjectCinfo."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!ReadCompileItem(vars, "tune_shape_list", compileInfo->tuneShapeList),
                ReportTilingFailure(context->GetNodeName(), "failed to get tune_shape_list from compile info."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(compileInfo->tuneShapeList.empty(),
                ReportTilingFailure(context->GetNodeName(), "tune_shape_list is empty in compile info."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicRNN).Tiling(TilingForDynamicRNN).TilingParse<DynamicRnnCompileInfo>(TilingPrepareForDynamicRNN);
} // namespace optiling
