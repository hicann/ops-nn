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
 * \file sigmoid_cross_entropy_with_logits_v2_tiling.cpp
 * \brief
 */
#include "sigmoid_cross_entropy_with_logits_v2_tiling.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/runtime2_util.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

const uint32_t INPUT_PREDICT_INDEX = 0;
const uint32_t INPUT_TARGET_INDEX = 1;
const uint32_t INPUT_WEIGHT_INDEX = 2;
const uint32_t INPUT_POS_WEIGHT_INDEX = 3;
const uint32_t REDUCTION_INDEX = 0;

const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
const map<std::string, uint32_t> REDUCTION_MODE_KEY = {{"none", 0}};
const map<ge::DataType, uint32_t> DTYPE_KEY = {{ge::DT_FLOAT16, 10}, {ge::DT_FLOAT, 20}, {ge::DT_BF16, 30}};

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter SigmoidCEWithLogitsV2TilingClass GetShapeAttrsInfo.");

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS,
                VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Check datatype failed. "),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS,
                VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Check shape failed. "),
                return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(), "End SigmoidCEWithLogitsV2TilingClass GetShapeAttrsInfo.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter SigmoidCEWithLogitsV2TilingClass GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    aicoreParams_.ubSize = ubSizePlatForm;
    aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();

    OP_LOGD(context_->GetNodeName(), "End SigmoidCEWithLogitsV2TilingClass GetPlatformInfo.");

    return ge::GRAPH_SUCCESS;
}

bool SigmoidCEWithLogitsV2TilingClass::IsCapable() { return true; }

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::DoOpTiling()
{
    if (context_ == nullptr) {
        OP_LOGE("SigmoidCrossEntropyWithLogitsV2", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "SigmoidCrossEntropyWithLogitsV2Tiling DoOpTiling enter.");

    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto reduction = attrs->GetAttrPointer<char>(REDUCTION_INDEX);
    auto iter = REDUCTION_MODE_KEY.find(reduction);
    OP_CHECK_IF(iter == REDUCTION_MODE_KEY.end(),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "reduction", reduction, "none"),
                return ge::GRAPH_FAILED);
    // 算子本身仅实现reduction = none的情况，其他情况会在接口层面调用其他算子融合实现
    this->reduction = 0;

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (this->outputDtype == this->predictDtype) {
        if (this->predictDtype == ge::DT_FLOAT16 || this->predictDtype == ge::DT_BF16) {
            ret = RunBroadcastTiling<half, half>();
        } else if (this->predictDtype == ge::DT_FLOAT) {
            ret = RunBroadcastTiling<float, float>();
        } else {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "predict", ToString(this->predictDtype).c_str(),
                                      "FLOAT16, FLOAT or BF16");
            ret = ge::GRAPH_FAILED;
        }
    } else if (this->outputDtype == ge::DT_FLOAT) {
        if (this->predictDtype == ge::DT_FLOAT16 || this->predictDtype == ge::DT_BF16) {
            ret = RunBroadcastTiling<half, float>();
        } else if (this->predictDtype == ge::DT_FLOAT) {
            ret = RunBroadcastTiling<float, float>();
        } else {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "predict", ToString(this->predictDtype).c_str(),
                                      "FLOAT16, FLOAT or BF16");
            ret = ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "loss", ToString(this->outputDtype).c_str(),
                                  "FLOAT16, FLOAT, BF16");
        ret = ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t SigmoidCEWithLogitsV2TilingClass::GetTilingKey() const { return this->tilingKey_; }

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::CheckDtype()
{
    auto predictPtr = context_->GetInputDesc(INPUT_PREDICT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, predictPtr);
    this->predictDtype = predictPtr->GetDataType();
    OP_CHECK_IF(SUPPORT_DTYPE.count(this->predictDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "predict", ToString(this->predictDtype).c_str(),
                                          "FLOAT16, FLOAT or BF16"),
                return ge::GRAPH_FAILED);

    auto targetPtr = context_->GetInputDesc(INPUT_TARGET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, targetPtr);
    auto targetDtype = targetPtr->GetDataType();
    OP_CHECK_IF(SUPPORT_DTYPE.count(targetDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "target", ToString(targetDtype).c_str(),
                                          "FLOAT16, FLOAT or BF16"),
                return ge::GRAPH_FAILED);

    auto outputPtr = context_->GetOutputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputPtr);
    this->outputDtype = outputPtr->GetDataType();
    OP_CHECK_IF(SUPPORT_DTYPE.count(this->outputDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "loss", ToString(this->outputDtype).c_str(),
                                          "FLOAT16, FLOAT or BF16"),
                return ge::GRAPH_FAILED);

    if (this->outputDtype != this->predictDtype && this->outputDtype != ge::DT_FLOAT) {
        std::string reasonMsg = "The dtype of output loss must be FLOAT or the same as the dtype {" +
                                ToString(this->predictDtype) + "} of input predict";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "loss", ToString(this->outputDtype).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCEWithLogitsV2TilingClass::CheckShape()
{
    auto predictPtr = context_->GetInputShape(INPUT_PREDICT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, predictPtr);
    auto predictShape = EnsureNotScalar(predictPtr->GetStorageShape());

    auto targetPtr = context_->GetInputShape(INPUT_TARGET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, targetPtr);
    auto targetShape = EnsureNotScalar(targetPtr->GetStorageShape());

    if (predictShape != targetShape) {
        std::string shapeMsg = ToString(predictShape) + " and " + ToString(targetShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "predict and target", shapeMsg.c_str(),
                                               "The shapes of input predict and input target must be the same");
        return ge::GRAPH_FAILED;
    }

    auto weightPtr = context_->GetOptionalInputShape(INPUT_WEIGHT_INDEX);
    if (weightPtr != nullptr) {
        this->hasWeight = 1;
    }

    auto posWeightPtr = context_->GetOptionalInputShape(INPUT_POS_WEIGHT_INDEX);
    if (posWeightPtr != nullptr) {
        this->hasPosWeight = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4SigmoidCrossEntropyWithLogitsV2(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                VECTOR_INNER_ERR_REPORT_TILIING("SigmoidCrossEntropyWithLogitsV2", "context should not be nullptr."),
                return ge::GRAPH_FAILED);

    SigmoidCEWithLogitsV2TilingClass tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingParse4SigmoidCrossEntropyWithLogitsV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to get compile info for SigmoidCrossEntropyWithLogitsV2.");

    auto compile_info = GetCompileInfoPtr<SigmoidCEWithLogitsV2CompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SigmoidCrossEntropyWithLogitsV2)
    .Tiling(Tiling4SigmoidCrossEntropyWithLogitsV2)
    .TilingParse<SigmoidCEWithLogitsV2CompileInfo>(TilingParse4SigmoidCrossEntropyWithLogitsV2);
} // namespace optiling