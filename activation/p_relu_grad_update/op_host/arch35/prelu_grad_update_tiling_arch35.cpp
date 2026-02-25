/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prelu_grad_update_tiling_arch35.cpp
 * \brief
 */
#include "prelu_grad_update_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "activation/p_relu_grad_update/op_kernel/arch35/prelu_grad_update_struct.h"
#include "activation/p_relu_grad_update/op_kernel/arch35/prelu_grad_update_dag.h"
#include "error_util.h"

#include <iostream>

using namespace Ops::Base;
namespace optiling {

static const size_t INPUT_FEATURE_INDEX = 1;
static const size_t INPUT_WEIGHT_INDEX = 2;
static const size_t INPUT_GRADIENT_INDEX = 0;
static const size_t OUT_BACKPROPS_DX_INDEX = 0;
static const size_t OUT_BACKPROPS_DA_INDEX = 1;

constexpr static uint64_t PRELU_GRAD_UPDATE_COMMON_TILING_PRIORITY = 0;

using namespace ge;

bool PReluGradUpdateTiling::IsCapable()
{
    return true;
}

ge::graphStatus PReluGradUpdateTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PReluGradUpdateTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t PReluGradUpdateTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus PReluGradUpdateTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PReluGradUpdateTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PReluGradUpdateTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PReluGradUpdateTiling::CalcInputDtype()
{
    OP_LOGD("CalcInputDtype", "PReluGradUpdateTiling CalcInputDtype enter.");

    auto inputFeturesDesc = context_->GetInputDesc(INPUT_FEATURE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputFeturesDesc);
    this->inputFeturesDtype = inputFeturesDesc->GetDataType();
    OPS_ERR_IF(
        this->inputFeturesDtype != ge::DT_FLOAT16 && this->inputFeturesDtype != ge::DT_BF16 &&
            this->inputFeturesDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input features dtype allow {float16, bfloat16, float32} ,but not support %s",ge::TypeUtils::DataTypeToSerialString(this->inputFeturesDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto inputWeightsDesc = context_->GetInputDesc(INPUT_WEIGHT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputWeightsDesc);
    this->inputWeightsDtype = inputWeightsDesc->GetDataType();
    OPS_ERR_IF(
        this->inputWeightsDtype != ge::DT_FLOAT16 && this->inputWeightsDtype != ge::DT_BF16 &&
            this->inputWeightsDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input weights dtype {float16, bfloat16, float32} ,but not support %s",ge::TypeUtils::DataTypeToSerialString(this->inputWeightsDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto inputGradientsDesc = context_->GetInputDesc(INPUT_GRADIENT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGradientsDesc);
    this->inputGradientsDtype = inputGradientsDesc->GetDataType();
    OPS_ERR_IF(
        this->inputGradientsDtype != ge::DT_FLOAT16 && this->inputGradientsDtype != ge::DT_BF16 &&
            this->inputGradientsDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input gradients dtype {float16, bfloat16, float32} ,but not support %s",ge::TypeUtils::DataTypeToSerialString(this->inputGradientsDtype).c_str()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        this->inputGradientsDtype != this->inputWeightsDtype || this->inputGradientsDtype != this->inputFeturesDtype,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input gradients features weights dtype are not the same"),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PReluGradUpdateTiling::CheckAndInferShape(std::vector<gert::Shape>& inputShapes)
{
    auto inputStorageFeatureShape = context_->GetInputShape(INPUT_FEATURE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputStorageFeatureShape);
    const gert::Shape& inputFeatureShape = Ops::Base::EnsureNotScalar(inputStorageFeatureShape->GetStorageShape());
    size_t featureDimNum = inputFeatureShape.GetDimNum();

    auto inputStorageWeightShape = context_->GetInputShape(INPUT_WEIGHT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputStorageWeightShape);
    const gert::Shape& inputWeightShape = Ops::Base::EnsureNotScalar(inputStorageWeightShape->GetStorageShape());
    size_t weightDimNum = inputWeightShape.GetDimNum();

    auto inputStorageGradientShape = context_->GetInputShape(INPUT_GRADIENT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputStorageGradientShape);
    const gert::Shape& inputGradientShape = Ops::Base::EnsureNotScalar(inputStorageGradientShape->GetStorageShape());

    OPS_ERR_IF(featureDimNum <= 1,
       VECTOR_INNER_ERR_REPORT_TILIING(
                context_->GetNodeName(), "feature don't support 1D or 0D shape"),
            return ge::GRAPH_FAILED);
    
    //检查weight的元素个数是否等于1或者features的通道数
    int64_t weightSize = inputWeightShape.GetShapeSize();
    OP_TILING_CHECK(weightSize != 1 && weightSize != inputFeatureShape.GetDim(1),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "weight shape size must be 1 or features dim1 size"),
                    return ge::GRAPH_FAILED);

    inputShapes.push_back(inputGradientShape);
    inputShapes.push_back(inputFeatureShape);
    gert::Shape weightReshape;
    if (weightDimNum == featureDimNum - 1 && weightDimNum != 1) {
        weightReshape.AppendDim(1);
        for (uint32_t i = 0; i < weightDimNum; i++) {
            weightReshape.AppendDim(inputWeightShape.GetDim(i));
        }
    } else {
        for (uint32_t i = 0; i < featureDimNum; i++) {
            if (i == 1) {
                weightReshape.AppendDim(inputWeightShape.GetDim(0));
                continue;
            }
            weightReshape.AppendDim(1);
        }
    }
    inputShapes.push_back(weightReshape);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PReluGradUpdateTiling::CalcOutputDtype()
{
    OP_LOGD("CalcOutputDtype", "PReluGradUpdateTiling CalcOutputDtype enter.");

    auto outputDxDesc = context_->GetOutputDesc(OUT_BACKPROPS_DX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDxDesc);
    this->outputBackpropsDxDtype = outputDxDesc->GetDataType();
    OPS_ERR_IF(
        this->outputBackpropsDxDtype != this->inputFeturesDtype,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "output backprops dx dtype not same as input features"),
        return ge::GRAPH_FAILED);

    auto outputDaDesc = context_->GetOutputDesc(OUT_BACKPROPS_DA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDaDesc);
    this->outputBackpropsDaDtype = outputDaDesc->GetDataType();
    OPS_ERR_IF(
        this->outputBackpropsDaDtype != this->inputFeturesDtype,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "output backprops da dtype not same as input features"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PReluGradUpdateTiling::DoOpTiling()
{
    OPS_ERR_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OPS_REPORT_VECTOR_INNER_ERR(context_, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OPS_REPORT_VECTOR_INNER_ERR(context_, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    std::vector<gert::Shape> inputShapes;
    OPS_ERR_IF(
        CheckAndInferShape(inputShapes) == ge::GRAPH_FAILED, OPS_REPORT_VECTOR_INNER_ERR(context_, "check and infer shape failed"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context_->GetInputDesc(INPUT_FEATURE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (inputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<PReluGradUpdate::PReluGradUpdateDAG<half>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
        OP_LOGD("DoOpTiling","DT_FLOAT16 GetSchMode: %lu",brcBaseTiling.GetSchMode());
        OP_LOGD("DoOpTiling","DT_FLOAT16 tiling key: %lu",tilingKey);
    } else if (inputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<PReluGradUpdate::PReluGradUpdateDAG<bfloat16_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
        OP_LOGD("DoOpTiling","DT_BF16 GetSchMode: %lu",brcBaseTiling.GetSchMode());
        OP_LOGD("DoOpTiling","DT_BF16 tiling key: %lu",tilingKey);
    } else if (inputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<PReluGradUpdate::PReluGradUpdateDAG<float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetOpInputStorageShapes(inputShapes);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
        OP_LOGD("DoOpTiling","DT_FLOAT GetSchMode: %lu",brcBaseTiling.GetSchMode());
        OP_LOGD("DoOpTiling","DT_FLOAT tiling key: %lu",tilingKey);
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "Input dtype only support fp16, bf16, fp32 currently is %s.",
            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(
        status != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context_, "BroadcastBaseTiling do tiling failed."), return ge::GRAPH_FAILED);
    OP_LOGD("prelugradupdate tiling","end do tiling");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4PReluGradUpdate(gert::TilingContext* context)
{
    OP_LOGD("PReluGradUpdateTiling", "Enter Tiling4PReluGradUpdate");
    if (context == nullptr) {
        OP_LOGE("PReluGradUpdateTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("PReluGradUpdateTiling", "Enter new PReluGradUpdateTiling");
    PReluGradUpdateTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBroadcast(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PReluGradUpdate).Tiling(Tiling4PReluGradUpdate).TilingParse<BroadcastCompileInfo>(TilingPrepareForBroadcast);
REGISTER_OPS_TILING_TEMPLATE(PReluGradUpdate, PReluGradUpdateTiling, PRELU_GRAD_UPDATE_COMMON_TILING_PRIORITY);
} // namespace optiling