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
 * \file elu_grad_v2_tiling_arch35.cpp
 * \brief
 */
#include "elu_grad_v2_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/arch35/elu_grad_v2_dag.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "../op_kernel/arch35/elu_grad_v2_struct.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace EluGradV2Op;

namespace optiling {
const uint64_t ALPHA_ATTR_IDX = 0;
const uint64_t SCALE_ATTR_IDX = 1;
const uint64_t INPUT_SCALE_ATTR_IDX = 2;
const uint64_t IS_RESULT_ATTR_IDX = 3;
const int64_t ASCEND_WORKSPACE = 16777216; //16M

const gert::Shape g_vec_1_shape = {1};
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape,
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus EluGradV2Tiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradV2Tiling CalcInputDtype enter.");
    auto gradsDesc = tilingContext->GetInputDesc(0);
    auto activationsDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, activationsDesc);
    this->gradsDtype = gradsDesc->GetDataType();
    this->activationsDtype = activationsDesc->GetDataType();
    OP_CHECK_IF(
        this->gradsDtype != ge::DT_FLOAT16 && this->gradsDtype != ge::DT_BF16 && this->gradsDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "Gradoutput dtype not support"),
        return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        this->activationsDtype != this->gradsDtype,
        OP_LOGE(tilingContext->GetNodeName(), "Activations dtype not support"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluGradV2Tiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradV2Tiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "Gradinput dtype not support"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->gradsDtype,
        OP_LOGE(tilingContext->GetNodeName(), "Gradinput dtype not same as gradoutput"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluGradV2Tiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradV2Tiling CheckShape enter."); 
    auto gradsStorageShapeV2 = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradsStorageShapeV2);
    const gert::Shape& gradsShape = EnsureNotScalar(gradsStorageShapeV2->GetStorageShape());
    
    auto activationsStorageShapeV2 = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, activationsStorageShapeV2);
    const gert::Shape& activationsShape = EnsureNotScalar(activationsStorageShapeV2->GetStorageShape());

    auto outStorageShapeV2 = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outStorageShapeV2);
    const gert::Shape& outputShape = EnsureNotScalar(outStorageShapeV2->GetStorageShape());

    OP_CHECK_IF(gradsShape != outputShape && gradsShape != activationsShape,
               OP_LOGE(tilingContext->GetNodeName(), "Input grads, activations and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluGradV2Tiling::SetTilingData(bool is_result)
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradV2Tiling SetTilingData enter.");
    if (this->outputDtype == ge::DT_FLOAT16 && is_result) {
        dType = static_cast<uint64_t>(EluGradV2_TPL_FP16);
    } else if (this->outputDtype == ge::DT_BF16 && is_result) {
        dType = static_cast<uint64_t>(EluGradV2_TPL_BF16);
    } else if (this->outputDtype == ge::DT_FLOAT && is_result) {
        dType = static_cast<uint64_t>(EluGradV2_TPL_FP32);
    } else if (this->outputDtype == ge::DT_FLOAT16 && !is_result) {
        dType = static_cast<uint64_t>(EluGradV2_TPL_FP16_N);
    }else if (this->outputDtype == ge::DT_BF16 && !is_result) {
        dType = static_cast<uint64_t>(EluGradV2_TPL_BF16_N);
    }else if (this->outputDtype == ge::DT_FLOAT && !is_result) {
        dType = static_cast<uint64_t>(EluGradV2_TPL_FP32_N);
    } else {
        OP_LOGE(tilingContext->GetNodeName(),
                                        "Self dtype is only support fp16、bf16、fp32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluGradV2Tiling::SetAttr()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradV2Tiling SetAttr enter.");
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* alphaValueAttr = attrs->GetAttrPointer<float>(ALPHA_ATTR_IDX);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(SCALE_ATTR_IDX);
    const float* inputScaleValueAttr = attrs->GetAttrPointer<float>(INPUT_SCALE_ATTR_IDX);
    this->isResult = *tilingContext->GetAttrs()->GetAttrPointer<bool>(IS_RESULT_ATTR_IDX);
    float alphaValue = alphaValueAttr == nullptr ? 1.0f : *alphaValueAttr;
    OP_CHECK_IF(this->isResult && alphaValue < 0,
               OP_LOGE(tilingContext->GetNodeName(), "If result is true, alpha must >= 0"),
               return ge::GRAPH_FAILED);
    float scale = scaleValueAttr == nullptr ? 1.0f : *scaleValueAttr;
    float inputScale = inputScaleValueAttr == nullptr ? 1.0f : *inputScaleValueAttr;
    float negcoef = alphaValue * scale;
    tiling->negcoef = negcoef;
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : negcoef=%f.", negcoef);
    tiling->scale = scale;
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : scale=%f.", scale);
    tiling->inputScale = inputScale;
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : inputScale=%f.", inputScale);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluGradV2Tiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradV2Tiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    ge::graphStatus status = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<EluGradV2TilingData>();
    OP_CHECK_IF((tiling == nullptr), OP_LOGE(tilingContext, "Get EleBaseTilingData from context failed"), return ge::GRAPH_FAILED);
    status = CalcInputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get input dtype failed"), return ge::GRAPH_FAILED);
    status = CalcOutputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get output dtype failed"), return ge::GRAPH_FAILED);
    status = CheckShape();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Check shape failed"), return ge::GRAPH_FAILED);
    status = SetAttr();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "SetAttr failed"), return ge::GRAPH_FAILED);
    status = SetTilingData(this->isResult);
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "SetTilingData failed"), return ge::GRAPH_FAILED);
    if (dType == static_cast<uint64_t>(EluGradV2_TPL_FP16)) {
        status = elewiseBaseTiling.DoTiling<EluGradV2IsResultOp<half>::OpDag>(tiling->baseTiling);
    } else if(dType == static_cast<uint64_t>(EluGradV2_TPL_BF16)) {
        status = elewiseBaseTiling.DoTiling<EluGradV2IsResultOp<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if(dType == static_cast<uint64_t>(EluGradV2_TPL_FP32)) {
        status = elewiseBaseTiling.DoTiling<EluGradV2IsResultOp<float>::OpDag>(tiling->baseTiling);
    } else if (dType == static_cast<uint64_t>(EluGradV2_TPL_FP16_N)) {
        status = elewiseBaseTiling.DoTiling<EluGradV2NoResultOp<half>::OpDag>(tiling->baseTiling);
    } else if(dType == static_cast<uint64_t>(EluGradV2_TPL_BF16_N)) {
        status = elewiseBaseTiling.DoTiling<EluGradV2NoResultOp<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if(dType == static_cast<uint64_t>(EluGradV2_TPL_FP32_N)) {
        status = elewiseBaseTiling.DoTiling<EluGradV2NoResultOp<float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling DoTiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);
    schMode = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%ld.", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetRawTilingData());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<uint64_t>(ASCEND_WORKSPACE);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4EluGradV2(gert::TilingContext *tilingContextSelf)
{
    OP_LOGD(tilingContextSelf->GetNodeName(), "Tiling4EluGradV2 rt2.0 is running.");
    auto compileInfo = static_cast<const EluGradV2CompileInfo*>(tilingContextSelf->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextSelf, compileInfo);
    OP_LOGD("EluGradV2Tiling", "Enter new EluGradV2Tiling");
    EluGradV2Tiling eluGradV2Tiling(tilingContextSelf);
    return eluGradV2Tiling.RunTiling();
}

static ge::graphStatus TilingPrepareForEluGradV2(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<EluGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EluGradV2)
    .Tiling(Tiling4EluGradV2)
    .TilingParse<EluGradV2CompileInfo>(TilingPrepareForEluGradV2);
}  // namespace optiling