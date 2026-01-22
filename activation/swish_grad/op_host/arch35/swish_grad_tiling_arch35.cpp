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
 * \file swish_grad_regbase_optiling.cc
 * \brief
 */
#include "swish_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "activation/swish_grad/op_kernel/arch35/swish_grad_dag.h"
#include "activation/swish_grad/op_kernel/arch35/swish_grad_struct.h"

#include <iostream>

namespace optiling {
const size_t ASCEND_WORKSPACE = 16777216; // 16M

ge::graphStatus SwishGradTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishGradTiling SetTilingData enter.");

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY((uint64_t)tiling->baseTiling.scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishGradTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(
            tilingContext->GetNodeName(), "input grad dtype[%s] not support",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto inputDesc1 = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc1);
    this->inputDtype1 = inputDesc1->GetDataType();
    OP_CHECK_IF(
        this->inputDtype1 != this->inputDtype,
        OP_LOGE(
            tilingContext->GetNodeName(), "input x dtype[%s] not same as input grad[%s]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype1).c_str(),
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto inputDesc2 = tilingContext->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc2);
    this->inputDtype2 = inputDesc2->GetDataType();
    OP_CHECK_IF(
        this->inputDtype2 != this->inputDtype,
        OP_LOGE(
            tilingContext->GetNodeName(), "input y dtype[%s] not same as input grad[%s]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype2).c_str(),
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishGradTiling::CheckShape()
{
    auto gradStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradStorageShape);
    const gert::Shape& inputGradShape = Ops::Base::EnsureNotScalar(gradStorageShape->GetStorageShape());
    auto xStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(xStorageShape->GetStorageShape());
    auto yStorageShape = tilingContext->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(yStorageShape->GetStorageShape());

    auto gradXStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradXStorageShape);
    const gert::Shape& outputGradXShape = Ops::Base::EnsureNotScalar(gradXStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputGradShape != inputXShape, OP_LOGE(tilingContext->GetNodeName(), "input grad and x shape not same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputGradShape != inputYShape, OP_LOGE(tilingContext->GetNodeName(), "input grad and y shape not same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputGradShape != outputGradXShape,
        OP_LOGE(tilingContext->GetNodeName(), "input grad and output grad_x shape not same"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishGradTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE(
            tilingContext->GetNodeName(), "output grad_x dtype[%s] not same as input y[%s]",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishGradTiling::SetAttr()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishGradTiling SetAttr enter.");
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
    float scaleValue = scaleValueAttr == nullptr ? 1.0f : *scaleValueAttr;
    tiling->scale = scaleValue;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwishGradTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "SwishGradTiling RunTiling enter.");

    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);
    
    // get tilingdata address in context
    tiling = tilingContext->GetTilingData<SwishGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    OP_CHECK_IF(SetAttr() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "set Attr failed"), return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<SwishGradDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<SwishGradDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<SwishGradDAG<float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(
            tilingContext->GetNodeName(), "output dtype[%s] not support",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4SwishGrad(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4SwishGrad rt2.0 is running.");

    SwishGradTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForSwishGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwishGrad).Tiling(Tiling4SwishGrad).TilingParse<ElewiseCompileInfo>(TilingPrepareForSwishGrad);
} // namespace optiling