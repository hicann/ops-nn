
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
 * \file relu_grad_v2_tiling_arch35.cpp
 * \brief
 */
#include "relu_grad_v2_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "../op_kernel/arch35/relu_grad_v2_dag.h"
#include "../op_kernel/arch35/relu_grad_v2_tiling_struct.h"

#include <iostream>

using namespace ge;
using namespace ReluGradV2Ns;

namespace optiling {
const gert::Shape g_vec_1_shape = {1};

inline static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

constexpr uint64_t ASCEND_WORKSPACE = 16777216;

ge::graphStatus ReluGradV2Tiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ReluGradV2Tiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_INT8 &&
            this->inputDtype != ge::DT_UINT8 && this->inputDtype != ge::DT_INT32 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "input x dtype not support"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluGradV2Tiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ReluGradV2Tiling CheckShape enter.");
    auto gradientsStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradientsStorageShape);
    const gert::Shape& inputGradientsShape = EnsureNotScalar(gradientsStorageShape->GetStorageShape());

    auto maskStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, maskStorageShape);
    const gert::Shape& inputMaskShape = EnsureNotScalar(maskStorageShape->GetStorageShape());

    auto backpropsStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, backpropsStorageShape);
    const gert::Shape& outputShape = EnsureNotScalar(backpropsStorageShape->GetStorageShape());

    auto dimNum = inputGradientsShape.GetDimNum();

    OP_CHECK_IF(
        (dimNum < 1 || inputGradientsShape.GetDim(dimNum - 1) % 8 != 0),
        OP_LOGE(tilingContext->GetNodeName(), "The last dimension must be divisible by 8."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputGradientsShape != inputMaskShape,
        OP_LOGE(tilingContext->GetNodeName(), "Input gradients and mask shape not same."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputMaskShape != outputShape,
        OP_LOGE(tilingContext->GetNodeName(), "Input mask and output backprops shape not same."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluGradV2Tiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ReluGradV2Tiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE(tilingContext->GetNodeName(), "Output backprops dtype not same as input gradients."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluGradV2Tiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ReluGradV2Tiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);

    OP_CHECK_IF(
        CalcInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "Get input dtype failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "Get output dtype failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "Check shape failed."), return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get ReluGradV2Tiling from GE context failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<ReluGradV2<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<ReluGradV2<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<ReluGradV2<float>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        dType = TPL_INT8;
        baseTilingResult = elewiseBaseTiling.DoTiling<ReluGradV2<int8_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_UINT8) {
        dType = TPL_UINT8;
        baseTilingResult = elewiseBaseTiling.DoTiling<ReluGradV2<uint8_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        dType = TPL_INT32;
        baseTilingResult = elewiseBaseTiling.DoTiling<ReluGradV2<int32_t>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "Output dtype not support.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        baseTilingResult != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "ElewiseBaseTiling failed."),
        return ge::GRAPH_FAILED);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tiling_Key=%ld", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForReluGradV2(gert::TilingContext* tilingContextGen)
{
    OP_LOGD("ReluGradV2", "Enter TilingForReluGradV2");
    if (tilingContextGen == nullptr) {
        OP_LOGE("ReluGradV2", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const ReluGradV2CompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);

    OP_LOGD("ReluGradV2", "Enter new ReluGradV2Tiling");
    ReluGradV2Tiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForReluGradV2(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ReluGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluGradV2).Tiling(TilingForReluGradV2).TilingParse<ReluGradV2CompileInfo>(TilingPrepareForReluGradV2);
} // namespace optiling