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
 * \file relu_v2_tiling_arch35.cpp
 * \brief
 */

#include "relu_v2_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "../op_kernel/arch35/relu_v2_dag.h"
#include "../op_kernel/arch35/relu_v2_tiling_struct.h"
#include "common/inc/tiling_base/tiling_util.h"

#include <iostream>

using namespace ge;
using namespace ReluV2Op;
using namespace ReluV2Ns;

namespace optiling {
constexpr uint64_t SYS_WORKSPACE = 16777216; // 16M
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_FP16 = 101;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_BF16 = 102;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_FP32 = 103;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT8 = 104;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_INT32 = 105;
constexpr uint64_t RELU_TILING_KEY_ELEMENTWISE_UINT8 = 106;
const gert::Shape g_vec_1_shape = {1};

class ReluV2Tiling {
public:
    explicit ReluV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    ReluV2TilingData* tiling;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext;
    ge::DataType inputDtype;
    ge::DataType outputDtype;
};

ge::graphStatus ReluV2Tiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter SetTilingData");
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORKSPACE;

    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluV2Tiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);

    this->inputDtype = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus ReluV2Tiling::CheckShape()
{
    auto gradientsStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradientsStorageShape);
    const gert::Shape& inputGradientsShape = EnsureNotScalar(gradientsStorageShape->GetStorageShape());

    auto backpropsStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, backpropsStorageShape);
    const gert::Shape& outputShape = EnsureNotScalar(backpropsStorageShape->GetStorageShape());
    auto maskStorageShape = tilingContext->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, maskStorageShape);
    const gert::Shape& outputMaskShape = EnsureNotScalar(maskStorageShape->GetStorageShape());

    auto dimNum = inputGradientsShape.GetDimNum();

    OP_CHECK_IF(
        (dimNum < 1 || inputGradientsShape.GetDim(dimNum - 1) % 8 != 0),
        OP_LOGE(tilingContext->GetNodeName(), "The last dimension must be divisible by 8."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputGradientsShape != outputMaskShape,
        OP_LOGE(tilingContext->GetNodeName(), "Input gradients and mask shape not same"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputGradientsShape != outputShape,
        OP_LOGE(tilingContext->GetNodeName(), "Input gradients and output backprops shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluV2Tiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(
        this->inputDtype != this->outputDtype, OP_LOGE(tilingContext, "Input and output dtype is diff, check failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluV2Tiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ReluV2Tiling RunTiling Enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Check shape failed"), return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<ReluV2TilingData>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext, "Get EleBaseTilingData from context failed"),
        return ge::GRAPH_FAILED);
    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<ReluV2DAG<half, half>::OpDag>(tiling->baseTiling);
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_FP16);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<ReluV2DAG<bfloat16_t, float>::OpDag>(tiling->baseTiling);
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_BF16);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<ReluV2DAG<float, float>::OpDag>(tiling->baseTiling);
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_FP32);
    } else if (this->outputDtype == ge::DT_INT8) {
        res = elewiseBaseTiling.DoTiling<ReluV2DAG<int8_t, half>::OpDag>(tiling->baseTiling);
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT8);
    } else if (this->outputDtype == ge::DT_INT32) {
        res = elewiseBaseTiling.DoTiling<ReluV2DAG<int32_t, int32_t>::OpDag>(tiling->baseTiling);
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_INT32);
    } else if (this->outputDtype == ge::DT_UINT8) {
        res = elewiseBaseTiling.DoTiling<ReluV2DAG<uint8_t, half>::OpDag>(tiling->baseTiling);
        tilingContext->SetTilingKey(RELU_TILING_KEY_ELEMENTWISE_UINT8);
    } else {
        OP_LOGE(tilingContext, "Data type check failed. Ge type：%d", this->outputDtype);
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "DoTiling failed"), return ge::GRAPH_FAILED);
    return SetTilingData();
}

static ge::graphStatus Tiling4ReluV2(gert::TilingContext* context)
{
    OP_LOGD("ReluV2Tiling", "Enter Tiling4ReluV2");
    if (context == nullptr) {
        OP_LOGE("ReluV2Tiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const ReluV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的模板tiling
    OP_LOGD("ReluV2Tiling", "Enter new ReluV2Tiling");
    ReluV2Tiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForReluV2(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ReluV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluV2).Tiling(Tiling4ReluV2).TilingParse<ReluV2CompileInfo>(TilingPrepareForReluV2);
} // namespace optiling