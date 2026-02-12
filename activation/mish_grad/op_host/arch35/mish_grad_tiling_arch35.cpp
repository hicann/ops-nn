/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mish_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "activation/mish_grad/op_kernel/arch35/mish_grad_dag.h"
#include "activation/mish_grad/op_kernel/arch35/mish_grad_struct.h"
#include <iostream>

using namespace ge;

namespace optiling {
constexpr int64_t ASCEND_WORKSPACE = 0;
constexpr int64_t THIRD = 2; // 第三个输入
const gert::Shape g_vec_1_shape = {1};

inline static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus MishGradTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "MishGradTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(
            tilingContext->GetNodeName(),
            "input grad dtype [%s] not support, only support [DT_BFLOAT16, DT_FLOAT16, DT_FLOAT32]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str()),
        return ge::GRAPH_FAILED);
    auto inputDesc1 = tilingContext->GetInputDesc(1);

    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc1);
    this->inputDtype1 = inputDesc1->GetDataType();
    OP_CHECK_IF(
        this->inputDtype1 != this->inputDtype,
        OP_LOGE(
            tilingContext->GetNodeName(),
            "input x dtype [%s] not support, only support [DT_BFLOAT16, DT_FLOAT16, DT_FLOAT32]",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype1).c_str()),
        return ge::GRAPH_FAILED);
    if (!unfullCompute) {
        auto inputDesc2 = tilingContext->GetInputDesc(THIRD);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc2);
        this->inputDtype2 = inputDesc2->GetDataType();
        OP_CHECK_IF(
            this->inputDtype2 != this->inputDtype,
            OP_LOGE(
                tilingContext->GetNodeName(),
                "input tanh dtype [%s] not support, only support [DT_BFLOAT16, DT_FLOAT16, DT_FLOAT32]",
                ge::TypeUtils::DataTypeToSerialString(this->inputDtype2).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MishGradTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "MishGradTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputYShape != outputZShape,
        OP_LOGE(tilingContext->GetNodeName(), "input x and output y shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MishGradTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "MishGradTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE(
            tilingContext->GetNodeName(),
            "output y dtype [%s] not support, only support [DT_BFLOAT16, DT_FLOAT16, DT_FLOAT32]",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MishGradTiling::RunTiling()
{
    auto tiling = tilingContext->GetTilingData<EleBaseTilingData16B>();
    OP_LOGD(tilingContext->GetNodeName(), "MishGradTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    
    if (tilingContext->GetInputDesc(THIRD) != nullptr) {
        unfullCompute = false;
    }
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16 && unfullCompute) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishGradDAG<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16 && unfullCompute) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishGradDAG<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT && unfullCompute) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishGradDAG<float>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT16 && !unfullCompute) {
        dType = TPL_FP16_FULL;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishGradFullDAG<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16 && !unfullCompute) {
        dType = TPL_BF16_FULL;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishGradFullDAG<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT && !unfullCompute) {
        dType = TPL_FP32_FULL;
        baseTilingResult = elewiseBaseTiling.DoTiling<MishGradFullDAG<float>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "MishGradTiling failed"),
        return ge::GRAPH_FAILED);
    
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
 	currentWorkspace[0] = ASCEND_WORKSPACE;
 	const uint64_t tilingKey = GET_TPL_TILING_KEY(1, dType);
 	OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
 	tilingContext->SetTilingKey(tilingKey);
 	tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
 	return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4MishGrad(gert::TilingContext *tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4MishGrad rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const MishGradCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);

    MishGradTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}


ge::graphStatus TilingPrepareForMishGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<MishGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(MishGrad).Tiling(Tiling4MishGrad).TilingParse<MishGradCompileInfo>(TilingPrepareForMishGrad);

} // namespace optiling