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
 * \file elu_grad_tiling_arch35.cpp
 * \brief
 */
#include "elu_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log/log.h"
#include <iostream>
#include "../op_kernel/arch35/elu_grad_dag.h"
#include "../op_kernel/arch35/elu_grad_struct.h"

using namespace ge;
using namespace EluGradOp;

namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; //16M

const gert::Shape g_vec_1_shape = {1};
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus EluGradTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradTiling CalcInputDtype enter.");
    auto gradsDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradsDesc);
    auto activationsDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, activationsDesc);
    this->gradsDtype = gradsDesc->GetDataType();
    OP_CHECK_IF(
        this->gradsDtype != ge::DT_FLOAT16 && this->gradsDtype != ge::DT_BF16 && this->gradsDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "Gradoutput dtype not support"),
        return ge::GRAPH_FAILED);
    this->activationsDtype = activationsDesc->GetDataType();
    OP_CHECK_IF(
        this->activationsDtype != this->gradsDtype,
        OP_LOGE(tilingContext->GetNodeName(), "Activations dtype not equal gradoutput"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluGradTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "Gradinput dtype not support"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->gradsDtype,
        OP_LOGE(tilingContext->GetNodeName(), "Gradinput dtype not the same as gradoutput"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EluGradTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradTiling CheckShape enter."); 
    auto gradsStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradsStorageShape);
    const gert::Shape& gradsShape = EnsureNotScalar(gradsStorageShape->GetStorageShape());
    
    auto activationsStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, activationsStorageShape);
    const gert::Shape& activationsShape = EnsureNotScalar(activationsStorageShape->GetStorageShape());

    auto outStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outStorageShape);
    const gert::Shape& outputShape = EnsureNotScalar(outStorageShape->GetStorageShape());

    OP_CHECK_IF(gradsShape != outputShape && gradsShape != activationsShape,
               OP_LOGE(tilingContext->GetNodeName(), "Input grads, activations and output y shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus EluGradTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "EluGradTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    // 获取tiling计算所需的参数
    ge::graphStatus status = ge::GRAPH_FAILED;
    status = CalcInputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get input dtype failed"),
               return ge::GRAPH_FAILED);
    status = CalcOutputDtype();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Get output dtype failed"),
               return ge::GRAPH_FAILED);
    status = CheckShape();
    OP_CHECK_IF(status == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "Check shape failed"),
               return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF((tiling == nullptr),
                    OP_LOGE(tilingContext->GetNodeName(), "Get EluGradTiling from GE context failed"),
                    return ge::GRAPH_FAILED);

    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = static_cast<uint64_t>(TPL_FP16);
        status = elewiseBaseTiling.DoTiling<EluGradDag<half>::OpDag>(*tiling);
    } else if(this->outputDtype == ge::DT_BF16) {
        dType = static_cast<uint64_t>(TPL_BF16);
        status = elewiseBaseTiling.DoTiling<EluGradDag<bfloat16_t>::OpDag>(*tiling);
    } else if(this->outputDtype == ge::DT_FLOAT) {
        dType = static_cast<uint64_t>(TPL_FP32);
        status = elewiseBaseTiling.DoTiling<EluGradDag<float>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "ElewiseBaseTiling DoTiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(status == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "ElewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%ld.", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);
    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = ASCEND_WORKSPACE;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4EluGrad(gert::TilingContext *tilingContextSelf)
{
    OP_LOGD(tilingContextSelf->GetNodeName(), "Tiling4EluGrad rt2.0 is running.");
    auto compileInfo = static_cast<const EluGradCompileInfo*>(tilingContextSelf->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextSelf, compileInfo);
    OP_LOGD("EluGradTiling", "Enter new EluTiling");
    EluGradTiling eluGradTiling(tilingContextSelf);
    return eluGradTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForEluGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<EluGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(EluGrad)
    .Tiling(Tiling4EluGrad)
    .TilingParse<EluGradCompileInfo>(TilingPrepareForEluGrad);
}  // namespace optiling
 