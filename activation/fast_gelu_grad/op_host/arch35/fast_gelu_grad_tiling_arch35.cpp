/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "activation/fast_gelu_grad/op_kernel/arch35/fast_gelu_grad_dag.h"
#include "activation/fast_gelu_grad/op_kernel/arch35/fast_gelu_grad_struct.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "fast_gelu_grad_tiling_arch35.h"

using namespace FastGeluGradOp;

namespace optiling {
const int64_t SYSWORKSPACE = 16777216; // 16 * 1024 * 1024

ge::graphStatus FastGeluGradTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "self dtype not support"), return ge::GRAPH_FAILED);

    auto inputDesc1 = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc1);
    this->inputDtype1 = inputDesc1->GetDataType();
    OP_CHECK_IF(
        this->inputDtype1 != ge::DT_FLOAT16 && this->inputDtype1 != ge::DT_BF16 && this->inputDtype1 != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "input dy dtype not support"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        this->inputDtype1 != this->inputDtype, OP_LOGE(tilingContext->GetNodeName(), "input dy dtype not support"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FastGeluGradTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE(tilingContext->GetNodeName(), "self dtype not support"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype, OP_LOGE(tilingContext->GetNodeName(), "out dtype not same as self"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FastGeluGradTiling::CheckShape()
{
    auto selfStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, selfStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(selfStorageShape->GetStorageShape());

    auto dyStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dyStorageShape);
    const gert::Shape& inputDyShape = Ops::Base::EnsureNotScalar(dyStorageShape->GetStorageShape());

    auto outStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outStorageShape);
    const gert::Shape& outputShape = Ops::Base::EnsureNotScalar(outStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != inputDyShape, OP_LOGE(tilingContext->GetNodeName(), "input x and input dy shape not same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputXShape != outputShape, OP_LOGE(tilingContext->GetNodeName(), "input x and output z shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

std::string FastGeluGradTiling::DataTypeToSerialString(const ge::DataType type) const
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("Neg", "datatype %d not find", type);
        return "UNDEFINED";
    }
}

ge::graphStatus FastGeluGradTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    // 获取tiling计算所需的参数
    ge::graphStatus status = ge::GRAPH_FAILED;
    status = CalcInputDtype();
    OP_CHECK_IF(
        status == ge::GRAPH_FAILED, OP_CHECK_NULL_WITH_CONTEXT(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    status = CalcOutputDtype();
    OP_CHECK_IF(
        status == ge::GRAPH_FAILED, OP_CHECK_NULL_WITH_CONTEXT(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    status = CheckShape();
    OP_CHECK_IF(
        status == ge::GRAPH_FAILED, OP_CHECK_NULL_WITH_CONTEXT(tilingContext, "check shape failed"),
        return ge::GRAPH_FAILED);
    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get FastGeluGradTiling from GE context failed"),
        return ge::GRAPH_FAILED);
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        status = elewiseBaseTiling.DoTiling<FastGeluGradDag::FastGeluGradNeedCast<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        status = elewiseBaseTiling.DoTiling<FastGeluGradDag::FastGeluGradNeedCast<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        status = elewiseBaseTiling.DoTiling<FastGeluGradDag::FastGeluGradNoCast<float>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling DoTiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        status == ge::GRAPH_FAILED, OP_CHECK_NULL_WITH_CONTEXT(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%ld.", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);
    size_t usr_workspaceSize = 0;
    size_t sys_workspaceSize = SYSWORKSPACE;
    size_t* current_workspace = tilingContext->GetWorkspaceSizes(1);
    current_workspace[0] = sys_workspaceSize + usr_workspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncFastGeluGrad(gert::TilingContext* tilingContext)
{
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);

    OP_LOGD(tilingContext->GetNodeName(), "START FastGeluGrad AscendC Tiling \n");
    FastGeluGradTiling FastGeluGradTiling(tilingContext);
    return FastGeluGradTiling.RunTiling();
}

ge::graphStatus TilingPrepareForFastGeluGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FastGeluGrad)
    .Tiling(TilingFuncFastGeluGrad)
    .TilingParse<ElewiseCompileInfo>(TilingPrepareForFastGeluGrad);
} // namespace optiling