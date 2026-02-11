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
 * \file log_sigmoid_tiling_arch35.cpp
 * \brief
 */

#include <iostream>

#include "platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

#include "log_sigmoid_tiling_arch35.h"

#include "../op_kernel/arch35/log_sigmoid_dag.h"
#include "../op_kernel/arch35/log_sigmoid_struct.h"
#include "common/inc/tiling_base/tiling_util.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace ge;
using namespace LogSigmoidOp;

namespace optiling {
const size_t LOG_SIGMOID_SYS_WORKSPACE = 16777216; // 16M

ge::graphStatus LogSigmoidTiling::CheckDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);

    auto inputDtype = inputDesc->GetDataType();
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(
        inputDtype != this->outputDtype, OP_LOGE(tilingContext->GetNodeName(), "dtype of input and output are not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogSigmoidTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputStorageShape->GetStorageShape());
    const gert::Shape& outputShape = Ops::NN::OpTiling::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputShape != outputShape, OP_LOGE(tilingContext->GetNodeName(), "shape of input and output are not same"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogSigmoidTiling::RunTiling()
{
    OP_CHECK_IF(
        tilingContext == nullptr, OP_LOGE("CheckContextValid", "tilingContext is nullptr!"),
        return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext->GetNodeName(), "LogSigmoidTiling RunTiling enter.");

    // check type and shape
    ElewiseBaseTiling eleBaseTiling(tilingContext);
    OP_CHECK_IF(
        CheckDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(
        (tiling == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get LogSigmoidTiling from GE context failed"),
        return ge::GRAPH_FAILED);

    // do tiling
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = eleBaseTiling.DoTiling<LogSigmoidDag::LogSigmoidNeedCast<half>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = eleBaseTiling.DoTiling<LogSigmoidDag::LogSigmoidNeedCast<bfloat16_t>::OpDag>(*tiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = eleBaseTiling.DoTiling<LogSigmoidDag::LogSigmoidNoCast<float>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype is not support.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    // set workspace, tilingkey and blocknum
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = LOG_SIGMOID_SYS_WORKSPACE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->scheMode), dType);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->blockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4LogSigmoid(gert::TilingContext* tilingContext)
{
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContext->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    
    OP_LOGD(tilingContext->GetNodeName(), "START LogSigmoid AscendC Tiling \n");
    LogSigmoidTiling LogSigmoidTiling(tilingContext);
    return LogSigmoidTiling.RunTiling();
}

ge::graphStatus TilingPrepareForLogSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogSigmoid).Tiling(Tiling4LogSigmoid)
                            .TilingParse<ElewiseCompileInfo>(TilingPrepareForLogSigmoid);
} // namespace optiling