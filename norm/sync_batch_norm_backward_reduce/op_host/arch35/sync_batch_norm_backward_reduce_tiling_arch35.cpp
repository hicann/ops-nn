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
 * \file sync_batch_norm_backward_reduce_tiling_arch35.cc
 * \brief
 */
#include "sync_batch_norm_backward_reduce_tiling_arch35.h"
#include "norm/sync_batch_norm_backward_reduce/op_kernel/arch35/sync_batch_norm_backward_reduce_dag.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include <graph/utils/type_utils.h>
#include <iostream>

using namespace AscendC;
using namespace ge;

namespace optiling {
constexpr uint64_t SYNC_BATCH_NORM_BACKWARD_REDUCE_TILING_KEY_ELEMENTWISE = 0;
constexpr int64_t INPUT_IDX_SUM_DY = 0;
constexpr int64_t INPUT_IDX_SUM_DY_DX_PAD = 1;
constexpr int64_t INPUT_IDX_MEAN = 2;
constexpr int64_t INPUT_IDX_INVERT_STD = 3;

constexpr int64_t OUTPUT_IDX_SUM_DY_XMU = 0;
constexpr int64_t OUTPUT_IDX_Y = 1;

ge::graphStatus SyncBatchNormBackwardReduceTiling::SetTilingData()
{
    fe::PlatFormInfos* platforminfoPtr = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platforminfoPtr);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(platforminfoPtr);
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);

    currentWorkspace[0] = ascendPlatform.GetLibApiWorkSpaceSize();
    tilingContext->SetTilingKey(SYNC_BATCH_NORM_BACKWARD_REDUCE_TILING_KEY_ELEMENTWISE);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormBackwardReduceTiling::CalcOutputDtype()
{
    auto inputSumDyDesc = tilingContext->GetInputDesc(INPUT_IDX_SUM_DY);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputSumDyDesc);
    ge::DataType inputSumDyDtype = inputSumDyDesc->GetDataType();

    auto inputSumDyDxPadDesc = tilingContext->GetInputDesc(INPUT_IDX_SUM_DY_DX_PAD);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputSumDyDxPadDesc);
    ge::DataType inputSumDyDxPadDtype = inputSumDyDxPadDesc->GetDataType();

    auto inputMeanDesc = tilingContext->GetInputDesc(INPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputMeanDesc);
    ge::DataType inputMeanDtype = inputMeanDesc->GetDataType();

    auto inputInvertStdDesc = tilingContext->GetInputDesc(INPUT_IDX_INVERT_STD);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputInvertStdDesc);
    ge::DataType inputInvertStdDtype = inputInvertStdDesc->GetDataType();

    auto outputSumDyXmuDesc = tilingContext->GetOutputDesc(OUTPUT_IDX_SUM_DY_XMU);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputSumDyXmuDesc);
    ge::DataType outputSumDyXmuDtype = outputSumDyXmuDesc->GetDataType();

    auto outputYDesc = tilingContext->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputYDesc);
    ge::DataType outputYDtype = outputYDesc->GetDataType();
    // 检查输入之间的dtype是否相同
    OP_CHECK_IF(
        inputSumDyDtype != inputSumDyDxPadDtype,
        OP_LOGE(tilingContext, "inputSumDy dtype[%s] and inputSumDyDxPad dtype[%s] are diff",
        ge::TypeUtils::DataTypeToSerialString(inputSumDyDtype).c_str(), 
        ge::TypeUtils::DataTypeToSerialString(inputSumDyDxPadDtype).c_str()), 
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputMeanDtype != inputInvertStdDtype,
        OP_LOGE(tilingContext, "inputMean dtype[%s] and inputInvertStd dtype[%s] are diff",
        ge::TypeUtils::DataTypeToSerialString(inputMeanDtype).c_str(), 
        ge::TypeUtils::DataTypeToSerialString(inputInvertStdDtype).c_str()), 
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputSumDyDtype != inputInvertStdDtype,
        OP_LOGE(tilingContext, "inputSumDy dtype[%s] and inputInvertStd dtype[%s] are diff",
        ge::TypeUtils::DataTypeToSerialString(inputSumDyDtype).c_str(), 
        ge::TypeUtils::DataTypeToSerialString(inputInvertStdDtype).c_str()), 
        return ge::GRAPH_FAILED);
    // 检查输出之间的dtype是否相同
    OP_CHECK_IF(
        outputSumDyXmuDtype != outputYDtype,
        OP_LOGE(tilingContext, "outputSumDyXmu dtype[%s] and outputY dtype[%s] are diff",
        ge::TypeUtils::DataTypeToSerialString(outputSumDyXmuDtype).c_str(), 
        ge::TypeUtils::DataTypeToSerialString(outputYDtype).c_str()), 
        return ge::GRAPH_FAILED);
    // 输入输出之间是dtype是否相同
    this->outputDtype = outputSumDyXmuDesc->GetDataType();
    OP_CHECK_IF(
        inputSumDyDtype != this->outputDtype,
        OP_LOGE(tilingContext, "output dtype[%s] and input dtype[%s] are diff",
        ge::TypeUtils::DataTypeToSerialString(outputYDtype).c_str(), 
        ge::TypeUtils::DataTypeToSerialString(inputSumDyDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormBackwardReduceTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<SyncBatchNormBackwardReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<SyncBatchNormBackwardReduceDag<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<SyncBatchNormBackwardReduceDag<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<SyncBatchNormBackwardReduceDag<bfloat16_t>::OpDag>(tiling->baseTiling);
    }

    OP_CHECK_IF(
        res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "input dtype[%s] not support",
        ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForSyncBatchNormBackwardReduce(gert::TilingContext* context)
{
    OP_LOGD("SyncBatchNormBackwardReduceTiling", "Enter TilingForSyncBatchNormBackwardReduce");
    OP_CHECK_IF(
        context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    SyncBatchNormBackwardReduceTiling syncBatchNormBackwardReduceTiling(context);
    return syncBatchNormBackwardReduceTiling.RunTiling();
}

ge::graphStatus TilingPrepareForSyncBatchNormBackwardReduce([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SyncBatchNormBackwardReduce).
    Tiling(TilingForSyncBatchNormBackwardReduce).
    TilingParse<ElewiseCompileInfo>(TilingPrepareForSyncBatchNormBackwardReduce);
} // namespace optiling
