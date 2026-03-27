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
 * \file sync_batch_norm_backward_elemt_tiling_arch35.cpp
 * \brief
 */
#include "sync_batch_norm_backward_elemt_tiling_arch35.h"
#include "norm/sync_batch_norm_backward_elemt/op_kernel/arch35/sync_batch_norm_backward_elemt_dag.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;

namespace optiling {
constexpr uint64_t SYNC_BATCH_NORM_BACKWARD_ELEMT_TILING_KEY_ELEMENTWISE = 0UL;
constexpr uint32_t GRAD_OUTPUT_INDEX = 0;
constexpr uint32_t SAVE_INPUT_INDEX = 1;
constexpr uint32_t MEAN_INDEX = 2;
constexpr uint32_t INVSTD_INDEX = 3;
constexpr uint32_t WEIGHT_INDEX = 4;
constexpr uint32_t MEAN_DY_INDEX = 5;
constexpr uint32_t MEAN_DY_XMU_INDEX = 6;
constexpr uint32_t GRAD_INPUT_INDEX = 0;

ge::graphStatus SyncBatchNormBackwardElemtTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        currentWorkspace == nullptr,
        OP_LOGE(tilingContext, "GetWorkspaceSizes failed"),
        return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfoPtr = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    tilingContext->SetTilingKey(SYNC_BATCH_NORM_BACKWARD_ELEMT_TILING_KEY_ELEMENTWISE);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormBackwardElemtTiling::CalcInputDtype()
{
    auto gradOutputDesc = tilingContext->GetInputDesc(GRAD_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, gradOutputDesc);
    this->gradOutputDtype = gradOutputDesc->GetDataType();
    OP_CHECK_IF(
        this->gradOutputDtype != ge::DT_FLOAT16 && this->gradOutputDtype != ge::DT_BF16 &&
            this->gradOutputDtype != ge::DT_FLOAT,
        OP_LOGE(
            tilingContext->GetNodeName(), "input grad dtype[%s] not support",
            ge::TypeUtils::DataTypeToSerialString(this->gradOutputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto saveInputDesc = tilingContext->GetInputDesc(SAVE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, saveInputDesc);
    this->saveInputDtype = saveInputDesc->GetDataType();

    auto meanDesc = tilingContext->GetInputDesc(MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, meanDesc);
    this->meanDtype = meanDesc->GetDataType();

    auto invstdDesc = tilingContext->GetInputDesc(INVSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, invstdDesc);
    this->invstdDtype = invstdDesc->GetDataType();

    auto weightDesc = tilingContext->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, weightDesc);
    this->weightDtype = weightDesc->GetDataType();

    auto meanDyDesc = tilingContext->GetInputDesc(MEAN_DY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, meanDyDesc);
    this->meanDyDtype = meanDyDesc->GetDataType();

    auto meanDyXmuDesc = tilingContext->GetInputDesc(MEAN_DY_XMU_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, meanDyXmuDesc);
    this->meanDyXmuDtype = meanDyXmuDesc->GetDataType();

    if (this->gradOutputDtype == ge::DT_FLOAT16 && this->meanDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(
            this->gradOutputDtype != this->saveInputDtype || this->meanDtype != this->invstdDtype ||
                this->meanDtype != this->weightDtype || this->meanDtype != this->meanDyDtype ||
                this->meanDtype != this->meanDyXmuDtype,
            OP_LOGE(tilingContext, "when need transform dtype, input dtype is diff, check failed"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            this->gradOutputDtype != this->saveInputDtype || this->gradOutputDtype != this->meanDtype ||
                this->gradOutputDtype != this->invstdDtype || this->gradOutputDtype != this->weightDtype ||
                this->gradOutputDtype != this->meanDyDtype || this->gradOutputDtype != this->meanDyXmuDtype,
            OP_LOGE(tilingContext, "input dtype is diff, check failed"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormBackwardElemtTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(GRAD_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(
        this->gradOutputDtype != this->outputDtype,
        OP_LOGE(tilingContext, "input dtype[%s] and output dtype[%s] is diff, check failed", 
            ge::TypeUtils::DataTypeToSerialString(this->gradOutputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormBackwardElemtTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check input dtype failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check output dtype failed"),
        return ge::GRAPH_FAILED);

    tiling = tilingContext->GetTilingData<SyncBatchNormBackwardElemtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    ge::graphStatus res = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        if (this->meanDtype == DT_FLOAT) {
            res = elewiseBaseTiling.DoTiling<SyncBatchNormBackwardElemtDag<half, float>::OpDag>(
                tiling->baseTiling);
        } else {
            res = elewiseBaseTiling.DoTiling<SyncBatchNormBackwardElemtDag<half, half>::OpDag>(tiling->baseTiling);
        }
    } else if (this->outputDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<SyncBatchNormBackwardElemtDag<float, float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<SyncBatchNormBackwardElemtDag<bfloat16_t, bfloat16_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext, "Data type check failed. Input grad dtype[%s] not support",
            ge::TypeUtils::DataTypeToSerialString(this->gradOutputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "DoTiling failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus result = SetTilingData();
    return result;
}

static ge::graphStatus TilingForSyncBatchNormBackwardElemt(gert::TilingContext* context)
{
    OP_LOGD("SyncBatchNormBackwardElemt", "Enter TilingForSyncBatchNormBackwardElemt");
    OP_CHECK_IF(
        context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    SyncBatchNormBackwardElemtTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForSyncBatchNormBackwardElemt([[maybe_unused]] gert::TilingParseContext *context){
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SyncBatchNormBackwardElemt).
    Tiling(TilingForSyncBatchNormBackwardElemt).
    TilingParse<ElewiseCompileInfo>(TilingPrepareForSyncBatchNormBackwardElemt);
} // namespace optiling