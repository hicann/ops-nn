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
    OP_CHECK_IF(this->gradOutputDtype != ge::DT_FLOAT16 && this->gradOutputDtype != ge::DT_BF16 && this->gradOutputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "grad_output",
            ge::TypeUtils::DataTypeToSerialString(gradOutputDtype).c_str(), "float16, bfloat16 and float"), return ge::GRAPH_FAILED);

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

    // Validate dtype consistency
    auto AllSame = [](ge::DataType ref, std::initializer_list<ge::DataType> dtypes) {
        return std::all_of(dtypes.begin(), dtypes.end(), [ref](ge::DataType d) { return d == ref; });
    };
    bool isFloat16FloatCase = (this->gradOutputDtype == ge::DT_FLOAT16 && this->meanDtype == ge::DT_FLOAT);
    bool valid = isFloat16FloatCase ? (this->gradOutputDtype == this->saveInputDtype &&
            AllSame(this->meanDtype, {this->invstdDtype, this->weightDtype, this->meanDyDtype, this->meanDyXmuDtype})) :
            AllSame(this->gradOutputDtype, {this->saveInputDtype, this->meanDtype, this->invstdDtype, this->weightDtype, this->meanDyDtype, this->meanDyXmuDtype});
    if (!valid) {
        std::string reasonMsg = isFloat16FloatCase ?
            "The dtype of save_input must be float16 and the dtypes of remaining tensors invstd, weight, mean_dy and mean_dy_xmu must be float, "
            "when the dtype of grad_output is float16 and the dtype of mean is float" :
            "The dtypes of all input parameters must be the same, when the dtype of grad_output is not float16 or the dtype of mean is not float";

        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(this->gradOutputDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(this->saveInputDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(this->meanDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(this->invstdDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(this->weightDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(this->meanDyDtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(this->meanDyXmuDtype);

        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(),
            "grad_output, save_input, mean, invstd, weight, mean_dy and mean_dy_xmu", dtypesStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormBackwardElemtTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(GRAD_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    if (this->gradOutputDtype != this->outputDtype) {
        std::string reasonMsg = "The dtype of output parameter grad_input must be the same as that (" +
                                ge::TypeUtils::DataTypeToSerialString(this->gradOutputDtype) + ") of input parameter grad_output";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "grad_input",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
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
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "grad_input",
            ge::TypeUtils::DataTypeToSerialString(this->gradOutputDtype).c_str(), "float16, bfloat16 and float");
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