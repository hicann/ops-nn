/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_bn_training_update_tiling_arch35.cc
 * \brief
 */
#include "sync_bn_training_update_tiling_arch35.h"
#include "norm/sync_bn_training_update/op_kernel/arch35/sync_bn_training_update_dag.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include <cmath>
#include <graph/utils/type_utils.h>

using namespace ge;

namespace optiling {
constexpr uint32_t SYNC_BN_TRAINING_UPDATE_TILING_KEY_ELEMENTWISE = 0;
constexpr uint32_t MEAN_INPUT_INDEX = 0;
constexpr uint32_t RUNNING_MEAN_INPUT_INDEX = 1;
constexpr uint32_t RUNNING_MEAN_UPDATE_OUTPUT_INDEX = 0;
constexpr float DEFAULT_MOMENTUM = 0.1;

ge::graphStatus SyncBNTrainingUpdateTiling::SetTilingData()
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
    tilingContext->SetTilingKey(SYNC_BN_TRAINING_UPDATE_TILING_KEY_ELEMENTWISE);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

void SyncBNTrainingUpdateTiling::PrintTilingData()
{
    OP_LOGD(opName, "momentum:%f.\n", momentum_);
    OP_LOGD(opName, "baseTiling.scheMode:%ld.\n", tiling->baseTiling.scheMode);
    OP_LOGD(opName, "baseTiling.dim0:%ld.\n", tiling->baseTiling.dim0);
    OP_LOGD(opName, "baseTiling.blockFormer:%ld.\n", tiling->baseTiling.blockFormer);
    OP_LOGD(opName, "baseTiling.blockNum:%ld.\n", tiling->baseTiling.blockNum);
    OP_LOGD(opName, "baseTiling.ubFormer:%d.\n", tiling->baseTiling.ubFormer);
    OP_LOGD(opName, "baseTiling.ubLoopOfFormerBlock:%ld.\n", tiling->baseTiling.ubLoopOfFormerBlock);
    OP_LOGD(opName, "baseTiling.ubLoopOfTailBlock:%ld.\n", tiling->baseTiling.ubLoopOfTailBlock);
    OP_LOGD(opName, "baseTiling.ubTailOfFormerBlock:%ld.\n", tiling->baseTiling.ubTailOfFormerBlock);
    OP_LOGD(opName, "baseTiling.ubTailOfTailBlock:%ld.\n", tiling->baseTiling.ubTailOfTailBlock);
    OP_LOGD(opName, "baseTiling.elemNum:%ld.\n", tiling->baseTiling.elemNum);
}

ge::graphStatus SyncBNTrainingUpdateTiling::CheckTensorDtype()
{
    auto meanDesc = tilingContext->GetInputDesc(MEAN_INPUT_INDEX);
    auto runningMeanDesc = tilingContext->GetInputDesc(RUNNING_MEAN_INPUT_INDEX);
    auto runningMeanUpdateDesc = tilingContext->GetOutputDesc(RUNNING_MEAN_UPDATE_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, meanDesc);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, runningMeanDesc);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, runningMeanUpdateDesc);

    meanDtype = meanDesc->GetDataType();
    runningMeanDtype = runningMeanDesc->GetDataType();
    runningMeanUpdateDtype = runningMeanUpdateDesc->GetDataType();

    OP_LOGD(opName, "SyncBNTrainingUpdateTiling CheckTensorDtype proc meanDtype meanDtype[%s], runningMeanDtype[%s], runningMeanUpdateDtype[%s]",
        ge::TypeUtils::DataTypeToSerialString(meanDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(runningMeanDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(runningMeanUpdateDtype).c_str());

    OP_CHECK_IF(
        meanDtype != ge::DT_FLOAT16 && meanDtype != ge::DT_BF16 && meanDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "mean", ge::TypeUtils::DataTypeToSerialString(meanDtype).c_str(), "float16, bfloat16 and float"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        runningMeanDtype != ge::DT_FLOAT16 && runningMeanDtype != ge::DT_BF16 && runningMeanDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "running_mean", ge::TypeUtils::DataTypeToSerialString(runningMeanDtype).c_str(),
            "float16, bfloat16 and float"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        runningMeanUpdateDtype != ge::DT_FLOAT16 && runningMeanUpdateDtype != ge::DT_BF16 &&
            runningMeanUpdateDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "running_mean_update", ge::TypeUtils::DataTypeToSerialString(runningMeanUpdateDtype).c_str(),
            "float16, bfloat16 and float"),
        return ge::GRAPH_FAILED);

    if (meanDtype != runningMeanDtype || meanDtype != runningMeanUpdateDtype) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(meanDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(runningMeanDtype) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(runningMeanUpdateDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "mean, running_mean and running_mean_update", dtypeMsg.c_str(),
            "The dtypes of mean, running_mean and running_mean_update must be the same");
        return ge::GRAPH_FAILED;
    }

    this->meanDtype = meanDtype;
    this->runningMeanDtype = runningMeanDtype;
    this->runningMeanUpdateDtype = runningMeanUpdateDtype;

    return ge::GRAPH_SUCCESS;
}

void SyncBNTrainingUpdateTiling::SetAttr()
{
    auto attrs = tilingContext->GetAttrs();
    const float* momenum = attrs->GetFloat(0);
    momentum_ = (momenum == nullptr) ? DEFAULT_MOMENTUM : *momenum;
    tiling->momentum = momentum_;
}

ge::graphStatus SyncBNTrainingUpdateTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CheckTensorDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "CheckTensorDtype failed"),
        return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<SyncBNTrainingUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    OP_LOGD(opName, "SyncBNTrainingUpdateTiling RunTiling proc meanDtype is[%s]",
        ge::TypeUtils::DataTypeToSerialString(this->meanDtype).c_str());
    if (this->meanDtype == ge::DT_FLOAT16) {
        res = elewiseBaseTiling.DoTiling<SyncBNTrainingUpdateDag<half>::OpDag>(tiling->baseTiling);
    } else if (this->meanDtype == ge::DT_FLOAT) {
        res = elewiseBaseTiling.DoTiling<SyncBNTrainingUpdateDag<float>::OpDag>(tiling->baseTiling);
    } else if (this->meanDtype == ge::DT_BF16) {
        res = elewiseBaseTiling.DoTiling<SyncBNTrainingUpdateDag<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            opName, "mean", ge::TypeUtils::DataTypeToSerialString(meanDtype).c_str(), "float16, bfloat16 and float");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "DoTiling failed"),
        return ge::GRAPH_FAILED);

    SetAttr();
    ge::graphStatus result = SetTilingData();

    /* 打印tiling的内容 */
    PrintTilingData();
    return result;
}

ge::graphStatus Tiling4SyncBNTrainingUpdate(gert::TilingContext *context)
{
    OP_LOGD(context, "Tiling4SyncBNTrainingUpdate is running");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    SyncBNTrainingUpdateTiling SyncBNTrainingUpdateTiling(context);
    return SyncBNTrainingUpdateTiling.RunTiling();
}

ge::graphStatus TilingPrepare4SyncBNTrainingUpdate([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SyncBNTrainingUpdate).Tiling(Tiling4SyncBNTrainingUpdate).TilingParse<ElewiseCompileInfo>(TilingPrepare4SyncBNTrainingUpdate);
}  // namespace optiling