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
 * \file smooth_l1_loss_v2_tiling_arch35.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "loss/smooth_l1_loss_v2/op_kernel/arch35/smooth_l1_loss_v2_dag.h"
#include "loss/smooth_l1_loss_v2/op_kernel/arch35/smooth_l1_loss_v2_tiling_key.h"
#include "smooth_l1_loss_v2_tiling_arch35.h"

using namespace std;

namespace {
const size_t INDEX_PREDICT = 0;
const size_t INPUT_INDEX_SHAPE = 1;
const size_t NEGTIVE_CONST_ONE = -1;
} // namespace
namespace optiling {
static const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;
static const uint32_t REDUCTION_MEAN = 2;
static const float NEGTIVE_CONST_HALF = -0.5;
static const map<std::string, uint32_t> STR_2_INT = {{"none", 0}, {"sum", 1}, {"mean", 2}};
static const map<ge::DataType, uint32_t> DTYEP_2_INT_KEY{{ge::DT_FLOAT16, 10}, {ge::DT_FLOAT, 20}, {ge::DT_BF16, 30}};
ge::graphStatus SmoothL1LossV2Tiling::CheckShape()
{
    auto predictStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, predictStorageShape);
    const gert::Shape& inputPShape = Ops::Base::EnsureNotScalar(predictStorageShape->GetStorageShape());
    auto labelStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, labelStorageShape);
    const gert::Shape& inputLShape = Ops::Base::EnsureNotScalar(labelStorageShape->GetStorageShape());
    auto inputPDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputPDesc);
    auto inputLDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputLDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    auto iter = DTYEP_2_INT_KEY.find(this->outputDtype);
    OP_CHECK_IF(
        iter == DTYEP_2_INT_KEY.end(), OP_LOGE(tilingContext->GetNodeName(), "output dtype is not support."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputPDesc->GetDataType() != inputLDesc->GetDataType(),
        OP_LOGE(tilingContext->GetNodeName(), "input predict and label dtype not same"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputPShape != inputLShape, OP_LOGE(tilingContext->GetNodeName(), "input predict and label shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SmoothL1LossV2Tiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter SetTilingData");
    auto rawTilingData = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTilingData);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, key.Reduction,
        key.Dtype);
    OP_LOGI(
        tilingContext->GetNodeName(),
        "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu, Reduction is : %u, Dtype is %u",
        key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, tilingKey,
        key.Reduction, key.Dtype);
    if (static_cast<int32_t>(this->reduction) < 1) {
        size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        currentWorkspace[0] = ASCEND_WORKSPACE;
        tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    }
    tilingContext->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SmoothL1LossV2Tiling::TilingEle()
{
    ElewiseBaseTiling eleBaseTiling(tilingContext);
    key.Dtype = DTYEP_2_INT_KEY.at(this->outputDtype);
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        // fp16需要cast成fp32处理
        OP_CHECK_IF(
            eleBaseTiling.DoTiling<SmoothL1LossV2::SmoothL1LossV2OpDag<half>::OpDag>(tiling->baseTiling) !=
                ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp16"), return ge::GRAPH_FAILED);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(
            eleBaseTiling.DoTiling<SmoothL1LossV2::SmoothL1LossV2OpDag<float>::OpDag>(tiling->baseTiling) !=
                ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp32"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "current dtype not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SmoothL1LossV2Tiling::TilingReduce(const SmoothL1LossV2CompileInfo* compileInfo)
{
    Ops::Base::ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (Ops::Base::ReduceOpTmpl::GetInputParam(tilingContext, opInput, 0) == ge::GRAPH_FAILED),
        OP_LOGE(tilingContext->GetNodeName(), "ReduceOp get x input failed"), return ge::GRAPH_FAILED);

    opInput.axes.resize(opInput.shape.size());
    for (size_t i = 0; i < opInput.shape.size(); i++) {
        opInput.axes[i] = i;
    }
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        if (static_cast<int32_t>(this->reduction) == 1) {
            OP_CHECK_IF(
                (Tiling4ReduceOp<SmoothL1LossV2::SmoothL1LossV2SumDag<half, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &(tiling->reduceTiling)) ==
                 ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "SmoothL1LossV2 Tiling failed"), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                (Tiling4ReduceOp<SmoothL1LossV2::SmoothL1LossV2MeanDag<half, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &(tiling->reduceTiling)) ==
                 ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "SmoothL1LossV2 Tiling failed"), return ge::GRAPH_FAILED);
        }
    } else {
        if (static_cast<int32_t>(this->reduction) == 1) {
            OP_CHECK_IF(
                (Tiling4ReduceOp<SmoothL1LossV2::SmoothL1LossV2SumDag<float, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &(tiling->reduceTiling)) ==
                 ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "SmoothL1LossV2 Tiling failed"), return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                (Tiling4ReduceOp<SmoothL1LossV2::SmoothL1LossV2MeanDag<float, float>::OpDag>(
                     tilingContext, opInput, key.ReduceTiling, &compileInfo->opInfo, &(tiling->reduceTiling)) ==
                 ge::GRAPH_FAILED),
                OP_LOGE(tilingContext->GetNodeName(), "SmoothL1LossV2 Tiling failed"), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SmoothL1LossV2Tiling::RunTiling(const SmoothL1LossV2CompileInfo* compileInfo)
{
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);
    tiling = tilingContext->GetTilingData<SmoothL1LossV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* sigmaPtr = attrs->GetAttrPointer<float>(0);
    const string reductionStr = string(attrs->GetAttrPointer<char>(1));
    float sigma = (sigmaPtr == nullptr) ? 1.0 : *sigmaPtr;
    OP_CHECK_IF(
        sigma <= 0, OP_LOGE(tilingContext->GetNodeName(), "the value of sigma must be positive number."),
        return ge::GRAPH_FAILED);
    tiling->Sigma = sigma;
    tiling->MultiplyValue = 1 / sigma;
    tiling->AddsValue = NEGTIVE_CONST_HALF * sigma;
    auto iter = STR_2_INT.find(reductionStr);
    OP_CHECK_IF(
        iter == STR_2_INT.end(),
        OP_LOGE(
            tilingContext->GetNodeName(), "reduction %s is not support, should be one of {mean, sum, none}.",
            reductionStr.c_str()),
        return ge::GRAPH_FAILED);
    this->reduction = iter->second;
    key.Reduction = this->reduction;
    if (reductionStr == "none") {
        OP_CHECK_IF(
            TilingEle() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for elewise"),
            return ge::GRAPH_FAILED);
    } else if (reductionStr == "mean" || reductionStr == "sum") {
        OP_CHECK_IF(
            TilingReduce(compileInfo) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for reduce"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "reduction %s is not supported", reductionStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return SetTilingData();
}

ge::graphStatus TilingForSmoothL1LossV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "%s begin.", __func__);
    auto predict = context->GetInputShape(INDEX_PREDICT);
    OP_CHECK_NULL_WITH_CONTEXT(context, predict);
    auto compileInfo = reinterpret_cast<const SmoothL1LossV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的模板tiling
    OP_LOGD("SmoothL1LossV2Tiling", "Enter new SmoothL1LossV2Tiling");
    SmoothL1LossV2Tiling smoothTiling(context);
    return smoothTiling.RunTiling(compileInfo);
}

static ge::graphStatus TilingPrepareForSmoothL1LossV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "%s begin.", __func__);
    auto compileInfo = context->GetCompiledInfo<SmoothL1LossV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}
// register tiling interface of the SmoothL1LossV2.
IMPL_OP_OPTILING(SmoothL1LossV2)
    .Tiling(TilingForSmoothL1LossV2)
    .TilingParse<SmoothL1LossV2CompileInfo>(TilingPrepareForSmoothL1LossV2);

} // namespace optiling
