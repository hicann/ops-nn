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
 * \file kl_div_target_loss_grad_tiling.cpp
 * \brief kl_div_target_loss_grad_tiling
 */
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/kl_div_target_loss_grad_dag.h"
#include "../../op_kernel/arch35/kl_div_target_loss_grad_tiling_key.h"
#include "kl_div_target_loss_grad_tiling_arch35.h"
#include "kl_div_target_loss_grad_tiling.h"
#include <map>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {
constexpr static uint64_t KL_DIV_TARGET_LOSS_GRAD_COMMON_TILING_PRIORITY = 0;

const size_t ASCEND_WORKSPACE = 16777216; // 16MB

static const size_t INPUT_GRAD_INDEX = 0;
static const size_t INPUT_INPUT_INDEX = 1;
static const size_t INPUT_TARGET_INDEX = 2;
static const size_t OUTPUT_Y_INDEX = 0;

static const size_t KDTLG_LOG_TARGET = 1;
static const size_t LOG_TARGET_FALSE = 0;
static const size_t LOG_TARGET_TRUE = 1;

static const size_t REDUCTION_INDEX = 0;
static const size_t LOGTARGET_INDEX = 1;
static const std::map<std::string, uint32_t> STR_2_INT = { { "none", 0 }, { "sum", 1 }, { "mean", 2 },{ "batchmean", 3 } };

ge::graphStatus KlDivTargetLossGradTiling::CalcInputDtype()
{
    auto inputGradDesc = context_->GetInputDesc(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGradDesc);
    this->inputGradDtype = inputGradDesc->GetDataType();
    OP_CHECK_IF(
        this->inputGradDtype != ge::DT_FLOAT16 && this->inputGradDtype != ge::DT_BF16 && this->inputGradDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "input Grad dtype allow {float16, bfloat16, float32} ,but not support %s",
                                        ge::TypeUtils::DataTypeToSerialString(this->inputGradDtype).c_str()),
                                        return ge::GRAPH_FAILED);

    auto inputInputDesc = context_->GetInputDesc(INPUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputInputDesc);
    this->inputInputDtype = inputInputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputInputDtype != ge::DT_FLOAT16 && this->inputInputDtype != ge::DT_BF16 && this->inputInputDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "input Input dtype allow {float16, bfloat16, float32} ,but not support %s",
                                        ge::TypeUtils::DataTypeToSerialString(this->inputInputDtype).c_str()),
                                        return ge::GRAPH_FAILED);

    auto inputTargetDesc = context_->GetInputDesc(INPUT_TARGET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputTargetDesc);
    this->inputTargetDtype = inputTargetDesc->GetDataType();
    OP_CHECK_IF(
        this->inputTargetDtype != ge::DT_FLOAT16 && this->inputTargetDtype != ge::DT_BF16 &&
        this->inputTargetDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "input Target dtype allow {float16, bfloat16, float32} ,but not support %s",
                                        ge::TypeUtils::DataTypeToSerialString(this->inputTargetDtype).c_str()),
                                        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTiling::CalcDiffDtype()
{
    auto gradDesc = context_->GetInputDesc(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradDesc);
    auto inputDesc = context_->GetInputDesc(INPUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto targetDesc = context_->GetInputDesc(INPUT_TARGET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, targetDesc);
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);

    this->inputGradDtype = gradDesc->GetDataType();
    this->inputInputDtype = inputDesc->GetDataType();
    this->inputTargetDtype = targetDesc->GetDataType();
    this->outputYDtype = outputYDesc->GetDataType();
    OP_CHECK_IF(
        this->inputGradDtype != this->inputInputDtype,
        OP_LOGE(context_->GetNodeName(), "dtype of grad[%s] and dtype of input[%s] not same",
                                        ge::TypeUtils::DataTypeToSerialString(this->inputGradDtype).c_str(),
                                        ge::TypeUtils::DataTypeToSerialString(this->inputInputDtype).c_str()),
                                        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        this->inputGradDtype != this->inputTargetDtype,
        OP_LOGE(context_->GetNodeName(), "dtype of grad[%s] and dtype of target[%s] not same",
                                        ge::TypeUtils::DataTypeToSerialString(this->inputGradDtype).c_str(),
                                        ge::TypeUtils::DataTypeToSerialString(this->inputTargetDtype).c_str()),
                                        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        this->inputGradDtype != this->outputYDtype,
        OP_LOGE(context_->GetNodeName(), "dtype of grad[%s] and dtype of y[%s] not same",
                                        ge::TypeUtils::DataTypeToSerialString(this->inputGradDtype).c_str(),
                                        ge::TypeUtils::DataTypeToSerialString(this->outputYDtype).c_str()),
                                        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTiling::CalcOutputDtype()
{
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    this->outputYDtype = outputYDesc->GetDataType();
    OP_CHECK_IF(
        this->outputYDtype != ge::DT_FLOAT16 && this->outputYDtype != ge::DT_BF16 && this->outputYDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "output Y dtype allow {float16, bfloat16, float32} ,but not support %s",
                                        ge::TypeUtils::DataTypeToSerialString(this->outputYDtype).c_str()),
                                        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->outputYDtype != this->inputGradDtype,
        OP_LOGE(context_->GetNodeName(), "output Y dtype is not same as inputs"),
                                        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

float KlDivTargetLossGradTiling::CalcReductionCof(const gert::Shape& inputLabelShape)
{
    float reducationCof = 1.0;
    float negReductionCof = -1.0f;
    int64_t dimVal = 1;
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
        OP_LOGE(context_->GetNodeName(), "attrs is nullptr"),
        return negReductionCof);

    this->reducationStr = attrs->GetAttrPointer<char>(REDUCTION_INDEX);
    auto iter = STR_2_INT.find(this->reducationStr);
    OP_CHECK_IF(iter == STR_2_INT.end(),
                    OP_LOGE(context_->GetNodeName(), "reduction is %s not in [none, mean , sum , batchmean].",this->reducationStr),
                    return negReductionCof);

    if (strcmp(this->reducationStr, "mean") == 0) {
        const size_t dimLen = inputLabelShape.GetDimNum();
        for (uint32_t i = 0; i < dimLen; i++) {
            if (inputLabelShape.GetDim(i) != 0) {
                dimVal = dimVal * inputLabelShape.GetDim(i);
            } else {
                OP_LOGE(context_->GetNodeName(), "The  [%u]  axis of the input is 0, which is not supported", i);
                return negReductionCof;
            }
        }
    } else if (strcmp(this->reducationStr, "batchmean") == 0){
        if (inputLabelShape.GetDim(0) != 0) {
            dimVal = dimVal * inputLabelShape.GetDim(0);
        } else {
            OP_LOGE(context_->GetNodeName(), "The 0 axis of the input is 0, which is not supported");
            return negReductionCof;
        }
    }
    OP_CHECK_IF(dimVal <= 0,
        OP_LOGE(context_->GetNodeName(), "dimVal must greater 0, but is %ld", dimVal),
        return negReductionCof);
    OP_LOGD(context_->GetNodeName(), "[dimVal] : dimVal = %ld", dimVal);
    reducationCof = static_cast<float>(reducationCof / static_cast<double>(dimVal));
    OP_LOGD(context_->GetNodeName(), "[TilingData] : reducationCof = %f", reducationCof);
    return reducationCof;
}


ge::graphStatus KlDivTargetLossGradTiling::DoOpTiling()
{
    if (context_ == nullptr) {
        OP_LOGE("KlDivTargetLossGrad", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "KlDivTargetLossGradTiling RunTiling enter.");

    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get input dtype failed"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(CalcDiffDtype() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get dtype failed, all input and out dtype need same"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get output dtype failed"), return ge::GRAPH_FAILED);

    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const bool* getLogtarget = attrs->GetAttrPointer<bool>(LOGTARGET_INDEX);
    this->logTarget_ = (getLogtarget != nullptr) ? *getLogtarget : false;
    OP_LOGD(context_->GetNodeName(), "logTarget is %d", static_cast<int32_t>(this->logTarget_));
    auto inputShape = context_->GetInputShape(INPUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto inputLabelShape = EnsureNotScalar(inputShape->GetStorageShape());
    this->reducationCof_ = CalcReductionCof(inputLabelShape);
    OP_CHECK_IF(this->reducationCof_ < 0.0f,
        OP_LOGE(context_->GetNodeName(), "reducationCof must > 0"), return ge::GRAPH_FAILED);
    if (input0DType == ge::DT_FLOAT16 || input0DType == ge::DT_BF16) {
        OP_CHECK_IF(RunFp16BroadcastTiling(this->reducationCof_) == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(), "get input dtype failed"),
                        return ge::GRAPH_FAILED);
    }  else if (input0DType == ge::DT_FLOAT) {
        OP_CHECK_IF(RunFp32BroadcastTiling(this->reducationCof_) == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(), "get input dtype failed"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context_->GetNodeName(),
            "input dtype is only support float16, bfloat16, float32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTiling::RunFp16BroadcastTiling(float reducationCof) 
{
    if (this->logTarget_){
        OP_LOGD(context_->GetNodeName(), "use logTarget true");
        BroadcastBaseTiling<KlDivTargetLossGrad::KDTLGLogTargetTrue<half, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(reducationCof);
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(),
                        "Do tiling failed. Please check the detailed log."),
                        return ge::GRAPH_FAILED);
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), LOG_TARGET_TRUE);
    } else {
        OP_LOGD(context_->GetNodeName(), "use logTarget false");
        BroadcastBaseTiling<KlDivTargetLossGrad::KDTLGLogTargetFalse<half, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(reducationCof);
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(),
                        "Do tiling failed. Please check the detailed log."),
                        return ge::GRAPH_FAILED);
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), LOG_TARGET_FALSE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool KlDivTargetLossGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus KlDivTargetLossGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t KlDivTargetLossGradTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus KlDivTargetLossGradTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTiling::RunFp32BroadcastTiling(float reducationCof)
{
    if (this->logTarget_ ){
        OP_LOGD(context_->GetNodeName(), "use logTarget true");
        BroadcastBaseTiling<KlDivTargetLossGrad::KDTLGLogTargetTrue<float, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(reducationCof);
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(),
                                                        "Do tiling failed. Please check the detailed log."),
                        return ge::GRAPH_FAILED);
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), LOG_TARGET_TRUE);
    } else {
        OP_LOGD(context_->GetNodeName(), "use logTarget false");
        BroadcastBaseTiling<KlDivTargetLossGrad::KDTLGLogTargetFalse<float, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(reducationCof);
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(),
                                                        "Do tiling failed. Please check the detailed log."),
                        return ge::GRAPH_FAILED);
        this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), LOG_TARGET_FALSE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KlDivTargetLossGradTilingAscendC(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus Tiling4KlDivTargetLossGrad(gert::TilingContext* context)
{
    OP_LOGD("TilingForKlDivTargetLossGrad", "Enter TilingForKlDivTargetLossGrad");
    if (context == nullptr) {
        OP_LOGE("TilingForKlDivTargetLossGrad", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const KlDivTargetLossGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "Enter ascendc KlDivTargetLossGradTilingAscendC");
    return KlDivTargetLossGradTilingAscendC(context);
}

ge::graphStatus KlDivTargetLossGradTilingPrepareAscendC(gert::TilingParseContext* context)
{
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<KlDivTargetLossGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                    OP_LOGE(context->GetNodeName(),
                                                    "get core num failed, core num: %lu",
                                                    compileInfo->coreNum),
                    return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                    OP_LOGE(context->GetNodeName(),
                                                    "get ub size failed, ub size: %lu",
                                                    compileInfo->ubSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4KlDivTargetLossGrad(gert::TilingParseContext* context)
{
    (void) context;
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the KlDivTargetLossGrad op.
IMPL_OP_OPTILING(KlDivTargetLossGrad)
    .Tiling(Tiling4KlDivTargetLossGrad)
    .TilingParse<KlDivTargetLossGradCompileInfo>(TilingPrepare4KlDivTargetLossGrad);

REGISTER_OPS_TILING_TEMPLATE(KlDivTargetLossGrad, KlDivTargetLossGradTiling,
    KL_DIV_TARGET_LOSS_GRAD_COMMON_TILING_PRIORITY);
}  // namespace optiling
