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
 * \file binary_cross_entropy_grad_tiling_arch35.cpp
 * \brief
 */
#include "binary_cross_entropy_grad_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "op_util.h"
#include "loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_dag.h"
#include "loss/binary_cross_entropy_grad/op_kernel/arch35/binary_cross_entropy_grad_tiling_key.h"

using namespace Ops::Base;

namespace optiling {
constexpr static uint64_t BINARY_CROSS_ENTROPY_GRAD_COMMON_TILING_PRIORITY = 0;
const size_t ASCEND_WORKSPACE = 16777216; // 16MB

static const size_t INPUT_X_INDEX = 0;
static const size_t INPUT_Y_INDEX = 1;
static const size_t INPUT_GRAD_OUTPUT_INDEX = 2;
static const size_t INPUT_WEIGHT_INDEX = 3;
static const size_t OUTPUT_INDEX = 0;
static const size_t REDUCTION_INDEX = 0;
static const size_t REDUCTION_SUM = 1;
static const size_t REDUCTION_MEAN = 2;
static const size_t BCEG_HAS_WEIGHT = 1;
static const map<std::string, uint32_t> STR_2_INT = { { "none", 0 }, { "sum", 1 }, { "mean", 2 } };
static const map<ge::DataType, uint32_t> DTYEP_2_INT_KEY{ { ge::DT_FLOAT16, 10 },
    { ge::DT_FLOAT, 20 },
    { ge::DT_BF16, 30 } };

ge::graphStatus BinaryCrossEntropyGradTiling::CalcInputDtype()
{
    auto inputXDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    this->inputXDtype = inputXDesc->GetDataType();
    OP_CHECK_IF(
        this->inputXDtype != ge::DT_FLOAT16 && this->inputXDtype != ge::DT_BF16 && this->inputXDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "input X dtype not support"),
        return ge::GRAPH_FAILED);
    auto inputYDesc = context_->GetInputDesc(INPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputYDesc);
    this->inputYDtype = inputYDesc->GetDataType();
    OP_CHECK_IF(
        this->inputYDtype != ge::DT_FLOAT16 && this->inputYDtype != ge::DT_BF16 && this->inputYDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "input Y dtype not support"),
        return ge::GRAPH_FAILED);
    auto inputGradOutputDesc = context_->GetInputDesc(INPUT_GRAD_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGradOutputDesc);
    this->inputGradOutputDtype = inputGradOutputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputGradOutputDtype != ge::DT_FLOAT16 && this->inputGradOutputDtype != ge::DT_BF16 &&
        this->inputGradOutputDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "input GradOutput dtype not support"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyGradTiling::CalcOutputDtype()
{
    auto outputDesc = context_->GetOutputDesc(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), "output dtype not support"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputXDtype,
                    OP_LOGE(context_->GetNodeName(), "output dtype not same as inputs"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

float BinaryCrossEntropyGradTiling::CalcMeanCof()
{
    float meanCof = 1.0;
    int64_t dimVal = 1;
    if (this->isReductionMean) {
        auto labelStorageShape = context_->GetOutputShape(OUTPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, labelStorageShape);
        const gert::Shape& inputLabelShape = Ops::Base::EnsureNotScalar(labelStorageShape->GetStorageShape());
        const size_t dimLen = inputLabelShape.GetDimNum();
        for (uint32_t i = 0; i < dimLen; i++) {
            if (inputLabelShape.GetDim(i) != 0) {
                dimVal = dimVal * inputLabelShape.GetDim(i);
            } else {
                OP_LOGE(context_->GetNodeName(), "the shape[%u] is 0, do not supported", i);
                return ge::GRAPH_FAILED;
            }
        }
        meanCof = meanCof / static_cast<float>(dimVal);
    }
    OP_LOGD(context_->GetNodeName(), "[TilingData] : meanCof = %f", meanCof);
    return meanCof;
}

ge::graphStatus BinaryCrossEntropyGradTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "BinaryCrossEntropyGradTiling RunTiling enter.");
    if (context_ == nullptr) {
        OP_LOGE("BinaryCrossEntropyGrad", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    // check input dtype
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get input dtype failed"), return ge::GRAPH_FAILED);
    // check output dtype
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get output dtype failed"), return ge::GRAPH_FAILED);

    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();

    auto weightStorageShape = context_->GetOptionalInputShape(INPUT_WEIGHT_INDEX);
    if (weightStorageShape != nullptr) {
        this->hasWeight_ = static_cast<uint64_t>(1);
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    this->reductionStr = attrs->GetAttrPointer<char>(REDUCTION_INDEX);
    auto iter = STR_2_INT.find(this->reductionStr);
    OP_CHECK_IF(iter == STR_2_INT.end(),
                    OP_LOGE(context_->GetNodeName(), "reduction is not in none, mean or sum."),
                    return ge::GRAPH_FAILED);
    this->isReductionNone = strcmp(this->reductionStr, "none") == 0;
    this->isReductionMean = strcmp(this->reductionStr, "mean") == 0;
    this->isReductionSum = strcmp(this->reductionStr, "sum") == 0;

    this->meanCof_ = CalcMeanCof();
    if (input0DType == ge::DT_FLOAT16 || input0DType == ge::DT_BF16) {
        OP_CHECK_IF(RunFp16BroadcastTiling(this->meanCof_) == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(), "get input dtype failed"),
                        return ge::GRAPH_FAILED);
    } else if (input0DType == ge::DT_FLOAT) {
        OP_CHECK_IF(RunFp32BroadcastTiling(this->meanCof_) == ge::GRAPH_FAILED,
                        OP_LOGE(context_->GetNodeName(), "get input dtype failed"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context_->GetNodeName(),
            "input dtype is only support fp16, bf16, fp32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyGradTiling::RunFp16BroadcastTiling(float meanCof) {
    if (this->isReductionMean){
        OP_LOGD(context_->GetNodeName(), "use reductionStr mean");
        if (this->hasWeight_ == static_cast<uint64_t>(1)) {
            BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGMeanHasWeight<half, float>::OpDag> brcBaseTiling(context_);
            brcBaseTiling.SetScalar(meanCof);
            OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                            OP_LOGE(context_->GetNodeName(),
                            "Do tiling failed. Please check the detailed log."),
                            return ge::GRAPH_FAILED);
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_MEAN, BCEG_HAS_WEIGHT);
        } else {
            BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGMeanHasNoWeight<half, float>::OpDag> brcBaseTiling(context_);
            brcBaseTiling.SetScalar(meanCof);
            OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                            OP_LOGE(context_->GetNodeName(),
                            "Do tiling failed. Please check the detailed log."),
                            return ge::GRAPH_FAILED);
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_MEAN, 0);
        }
    } else if (this->isReductionNone || this->isReductionSum) {
        OP_LOGD(context_->GetNodeName(), "use reducation none or sum");
            if (this->hasWeight_ == static_cast<uint64_t>(1)) {
                BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGSumHasWeight<half, float>::OpDag> brcBaseTiling(context_);
                OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                                OP_LOGE(context_->GetNodeName(),
                                "Do tiling failed. Please check the detailed log."),
                                return ge::GRAPH_FAILED);
                this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_SUM, BCEG_HAS_WEIGHT);
            } else {
                BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGSumHasNoWeight<half, float>::OpDag> brcBaseTiling(context_);
                OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                                OP_LOGE(context_->GetNodeName(),
                                "Do tiling failed. Please check the detailed log."),
                                return ge::GRAPH_FAILED);
                this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_SUM, 0);
            }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyGradTiling::RunFp32BroadcastTiling(float meanCof) {
    if (this->isReductionNone || this->isReductionSum) {
        OP_LOGD(context_->GetNodeName(), "use reducation none or sum");
        if (this->hasWeight_ == static_cast<uint64_t>(1)) {
            BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGSumHasWeight<half, float>::OpDag> brcBaseTiling(context_);
            OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                            OP_LOGE(context_->GetNodeName(),
                            "Do tiling failed. Please check the detailed log."), return ge::GRAPH_FAILED);
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_SUM, BCEG_HAS_WEIGHT);
        } else {
            BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGSumHasNoWeight<half, float>::OpDag> brcBaseTiling(context_);
            OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                            OP_LOGE(context_->GetNodeName(),
                            "Do tiling failed. Please check the detailed log."), return ge::GRAPH_FAILED);
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_SUM, 0);
        }
    } else if (this->isReductionMean){
        OP_LOGD(context_->GetNodeName(), "use reductionStr mean");
        if (this->hasWeight_ == static_cast<uint64_t>(0)) {
            BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGMeanHasNoWeight<float, float>::OpDag> brcBaseTiling(context_);
            brcBaseTiling.SetScalar(meanCof);
            OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                            OP_LOGE(context_->GetNodeName(),
                            "Do tiling failed. Please check the detailed log."), return ge::GRAPH_FAILED);
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_MEAN, 0);
        } else {
            BroadcastBaseTiling<BinaryCrossEntropyGrad::BCEGMeanHasWeight<float, float>::OpDag> brcBaseTiling(context_);
            brcBaseTiling.SetScalar(meanCof);
            OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                            OP_LOGE(context_->GetNodeName(),
                            "Do tiling failed. Please check the detailed log."), return ge::GRAPH_FAILED);
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), REDUCTION_MEAN, BCEG_HAS_WEIGHT);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyGradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool BinaryCrossEntropyGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus BinaryCrossEntropyGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BinaryCrossEntropyGradTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus BinaryCrossEntropyGradTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyGradTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BinaryCrossEntropyGradTilingAscendC(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus Tiling4BinaryCrossEntropyGrad(gert::TilingContext* context)
{
    OP_LOGD("TilingForBinaryCrossEntropyGrad", "Enter TilingForBinaryCrossEntropyGrad");
    if (context == nullptr) {
        OP_LOGE("TilingForBinaryCrossEntropyGrad", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const BinaryCrossEntropyGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "Enter ascendc BinaryCrossEntropyGradTilingAscendC");
    return BinaryCrossEntropyGradTilingAscendC(context);
}

ge::graphStatus BinaryCrossEntropyGradTilingPrepareAscendC(gert::TilingParseContext* context)
{
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<BinaryCrossEntropyGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4BinaryCrossEntropyGrad(gert::TilingParseContext* context)
{
    OP_LOGD("TilingPrepareForBinaryCrossEntropyGrad", "Enter TilingPrepareForBinaryCrossEntropyGrad");
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForBinaryCrossEntropyGrad", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = context->GetCompiledInfo<BinaryCrossEntropyGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter BinaryCrossEntropyGradTilingPrepareAscendC");
    return BinaryCrossEntropyGradTilingPrepareAscendC(context);
}

// register tiling interface of the BinaryCrossEntropyGrad op.
IMPL_OP_OPTILING(BinaryCrossEntropyGrad)
    .Tiling(Tiling4BinaryCrossEntropyGrad)
    .TilingParse<BinaryCrossEntropyGradCompileInfo>(TilingPrepare4BinaryCrossEntropyGrad);

REGISTER_OPS_TILING_TEMPLATE(BinaryCrossEntropyGrad, BinaryCrossEntropyGradTiling,
    BINARY_CROSS_ENTROPY_GRAD_COMMON_TILING_PRIORITY);
}  // namespace optiling