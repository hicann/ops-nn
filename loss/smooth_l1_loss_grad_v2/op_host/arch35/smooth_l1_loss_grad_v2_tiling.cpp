/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_grad_v2_tiling.cpp
 * \brief smooth_l1_loss_grad_v2_tiling
 */

#include <graph/utils/type_utils.h>
#include <map>
#include "atvoss/broadcast/broadcast_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "smooth_l1_loss_grad_v2_tiling_base.h"
#include "smooth_l1_loss_grad_v2_tiling.h"
#include "../../op_kernel/arch35/smooth_l1_loss_grad_v2_tiling_key.h"
#include "../../op_kernel/arch35/smooth_l1_loss_grad_v2_dag.h"

using namespace ge;

namespace optiling {

constexpr static uint64_t COMMON_TILING_PRIORITY = 0;
constexpr static int32_t INPUT_PREDICT_IDX = 0;
constexpr static int32_t INPUT_LABEL_IDX = 1;
constexpr static int32_t INPUT_DOUT_IDX = 2;
constexpr static int32_t OUTPUT_IDX = 0;
constexpr static float NEGTIVE_ONE = static_cast<float>(-1.0);
static const std::map<std::string, uint32_t> STR_2_INT = { { "none", 0 }, { "sum", 1 }, { "mean", 2 } };
 
ge::graphStatus SmoothL1LossGradV2TilingClass::GetShapeAttrsInfo()
{
    auto predictDesc = context_->GetInputDesc(INPUT_PREDICT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, predictDesc);
    auto labelDesc = context_->GetInputDesc(INPUT_LABEL_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, labelDesc);
    auto doutDesc = context_->GetInputDesc(INPUT_DOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, doutDesc);
    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);

    ge::DataType predictDType = predictDesc->GetDataType();
    OP_CHECK_IF(predictDType != ge::DT_FLOAT16 && predictDType != ge::DT_BF16 && predictDType != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(), 
            "input dtype is only support float16, bfloat16, float32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(predictDType).c_str()),
        return ge::GRAPH_FAILED);
    ge::DataType labelDType = labelDesc->GetDataType();
    ge::DataType doutDType = doutDesc->GetDataType();
    ge::DataType outputDType = outputDesc->GetDataType();
    OP_CHECK_IF(predictDType != labelDType,
        OP_LOGE(context_->GetNodeName(), "dtype of predict[%s] and dtype of label[%s] not same",
                                        ge::TypeUtils::DataTypeToSerialString(predictDType).c_str(),
                                        ge::TypeUtils::DataTypeToSerialString(labelDType).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(predictDType != doutDType,
        OP_LOGE(context_->GetNodeName(), "dtype of predict[%s] and dtype of dout[%s] not same",
                                        ge::TypeUtils::DataTypeToSerialString(predictDType).c_str(),
                                        ge::TypeUtils::DataTypeToSerialString(doutDType).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(predictDType != outputDType,
        OP_LOGE(context_->GetNodeName(), "dtype of predict[%s] and dtype of y[%s] not same",
                                        ge::TypeUtils::DataTypeToSerialString(predictDType).c_str(),
                                        ge::TypeUtils::DataTypeToSerialString(outputDType).c_str()),
        return ge::GRAPH_FAILED);
    this->inputDtype = predictDType;
    OP_LOGD(context_->GetNodeName(), "[TilingData] : inputDtype is %s", ge::TypeUtils::DataTypeToSerialString(this->inputDtype).c_str());
    
    return ge::GRAPH_SUCCESS;
}
 
bool SmoothL1LossGradV2TilingClass::IsCapable()
{
    return true;
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::GetDoutIsScalar()
{
    auto inputDoutShape = context_->GetInputShape(INPUT_DOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDoutShape);
    auto storageShape = inputDoutShape->GetStorageShape();
    if (storageShape.IsScalar() || storageShape.GetShapeSize() == 1) {
        this->doutIsScalar = static_cast<uint32_t>(ATTR_IS_TRUE);
    }
    OP_LOGD(context_->GetNodeName(), "[TilingData] : doutIsScalar is %u", this->doutIsScalar);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::CalcReduceMeanCof()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    this->reducationStr = attrs->GetAttrPointer<char>(1);
    auto iter = STR_2_INT.find(this->reducationStr);
    OP_CHECK_IF(iter == STR_2_INT.end(),
                    OP_LOGE(context_->GetNodeName(), "reduction is not in [none, mean, sum]"),
                    return ge::GRAPH_FAILED);
    this->reduceMeanCof = 1.0f;
    int64_t dimVal = 1;
    if (strcmp(this->reducationStr, "mean") == 0) {
        auto inputStorageShape = context_->GetInputShape(INPUT_PREDICT_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputStorageShape);
        const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());
        const size_t dimLen = inputShape.GetDimNum();
        for (uint32_t i = 0; i < dimLen; i++) {
            if (inputShape.GetDim(i) != 0) {
                dimVal = dimVal * inputShape.GetDim(i);
            } else {
                OP_LOGE(context_->GetNodeName(), "the shape[%d] of output is 0, do not supported", i);
                return ge::GRAPH_FAILED;
            }
        }
        this->reduceMeanCof = static_cast<float>(this->reduceMeanCof / static_cast<double>(dimVal));
    }
    OP_LOGD(context_->GetNodeName(), "[TilingData] : reduceMeanCof = %f", this->reduceMeanCof);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SmoothL1LossGradV2TilingClass::CalcSigma()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
 
    const float* sigmaPtr = attrs->GetAttrPointer<float>(0);
    this->sigma = (sigmaPtr == nullptr) ? 1.0 : *sigmaPtr;
    OP_CHECK_IF(this->sigma <= 0,
        OP_LOGE(context_->GetNodeName(), "the value must be non-negtive."),
        return ge::GRAPH_FAILED);
    this->negSigma = NEGTIVE_ONE * this->sigma;
    this->invertSigma = 1 / this->sigma;
    OP_LOGD(context_->GetNodeName(), "[TilingData] : sigma = %f, negSigma = %f, invertSigma = %f", 
            this->sigma, this->negSigma, this->invertSigma);
    return ge::GRAPH_SUCCESS;
 }

template <typename T>
ge::graphStatus SmoothL1LossGradV2TilingClass::DoScalarDagTilingForType() {
    Ops::Base::BroadcastBaseTiling<typename SmoothL1LossGradV2Op::SmoothL1LossGradV2OpScalarDag<T, float>::OpDag> brcBaseTiling(context_);
    brcBaseTiling.SetScalar(this->sigma);
    brcBaseTiling.SetScalar(this->negSigma);
    brcBaseTiling.SetScalar(this->invertSigma);
    brcBaseTiling.SetScalar(this->reduceMeanCof);
    OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(),
                        "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
    this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::DoScalarDagOpTiling()
{
    if (this->inputDtype == ge::DT_FLOAT16) {
        return DoScalarDagTilingForType<Ops::Base::half>();
    } else if (this->inputDtype == ge::DT_BF16) {
        return DoScalarDagTilingForType<Ops::Base::bfloat16_t>();
    } else if (this->inputDtype == ge::DT_FLOAT) {
        return DoScalarDagTilingForType<float>();
    } else {
        OP_LOGE(context_->GetNodeName(),
            "input dtype is only support float16, bfloat16, float32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
}

template <typename T>
ge::graphStatus SmoothL1LossGradV2TilingClass::DoDagTilingForType()
{
    Ops::Base::BroadcastBaseTiling<typename SmoothL1LossGradV2Op::SmoothL1LossGradV2OpDag<T, float>::OpDag> brcBaseTiling(context_);
    brcBaseTiling.SetScalar(this->sigma);
    brcBaseTiling.SetScalar(this->negSigma);
    brcBaseTiling.SetScalar(this->invertSigma);
    brcBaseTiling.SetScalar(this->reduceMeanCof);
     OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(),
                                                    "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
    this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::DoTensorDagOpTiling()
{
    if (this->inputDtype == ge::DT_FLOAT16) {
        return DoDagTilingForType<Ops::Base::half>();
    } else if (this->inputDtype == ge::DT_BF16) {
        return DoDagTilingForType<Ops::Base::bfloat16_t>();
    } else if (this->inputDtype == ge::DT_FLOAT) {
        return DoDagTilingForType<float>();
    } else {
        OP_LOGE(context_->GetNodeName(),
            "input dtype is only support float16, bfloat16, float32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::DoOpTiling()
{
    OP_CHECK_IF(GetDoutIsScalar() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "check dout is scalar failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcReduceMeanCof() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get reduceMeanCof failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcSigma() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get sigma failed"), return ge::GRAPH_FAILED);
    if (this->doutIsScalar == static_cast<uint32_t>(ATTR_IS_TRUE)) {
        OP_CHECK_IF(DoScalarDagOpTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "do tiling failed, when dout is scalar"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(DoTensorDagOpTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "do tiling failed, when dout is tensor"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}
 
uint64_t SmoothL1LossGradV2TilingClass::GetTilingKey() const
{
    return tilingKey;
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SmoothL1LossGradV2TilingClass::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus SmoothL1LossGradV2TilingClass::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus TilingForSmoothL1LossGradV2(gert::TilingContext* context)
{
    OP_LOGD("SmoothL1LossGradV2Tiling", "Enter TilingForSmoothL1LossGradV2");
    if (context == nullptr) {
        OP_LOGE("SmoothL1LossGradV2Tiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const SmoothL1LossGradV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc SmoothL1LossGradV2Tiling");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}
 
 
ge::graphStatus SmoothL1LossGradV2TilingPrepareAscendC(gert::TilingParseContext* context)
{
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<SmoothL1LossGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                    OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %lu",
                                                    compileInfo->coreNum),
                    return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF((compileInfo->ubSize <= 0),
                    OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %lu",
                                                    compileInfo->ubSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus TilingPrepareForSmoothL1LossGradV2(gert::TilingParseContext* context)
{
    OP_LOGD("TilingPrepareForSmoothL1LossGradV2", "Enter TilingPrepareForSmoothL1LossGradV2");
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForSmoothL1LossGradV2", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = context->GetCompiledInfo<SmoothL1LossGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter SmoothL1LossGradV2TilingPrepareAscendC");
    return SmoothL1LossGradV2TilingPrepareAscendC(context);
}
IMPL_OP_OPTILING(SmoothL1LossGradV2)
    .Tiling(TilingForSmoothL1LossGradV2)
    .TilingParse<SmoothL1LossGradV2CompileInfo>(TilingPrepareForSmoothL1LossGradV2);

REGISTER_OPS_TILING_TEMPLATE(SmoothL1LossGradV2, SmoothL1LossGradV2TilingClass, COMMON_TILING_PRIORITY);
}   // namespace optiling