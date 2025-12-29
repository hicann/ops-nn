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
 * \file mse_loss_grad_tiling.cc
 * \brief mse_loss_grad_tiling
 */

#include <graph/utils/type_utils.h>
#include "mse_loss_grad_tiling_arch35.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "loss/mse_loss_grad/op_kernel/arch35/mse_loss_grad_tiling_key.h"
#include "loss/mse_loss_grad/op_kernel/arch35/mse_loss_grad_dag.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

namespace optiling {

constexpr static uint64_t COMMON_TILING_PRIORITY = 0;
constexpr static int32_t INPUT_PREDICT_IDX = 0;
constexpr static int32_t INPUT_LABEL_IDX = 1;
constexpr static int32_t INPUT_DOUT_IDX = 2;
constexpr static int32_t OUTPUT_IDX = 0;
static const std::map<std::string, uint32_t> STR_2_INT = {{"none", 0}, {"sum", 1}, {"mean", 2}};

inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

ge::graphStatus MseLossGradTilingClass::GetShapeAttrsInfo()
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
    ge::DataType labelDType = labelDesc->GetDataType();
    ge::DataType doutDType = doutDesc->GetDataType();
    ge::DataType outputDType = outputDesc->GetDataType();
    OP_CHECK_IF((predictDType != labelDType),
        OP_LOGE(context_->GetNodeName(), "dtype of predict[%s] and dtype of label[%s] not same",
            ge::TypeUtils::DataTypeToSerialString(predictDType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(labelDType).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((predictDType != doutDType),
        OP_LOGE(context_->GetNodeName(), "dtype of predict[%s] and dtype of dout[%s] not same",
            ge::TypeUtils::DataTypeToSerialString(predictDType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(doutDType).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((predictDType != outputDType),
        OP_LOGE(context_->GetNodeName(), "dtype of predict[%s] and dtype of y[%s] not same",
            ge::TypeUtils::DataTypeToSerialString(predictDType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputDType).c_str()),
        return ge::GRAPH_FAILED);
    this->inputDtype = predictDType;
    return ge::GRAPH_SUCCESS;
}

bool MseLossGradTilingClass::IsCapable()
{
    return true;
}

ge::graphStatus MseLossGradTilingClass::CheckDoutIsScalar()
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

ge::graphStatus MseLossGradTilingClass::CalcReduceMeanCof()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    this->reducationStr = attrs->GetAttrPointer<char>(0);
    auto iter = STR_2_INT.find(this->reducationStr);
    OP_CHECK_IF((iter == STR_2_INT.end()),
        OP_LOGE(context_->GetNodeName(), "reduction is not in [none, mean, sum]"),
        return ge::GRAPH_FAILED);
    this->reduceMeanCof = 2.0f;
    int64_t dimVal = 1;
    if (strcmp(this->reducationStr, "mean") == 0) {
        auto inputStorageShape = context_->GetInputShape(INPUT_PREDICT_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputStorageShape);
        const gert::Shape& inputShape = EnsureNotScalar(inputStorageShape->GetStorageShape());
        const size_t dimLen = inputShape.GetDimNum();
        for (uint32_t i = 0; i < dimLen; i++) {
            if (inputShape.GetDim(i) != 0) {
                dimVal = dimVal * inputShape.GetDim(i);
            } else {
                OP_LOGE(context_->GetNodeName(), "the shape[%u] of output is 0, do not supported", i);
                return ge::GRAPH_FAILED;
            }
        }
        this->reduceMeanCof = static_cast<float>(this->reduceMeanCof / static_cast<double>(dimVal));
    }
    OP_LOGD(context_->GetNodeName(), "[TilingData] : reduceMeanCof = %f", this->reduceMeanCof);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradTilingClass::DoScalarDagOpTiling()
{
    if (this->inputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<MseLossGradOp::MseLossGradScalarDag<half, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(this->reduceMeanCof);
        OP_CHECK_IF((brcBaseTiling.DoTiling() == ge::GRAPH_FAILED),
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    } else if (this->inputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<MseLossGradOp::MseLossGradScalarDag<bfloat16_t, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(this->reduceMeanCof);
        OP_CHECK_IF((brcBaseTiling.DoTiling() == ge::GRAPH_FAILED),
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<MseLossGradOp::MseLossGradScalarDag<float, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(this->reduceMeanCof);
        OP_CHECK_IF((brcBaseTiling.DoTiling() == ge::GRAPH_FAILED),
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    } else {
        OP_LOGE(context_->GetNodeName(), "input dtype is only support float16, bfloat16, float32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradTilingClass::DoTensorDagOpTiling()
{
    if (this->inputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<MseLossGradOp::MseLossGradDag<half, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(this->reduceMeanCof);
        OP_CHECK_IF((brcBaseTiling.DoTiling() == ge::GRAPH_FAILED),
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    } else if (this->inputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<MseLossGradOp::MseLossGradDag<bfloat16_t, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(this->reduceMeanCof);
        OP_CHECK_IF((brcBaseTiling.DoTiling() == ge::GRAPH_FAILED),
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<MseLossGradOp::MseLossGradDag<float, float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar(this->reduceMeanCof);
        OP_CHECK_IF((brcBaseTiling.DoTiling() == ge::GRAPH_FAILED),
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        this->tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->doutIsScalar);
    } else {
        OP_LOGE(context_->GetNodeName(), "input tensor dtype is only support float16, bfloat16, float32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradTilingClass::DoOpTiling()
{
    OP_CHECK_IF(CheckDoutIsScalar() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "check dout is scalar failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcReduceMeanCof() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "get reduceMeanCof failed"), return ge::GRAPH_FAILED);
    if (this->doutIsScalar == static_cast<uint32_t>(ATTR_IS_TRUE)) {
        OP_CHECK_IF(DoScalarDagOpTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "do tiling failed, when dout is scalar"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(DoTensorDagOpTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "do tiling failed, when dout is tensor"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradTilingClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MseLossGradTilingClass::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus MseLossGradTilingClass::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradTilingClass::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradTilingClass::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMseLossGrad(gert::TilingContext* context)
{
    OP_LOGD("MseLossGradTiling", "Enter TilingForMseLossGrad");
    if (context == nullptr) {
        OP_LOGE("MseLossGradTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Enter ascendc MseLossGradTiling");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus MseLossGradTilingPrepareAscendC(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<MseLossGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Get core num failed, core num: %lu", compileInfo->coreNum),
        return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);

    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "Get ub size failed, ub size: %lu", compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMseLossGrad(gert::TilingParseContext* context)
{
    OP_LOGD("TilingPrepareForMseLossGrad", "Enter TilingPrepareForMseLossGrad");
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForMseLossGrad", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Enter MseLossGradTilingPrepareAscendC");
    return MseLossGradTilingPrepareAscendC(context);
}

IMPL_OP_OPTILING(MseLossGrad)
    .Tiling(TilingForMseLossGrad)
    .TilingParse<MseLossGradCompileInfo>(TilingPrepareForMseLossGrad);

REGISTER_OPS_TILING_TEMPLATE(MseLossGrad, MseLossGradTilingClass, COMMON_TILING_PRIORITY);

} // namespace optiling