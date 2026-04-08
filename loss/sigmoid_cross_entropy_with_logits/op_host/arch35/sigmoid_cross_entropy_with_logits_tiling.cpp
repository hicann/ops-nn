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
 * \file sigmoid_cross_entropy_with_logits_tiling.cpp
 * \brief SigmoidCrossEntropyWithLogits 算子 Tiling 实现
 */
#include "sigmoid_cross_entropy_with_logits_tiling.h"
#include <graph/utils/type_utils.h>
#include "error_util.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_util.h"

using namespace ge;
using namespace SigmoidCrossEntropyWithLogitsOp;

namespace optiling {

const size_t ASCEND_WORKSPACE = 16777216;
const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsTiling::CalcInputDtype() {
    OP_LOGD(tilingContext_->GetNodeName(), "CalcInputDtype enter.");
    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    this->inputDtype_ = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype_ != ge::DT_FLOAT16 && this->inputDtype_ != ge::DT_BF16 && this->inputDtype_ != ge::DT_FLOAT,
        OP_LOGE(tilingContext_->GetNodeName(), "input predict dtype[%s] not support",
        ge::TypeUtils::DataTypeToSerialString(this->inputDtype_).c_str()),
        return ge::GRAPH_FAILED);
    
    auto targetDesc = tilingContext_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, targetDesc);
    auto targetDtype = targetDesc->GetDataType();
    OP_CHECK_IF(
        targetDtype != this->inputDtype_,
        OP_LOGE(tilingContext_->GetNodeName(), "input target dtype[%s] not same as predict[%s]",
        ge::TypeUtils::DataTypeToSerialString(targetDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(this->inputDtype_).c_str()),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsTiling::CalcOutputDtype() {
    OP_LOGD(tilingContext_->GetNodeName(), "CalcOutputDtype enter.");
    auto outputDesc = tilingContext_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputDesc);
    this->outputDtype_ = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype_ != this->inputDtype_,
        OP_LOGE(tilingContext_->GetNodeName(), "output loss dtype[%s] not same as input predict[%s]",
        ge::TypeUtils::DataTypeToSerialString(this->outputDtype_).c_str(),
        ge::TypeUtils::DataTypeToSerialString(this->inputDtype_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsTiling::CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto curStorageShape = EnsureNotScalar(inputShape->GetStorageShape());
    if (curStorageShape != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsTiling::CheckShape() {
    OP_LOGD(tilingContext_->GetNodeName(), "CheckShape enter.");
    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    const gert::Shape& inputStorageShape = EnsureNotScalar(inputShape->GetStorageShape());

    OP_CHECK_IF(
        CheckSameShape(1, inputStorageShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(tilingContext_->GetNodeName(), "the shape of input target is different from that of input predict."),
        return ge::GRAPH_FAILED);

    auto outputShape = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputShape);
    const gert::Shape& outputStorageShape = EnsureNotScalar(outputShape->GetStorageShape());
    OP_CHECK_IF(
        outputStorageShape != inputStorageShape,
        OP_LOGE(tilingContext_->GetNodeName(), "the shape of output loss is different from that of input predict."),
        return ge::GRAPH_FAILED);

    auto inputShapeSize = inputStorageShape.GetShapeSize();
    OP_CHECK_IF(
        inputShapeSize == 0,
        OP_LOGE(tilingContext_->GetNodeName(), "empty tensor is not supported, input predict shape size is 0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsTiling::DoElewiseTiling() {
    auto tiling = tilingContext_->GetTilingData<Ops::Base::EleBaseTilingData16B>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, tiling);
    
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    ge::graphStatus ret = ge::GRAPH_FAILED;

    if (this->inputDtype_ == ge::DT_FLOAT16) {
        dType_ = TPL_FP16;
        ret = eleBaseTiling.DoTiling<SigmoidCrossEntropyWithLogitsOp::SigmoidCrossEntropyWithLogitsDagWithCast<half>::OpDag>(*tiling);
    } else if (this->inputDtype_ == ge::DT_BF16) {
        dType_ = TPL_BF16;
        ret = eleBaseTiling.DoTiling<SigmoidCrossEntropyWithLogitsOp::SigmoidCrossEntropyWithLogitsDagWithCast<bfloat16_t>::OpDag>(*tiling);
    } else if (this->inputDtype_ == ge::DT_FLOAT) {
        dType_ = TPL_FP32;
        ret = eleBaseTiling.DoTiling<SigmoidCrossEntropyWithLogitsOp::SigmoidCrossEntropyWithLogitsDagNoCast<float>::OpDag>(*tiling);
    } else {
        OP_LOGE(tilingContext_->GetNodeName(), "input dtype[%s] not support!",
        ge::TypeUtils::DataTypeToSerialString(this->inputDtype_).c_str());
        return ge::GRAPH_FAILED;
    }
    
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(tilingContext_->GetNodeName(), "elewiseBaseTiling failed");
        return ge::GRAPH_FAILED;
    }
    tilingKey_ = GET_TPL_TILING_KEY(dType_);
    OP_LOGD(tilingContext_->GetNodeName(), "[TilingData] tilingKey=%lu, dType=%lu", tilingKey_, dType_);
    tilingContext_->SetTilingKey(tilingKey_);
    tilingContext_->SetBlockDim(eleBaseTiling.GetBlockDim());

    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SigmoidCrossEntropyWithLogitsTiling::RunTiling() {
    OP_LOGD(tilingContext_->GetNodeName(), "RunTiling enter.");
    if (tilingContext_ == nullptr) {
        OP_LOGE("SigmoidCrossEntropyWithLogits", "Get nullptr while obtaining tilingContext_.");
        return ge::GRAPH_FAILED;
    }
    
    OP_CHECK_IF(CalcInputDtype() != ge::GRAPH_SUCCESS, 
                OP_LOGE(tilingContext_->GetNodeName(), "CalcInputDtype failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(), "CalcOutputDtype failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(), "CheckShape failed."),
                return ge::GRAPH_FAILED);
    
    return DoElewiseTiling();
}

ge::graphStatus Tiling4SigmoidCrossEntropyWithLogits(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4SigmoidCrossEntropyWithLogits running begin");
    auto compileInfo = reinterpret_cast<const SigmoidCrossEntropyWithLogitsCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    
    SigmoidCrossEntropyWithLogitsTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForSigmoidCrossEntropyWithLogits(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForSigmoidCrossEntropyWithLogits running begin");
    auto compileInfoPtr = context->GetCompiledInfo<SigmoidCrossEntropyWithLogitsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SigmoidCrossEntropyWithLogits)
    .Tiling(Tiling4SigmoidCrossEntropyWithLogits)
    .TilingParse<SigmoidCrossEntropyWithLogitsCompileInfo>(TilingPrepareForSigmoidCrossEntropyWithLogits);

}  // namespace optiling
