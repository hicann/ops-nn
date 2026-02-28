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
 * \file l1_loss_grad_tiling.cpp
 * \brief l1_loss_grad tiling source file
 */
#include <graph/utils/type_utils.h>
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/l1_loss_grad_dag.h"
#include "l1_loss_grad_tiling_base.h"
#include "l1_loss_grad_tiling.h"
#include "op_host/tiling_templates_registry.h"

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace {
const size_t INDEX_PREDICT = 0;
}  // namespace

namespace optiling {
using namespace L1LossGradKernel;

const size_t ASCEND_WORKSPACE = 16777216; // 16MB
constexpr static uint64_t COMMON_TILING_PRIORITY = 0;
static const size_t INPUT_PREDICT_INDEX = 1;
static const size_t INPUT_GRADS_INDEX = 0;
static const size_t INPUT_LABEL_INDEX = 2;
static const size_t OUTPUT_Y_INDEX = 0;
static const size_t REDUCTION_INDEX = 0;

ge::graphStatus L1LossGradTiling::GetShapeAttrsInfo()
{
    auto inputGradsDesc = context_->GetInputDesc(INPUT_GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGradsDesc);
    auto inputPredictDesc = context_->GetInputDesc(INPUT_PREDICT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputPredictDesc);
    auto inputLabelDesc = context_->GetInputDesc(INPUT_LABEL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLabelDesc);
    auto outputDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);

    ge::DataType inputGradsDtype = inputGradsDesc->GetDataType();
    ge::DataType inputPredictDtype = inputPredictDesc->GetDataType();
    ge::DataType inputLabelDtype = inputLabelDesc->GetDataType();
    ge::DataType outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(inputGradsDtype != inputPredictDtype,
        OP_LOGE(context_->GetNodeName(), "Input grads dtype not match with input predict dtype."),
        return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(inputGradsDtype != inputLabelDtype,
        OP_LOGE(context_->GetNodeName(),  "Input grads dtype not match with input label dtype."),
        return ge::GRAPH_FAILED);
   
    OP_CHECK_IF(inputPredictDtype != ge::DT_FLOAT16 && inputPredictDtype != ge::DT_BF16 && inputPredictDtype != ge::DT_FLOAT,
        OP_LOGE(context_->GetNodeName(),  "input predict dtype not support"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputDtype != inputGradsDtype ,
        OP_LOGE(context_->GetNodeName(),  "output y dtype not same as inputs"),
        return ge::GRAPH_FAILED);

    this->outputDtype_ = outputDtype;
    return ge::GRAPH_SUCCESS;
}

bool L1LossGradTiling::IsCapable()
{
    return true;
}

inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus L1LossGradTiling::CaluateReduceElts()
{
    float reduceEltsTemp = 1.0;

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const char* reductionAttr = attrs->GetAttrPointer<char>(REDUCTION_INDEX);
    const char* reduction = reductionAttr != nullptr ? reductionAttr : "mean";
    
    OP_CHECK_IF(strcmp(reduction, "mean") != 0 && strcmp(reduction, "sum") != 0 && strcmp(reduction, "none") != 0,
               OP_LOGE(context_, "reduction is not in none, mean or sum"),
               return ge::GRAPH_FAILED);

    if (strcmp(reduction, "mean") == 0) {
        auto labelStorageShape = context_->GetInputShape(INPUT_LABEL_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, labelStorageShape);
        const gert::Shape& inputLabelShape = EnsureNotScalar(labelStorageShape->GetStorageShape());
        const size_t dimLen = inputLabelShape.GetDimNum();
        for (uint32_t i = 0; i < dimLen; i++) {
            if (inputLabelShape.GetDim(i) != 0) {
                reduceEltsTemp = reduceEltsTemp / inputLabelShape.GetDim(i);
                this->reduceElts_ = reduceEltsTemp;
            } else {
                OP_LOGE(context_->GetNodeName(), "the shape[%u] is 0, do not supported", i);
                return ge::GRAPH_FAILED;
            }
        }
    }
    OP_LOGD(context_->GetNodeName(), "[TilingData] : reduceElts=%f", reduceEltsTemp);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus L1LossGradTiling::CheckGradsIsScalar() {
    auto gradsStorageShape = context_->GetInputShape(INPUT_GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradsStorageShape);
    const gert::Shape& inputGradsShape = EnsureNotScalar(gradsStorageShape->GetStorageShape());
    if (inputGradsShape.GetDimNum() == 1 && inputGradsShape.GetDim(0) == 1) {
        this->inputGradsIsScalar_ = static_cast<uint32_t>(ATTR_IS_TRUE);
        OP_LOGD(context_->GetNodeName(), "input grads is sclar");
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(context_->GetNodeName(), "input grads is tensor");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus L1LossGradTiling::DoTiling() {
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->inputGradsIsScalar_ == static_cast<uint32_t>(ATTR_IS_TRUE)) {
        if (this->outputDtype_ == ge::DT_FLOAT16) {
            BroadcastBaseTiling<L1LossGradScalarCast<half>::OpDag> brcBaseTiling(context_,
                static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
            brcBaseTiling.SetScalar(this->reduceElts_);
            baseTilingResult = brcBaseTiling.DoTiling();
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->inputGradsIsScalar_);
        } else if (this->outputDtype_ == ge::DT_BF16) {
            BroadcastBaseTiling<L1LossGradScalarCast<bfloat16_t>::OpDag> brcBaseTiling(context_,     // bfloat16类型host没定义
                static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
            brcBaseTiling.SetScalar(this->reduceElts_);
            baseTilingResult = brcBaseTiling.DoTiling();
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->inputGradsIsScalar_);
        } else {
            BroadcastBaseTiling<L1LossGradScalarCast<float>::OpDag> brcBaseTiling(context_, 
                static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
            brcBaseTiling.SetScalar(this->reduceElts_);
            baseTilingResult = brcBaseTiling.DoTiling();
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->inputGradsIsScalar_);
        } 
    } else {
        if (this->outputDtype_ == ge::DT_FLOAT16) {
            BroadcastBaseTiling<L1LossGradCast<half>::OpDag> brcBaseTiling(context_,     
                static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
            brcBaseTiling.SetScalar(this->reduceElts_);
            baseTilingResult = brcBaseTiling.DoTiling();
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->inputGradsIsScalar_);
        } else if (this->outputDtype_ == ge::DT_BF16) {
            BroadcastBaseTiling<L1LossGradCast<bfloat16_t>::OpDag> brcBaseTiling(context_,     // bfloat16类型host没定义
                static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
            brcBaseTiling.SetScalar(this->reduceElts_);
            baseTilingResult = brcBaseTiling.DoTiling();
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->inputGradsIsScalar_);
        } else {
            BroadcastBaseTiling<L1LossGradCast<float>::OpDag> brcBaseTiling(context_,    
                static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
            brcBaseTiling.SetScalar(this->reduceElts_);
            baseTilingResult = brcBaseTiling.DoTiling();
            this->tilingKey_ = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), this->inputGradsIsScalar_);
        } 
    }
    return baseTilingResult;
}

ge::graphStatus L1LossGradTiling::DoOpTiling() 
{
    OP_LOGD(context_->GetNodeName(), "L1LossGradTiling RunTiling enter.");
    OP_CHECK_IF(CheckGradsIsScalar() == ge::GRAPH_FAILED,
         OP_LOGE(context_->GetNodeName(), "check inputGrads is scalar failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CaluateReduceElts() == ge::GRAPH_FAILED,
         OP_LOGE(context_->GetNodeName(), "get CaluateReduceElts failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_, "DoTiling failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

 ge::graphStatus L1LossGradTiling::DoLibApiTiling()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 uint64_t L1LossGradTiling::GetTilingKey() const
 {
     return tilingKey_;
 }
 
 ge::graphStatus L1LossGradTiling::GetWorkspaceSize()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus L1LossGradTiling::PostTiling()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus L1LossGradTiling::GetPlatformInfo()
 {
     return ge::GRAPH_SUCCESS;
 }

static ge::graphStatus TilingPrepareForL1LossGrad(gert::TilingParseContext* context) {
    (void) context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4L1LossGrad(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4L1LossGrad rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const L1LossGradCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    OP_LOGD(tilingContextGen, "Enter ascendc L1LossGradTiling");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(tilingContextGen);
}
// register tiling interface of the L1LossGrad.
IMPL_OP_OPTILING(L1LossGrad)
    .Tiling(Tiling4L1LossGrad)
    .TilingParse<L1LossGradCompileInfo>(TilingPrepareForL1LossGrad);

REGISTER_OPS_TILING_TEMPLATE(L1LossGrad, L1LossGradTiling, COMMON_TILING_PRIORITY);
}  // namespace optiling