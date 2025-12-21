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
 * \file apply_adam_w_v2_tiling_arch35.cpp
 * \brief
 */
#include "apply_adam_w_v2_tiling_arch35.h"

#include <graph/utils/type_utils.h>

#include <iostream>

#include "error_util.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace ApplyAdamWV2RegbaseOp;

namespace optiling {
const size_t SYS_WORKSPACE = 16777216;  // 16M
constexpr static int32_t OPTIONAL_INPUT_INDEX = 11;
constexpr static int32_t SCALAR_INPUT_INDEX = 4;
constexpr static int32_t INPUT_GRAD_INDEX = 3;
constexpr static float ATTR_INIT_VALUE = 0.1;
constexpr static float EPS_INIT_VALUE = 1e-8;
const static std::map<size_t, string> TENSOR_INDEX_LIST = {{1, "m"}, {2, "v"}};
void ApplyAdamWV2RegbaseTiling::PrintInfo() {
    auto nodeName = tilingContext_->GetNodeName();
    OP_LOGD(nodeName, "Start to print ApplyAdamWV2RegbaseTiling custom tiling data");
    OP_LOGD(nodeName, "lr: %f", tiling_->lr);
    OP_LOGD(nodeName, "beta1: %f", tiling_->beta1);
    OP_LOGD(nodeName, "beta2: %f", tiling_->beta2);
    OP_LOGD(nodeName, "weightDecay: %f", tiling_->weightDecay);
    OP_LOGD(nodeName, "eps: %f", tiling_->eps);
    OP_LOGD(nodeName, "gt: %f", tiling_->gt);
    OP_LOGD(nodeName, "Print ApplyAdamWV2RegbaseTiling custom tiling data end");
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::SetTilingData() {
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    tiling_->lr = lrAttr_;
    tiling_->beta1 = beta1Attr_;
    tiling_->beta2 = beta2Attr_;
    tiling_->weightDecay = weightDecayAttr_;
    tiling_->eps = epsAttr_;
    if (maximizeAttr_) {
        tiling_->gt = -1.0f;
    } else {
        tiling_->gt = 1.0f;
    }
    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE;
    PrintInfo();
    tilingKey = GET_TPL_TILING_KEY(schMode, amsgrad);
    tilingContext_->SetTilingKey(tilingKey);
    uint32_t blockDim = static_cast<uint32_t>(tiling_->elewiseTiling.blockNum);
    OP_CHECK_IF(blockDim <= 0, OP_LOGE(tilingContext_, "Get blockDim failed"),
                return ge::GRAPH_FAILED);
    tilingContext_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::CheckScalarInput() {
    size_t index = SCALAR_INPUT_INDEX;
    auto inputShape = tilingContext_->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    OP_CHECK_IF((!storageShape.IsScalar() && storageShape.GetShapeSize() != 1),
                OP_LOGE(tilingContext_, "input step must be scalar."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::CheckSameShape(size_t inputIdx, const gert::Shape& input0Shape) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto curStorageShape = inputShape->GetStorageShape();
    if (curStorageShape != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::CheckSameDtype(size_t inputIdx, ge::DataType& input0Dtype) {
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto curDtype = inputDesc->GetDataType();
    if (curDtype != input0Dtype) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::CheckMixAndOptionalInput(const gert::Shape& inputStorageShape) {
    auto inputGradShape = tilingContext_->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputGradShape);
    auto gradStorageShape = inputGradShape->GetStorageShape();

    OP_CHECK_IF(gradStorageShape != inputStorageShape,
                OP_LOGE(tilingContext_, "the shape of input grad is different from that of input var."),
                return ge::GRAPH_FAILED);
    auto inputGradDesc = tilingContext_->GetInputDesc(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputGradDesc);
    auto gradDtype = inputGradDesc->GetDataType();
    // check optional input
    auto maxGradNormDesc = tilingContext_->GetOptionalInputDesc(OPTIONAL_INPUT_INDEX);
    if (maxGradNormDesc != nullptr) {
        auto maxGradNormDtype = maxGradNormDesc->GetDataType();
        OP_CHECK_IF(maxGradNormDtype != gradDtype,
                    OP_LOGE(tilingContext_, "Optinal input max_grad_norm dtype not match with input grad dtype."),
                    return ge::GRAPH_FAILED);
    }
    auto maxGradNormShape = tilingContext_->GetOptionalInputShape(OPTIONAL_INPUT_INDEX);
    if (maxGradNormShape != nullptr) {
        auto maxGradNormStorageShape = maxGradNormShape->GetStorageShape();
        OP_CHECK_IF(
            maxGradNormStorageShape != gradStorageShape,
            OP_LOGE(tilingContext_, "the shape of optinal input max_grad_norm is different from that of input grad."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::CheckShapeAndType() {
    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto inputStorageShape = inputShape->GetStorageShape();

    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto inputDtype = inputDesc->GetDataType();
    // check scalar input
    OP_CHECK_IF(CheckScalarInput() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "CheckScalarInput failed."),
                return ge::GRAPH_FAILED);
    // check tensor input
    for (const auto& pair : TENSOR_INDEX_LIST) {
        OP_CHECK_IF(
            CheckSameShape(pair.first, inputStorageShape) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext_, "the shape of input %s is different from that of input var.", pair.second.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckSameDtype(pair.first, inputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext_, "the dtype of input %s is different from that of input var.", pair.second.c_str()),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckMixAndOptionalInput(inputStorageShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_, "CheckMixDtypeAndOptionalInput failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::DoAmsGradTiling(ElewiseBaseTiling& eleBaseTiling, ge::DataType& varDType,
                                                           ge::DataType& gradDType, ge::DataType& stepDType) {
    ge::graphStatus ret = ge::GRAPH_FAILED;
    amsgrad = static_cast<uint64_t>(ATTR_IS_TRUE);
    if (varDType != ge::DT_FLOAT && stepDType == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2AmsGradDAG<half, half, float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType != ge::DT_FLOAT && stepDType == ge::DT_INT64) {
        ret =
            eleBaseTiling.DoTiling<ApplyAdamWV2AmsGradDAG<half, half, int64_t, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType == ge::DT_FLOAT && stepDType == ge::DT_FLOAT) {
        ret =
            eleBaseTiling.DoTiling<ApplyAdamWV2AmsGradDAG<float, float, float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType == ge::DT_FLOAT && stepDType == ge::DT_INT64) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2AmsGradDAG<float, float, int64_t, float>::OpDag>(
            tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType != ge::DT_FLOAT && stepDType == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2AmsGradDAG<float, half, float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType != ge::DT_FLOAT && stepDType == ge::DT_INT64) {
        ret =
            eleBaseTiling.DoTiling<ApplyAdamWV2AmsGradDAG<float, half, int64_t, float>::OpDag>(tiling_->elewiseTiling);
    } else {
        OP_LOGE(tilingContext_, "input dtype is not support!");
        ret = ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::DoNormTiling(ElewiseBaseTiling& eleBaseTiling, ge::DataType& varDType,
                                                        ge::DataType& gradDType, ge::DataType& stepDType) {
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (varDType != ge::DT_FLOAT && stepDType == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2DAG<half, half, float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType != ge::DT_FLOAT && stepDType == ge::DT_INT64) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2DAG<half, half, int64_t, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType == ge::DT_FLOAT && stepDType == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2DAG<float, float, float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType == ge::DT_FLOAT && stepDType == ge::DT_INT64) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2DAG<float, float, int64_t, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType != ge::DT_FLOAT && stepDType == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2DAG<float, half, float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT && gradDType != ge::DT_FLOAT && stepDType == ge::DT_INT64) {
        ret = eleBaseTiling.DoTiling<ApplyAdamWV2DAG<float, half, int64_t, float>::OpDag>(tiling_->elewiseTiling);
    } else {
        OP_LOGE(tilingContext_, "input dtype is not support!");
        ret = ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::DoElewiseTiling() {
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    auto varDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varDesc);
    auto gradDesc = tilingContext_->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, gradDesc);
    auto stepDesc = tilingContext_->GetInputDesc(4);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, stepDesc);

    ge::DataType varDType = varDesc->GetDataType();
    ge::DataType gradDType = gradDesc->GetDataType();
    ge::DataType stepDType = stepDesc->GetDataType();
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (amsgradAttr_) {
        ret = DoAmsGradTiling(eleBaseTiling, varDType, gradDType, stepDType);
    } else {
        ret = DoNormTiling(eleBaseTiling, varDType, gradDType, stepDType);
    }
    return ret;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::GetAttributes() {
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const float* lr = attrs->GetAttrPointer<float>(0);
    lrAttr_ = lr != nullptr ? *lr : ATTR_INIT_VALUE;

    const float* beta1 = attrs->GetAttrPointer<float>(1);
    beta1Attr_ = beta1 != nullptr ? *beta1 : ATTR_INIT_VALUE;

    const float* beta2 = attrs->GetAttrPointer<float>(2);
    beta2Attr_ = beta2 != nullptr ? *beta2 : ATTR_INIT_VALUE;

    const float* weightDecay = attrs->GetAttrPointer<float>(3);
    weightDecayAttr_ = weightDecay != nullptr ? *weightDecay : ATTR_INIT_VALUE;

    const float* eps = attrs->GetAttrPointer<float>(4);
    epsAttr_ = eps != nullptr ? *eps : EPS_INIT_VALUE;

    const bool* amsgradAttr = attrs->GetAttrPointer<bool>(5);
    amsgradAttr_ = amsgradAttr != nullptr ? *amsgradAttr : false;

    const bool* maximizeAttr = attrs->GetAttrPointer<bool>(6);
    maximizeAttr_ = maximizeAttr != nullptr ? *maximizeAttr : false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWV2RegbaseTiling::RunTiling() {
    if (tilingContext_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckShapeAndType() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "Shape and Dtype check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttributes() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "GetAttributes failed"),
                return ge::GRAPH_FAILED);

    tiling_ = tilingContext_->GetTilingData<ApplyAdamWV2RegbaseTilingData>();
    OP_CHECK_IF((tiling_ == nullptr), OP_LOGE(tilingContext_, "Get ApplyAdamWV2RegbaseTiling from GE context failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoElewiseTiling() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext_, "elewiseBaseTiling failed"),
                return ge::GRAPH_FAILED);
    return SetTilingData();
}
}  // namespace optiling