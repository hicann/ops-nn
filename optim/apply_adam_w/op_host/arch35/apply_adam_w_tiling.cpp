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
 * \file apply_adam_w_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "../op_kernel/arch35/apply_adam_w_dag.h"
#include "apply_adam_w_tiling.h"

using namespace ge;
using namespace ApplyAdamWOp;
using namespace ApplyAdamWNs;

namespace optiling {
const size_t SYS_WORKSPACE = 16777216;  // 16M
constexpr static int32_t OPTIONAL_INPUT_INDEX = 11;
const static std::map<size_t, string> TENSOR_INDEX_LIST = {{1, "m"}, {2, "v"}, {10, "grad"}};
const static std::map<size_t, string> SCALAR_INDEX_LIST = {{3, "beta1_power"},  {4, "beta2_power"}, {5, "lr"},
                                                           {6, "weight_decay"}, {7, "beta1"},       {8, "beta2"},
                                                           {9, "epsilon"}};

ge::graphStatus ApplyAdamWTiling::SetTilingData() {
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    if (maximizeAttr_) {
        tiling_->maximizeFactor = static_cast<float>(-1.0);
    } else {
        tiling_->maximizeFactor = static_cast<float>(1.0);
    }
    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);

    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE;

    tilingKey = GET_TPL_TILING_KEY(schMode, amsgrad, dType);
    tilingContext_->SetTilingKey(tilingKey);
    tilingContext_->SetBlockDim(tiling_->eleBaseTilingData.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::CheckIsScalar(size_t inputIdx) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    if (storageShape.IsScalar() || storageShape.GetShapeSize() == 1) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus ApplyAdamWTiling::CheckSameShape(size_t inputIdx, const gert::Shape& input0Shape) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto curStorageShape = inputShape->GetStorageShape();
    if (curStorageShape != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::CheckSameDtype(size_t inputIdx, const ge::DataType& input0Dtype) {
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto curDtype = inputDesc->GetDataType();
    if (curDtype != input0Dtype) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::CheckShapeAndType() {
    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto inputStorageShape = inputShape->GetStorageShape();

    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto inputDtype = inputDesc->GetDataType();
    for (const auto& pair : SCALAR_INDEX_LIST) {
        OP_CHECK_IF(CheckIsScalar(pair.first) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Input %s must be scalar.",
                                                        pair.second.c_str()),
                        return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckSameDtype(pair.first, inputDtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(),
                                                        "the shape of input %s is different from that of input var.",
                                                        pair.second.c_str()),
                        return ge::GRAPH_FAILED);
    }
    for (const auto& pair : TENSOR_INDEX_LIST) {
        OP_CHECK_IF(CheckSameShape(pair.first, inputStorageShape) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(),
                                                        "the shape of input %s is different from that of input var.",
                                                        pair.second.c_str()),
                        return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckSameDtype(pair.first, inputDtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(),
                                                        "the shape of input %s is different from that of input var.",
                                                        pair.second.c_str()),
                        return ge::GRAPH_FAILED);
    }

    auto maxGradNormDesc = tilingContext_->GetOptionalInputDesc(OPTIONAL_INPUT_INDEX);
    if (maxGradNormDesc != nullptr) {
        auto maxGradNormDtype = maxGradNormDesc->GetDataType();
        OP_CHECK_IF(
            maxGradNormDtype != inputDtype,
            OP_LOGE(tilingContext_->GetNodeName(),
                                            "Optinal input max_grad_norm dtype not match with input var dtype."),
            return ge::GRAPH_FAILED);
    }
    
    auto maxGradNormShape = tilingContext_->GetOptionalInputShape(OPTIONAL_INPUT_INDEX);
    if (maxGradNormShape != nullptr) {
        auto maxGradNormStorageShape = maxGradNormShape->GetStorageShape();
        OP_CHECK_IF(
            maxGradNormStorageShape != inputStorageShape,
            OP_LOGE(tilingContext_->GetNodeName(),
                                            "Optinal input max_grad_norm dtype not match with input var dtype."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::DoElewiseTiling() {
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    auto input0Desc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (amsgradAttr_) {
        amsgrad = ATTR_IS_TRUE;
        if (input0DType == ge::DT_FLOAT16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWAmsGradDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP16;
        } else if (input0DType == ge::DT_BF16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWAmsGradDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_BF16;
        } else if (input0DType == ge::DT_FLOAT) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWAmsGradDAG<float, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP32;
        } else {
            OP_LOGE(tilingContext_->GetNodeName(),
                                            "input dtype is only support fp16, bf16, fp32, while got %s!",
                                            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str());
            ret = ge::GRAPH_FAILED;
        }
    } else {
        if (input0DType == ge::DT_FLOAT16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP16;
        } else if (input0DType == ge::DT_BF16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_BF16;
        } else if (input0DType == ge::DT_FLOAT) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWDAG<float, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP32;
        } else {
            OP_LOGE(tilingContext_->GetNodeName(),
                                            "input dtype is only support fp16, bf16, fp32, while got %s!",
                                            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str());
            ret = ge::GRAPH_FAILED;
        }
    }
    return ret;
}

ge::graphStatus ApplyAdamWTiling::RunTiling() {
    if (tilingContext_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "Shape and Dtype check failed."),
                    return ge::GRAPH_FAILED);
    ElewiseBaseTiling elewiseBaseTiling(tilingContext_);
    tiling_ = tilingContext_->GetTilingData<ApplyAdamWTilingData>();
    OP_CHECK_IF((tiling_ == nullptr), OP_LOGE(tilingContext_, "Get EleBaseTilingData from context failed"), return ge::GRAPH_FAILED);
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const bool* amsgradAttr = attrs->GetAttrPointer<bool>(0);
    amsgradAttr_ = amsgradAttr != nullptr ? *amsgradAttr : false;

    const bool* maximizeAttr = attrs->GetAttrPointer<bool>(1);
    maximizeAttr_ = maximizeAttr != nullptr ? *maximizeAttr : false;
    OP_CHECK_IF(DoElewiseTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_, "elewiseBaseTiling failed"),
                    return ge::GRAPH_FAILED);
    return SetTilingData();
}

static ge::graphStatus TilingForApplyAdamW(gert::TilingContext* context) {
    if (context == nullptr) {
        OP_LOGE("ApplyAdamWTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD("ApplyAdamWTiling", "Enter TilingForApplyAdamW");

    auto compileInfo = reinterpret_cast<const ApplyAdamWCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD("ApplyAdamWTiling", "Enter new ApplyAdamWTiling");
    ApplyAdamWTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForApplyAdamW(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ApplyAdamWCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdamW).Tiling(TilingForApplyAdamW).TilingParse<ApplyAdamWCompileInfo>(TilingPrepareForApplyAdamW);
}  // namespace optiling