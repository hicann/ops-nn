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
 * \file apply_adagrad_d_tiling_arch35.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <iostream>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "optim/apply_adagrad_d/op_kernel/arch35/apply_adagrad_d_dag.h"
#include "optim/apply_adagrad_d/op_kernel/arch35/apply_adagrad_d_tiling_key.h"
#include "apply_adagrad_d_tiling_arch35.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace ge;
using namespace ApplyAdagradDTilingData;

namespace optiling {
const size_t ASCEND_WORKSPACE = 16777216; // 16MB
const int32_t VAR_INDEX = 0;
const int32_t ACCUM_INDEX = 1;
const int32_t LR_INDEX = 2; 
const int32_t GRAD_INDEX = 3;
const int32_t INPUT_NUM = 4;
const int32_t OUTPUT_NUM = 2;

const gert::Shape g_vec_1_shape = {1};
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape,
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus ApplyAdagradDTiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    schMode = tiling_->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, updateSlots, dType);
    OP_LOGD(tilingContext_->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext_->SetTilingKey(tilingKey);
    tilingContext_->SetBlockDim(tiling_->baseTiling.blockNum);

    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);

    size_t *currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradDTiling::CheckDtype() {
    auto varDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varDesc);

    this->varDtype_ = varDesc->GetDataType();

    for (int32_t inputIdx = ACCUM_INDEX; inputIdx < INPUT_NUM; inputIdx++) {
        auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);

        auto curDtype = inputDesc->GetDataType();
        OP_CHECK_IF(curDtype != varDtype_,
                        OP_LOGE(tilingContext_->GetNodeName(), "Input %d dtype not match with var dtype.", inputIdx),
                        return ge::GRAPH_FAILED);
    }

    for (int32_t outputIdx = 0; outputIdx < OUTPUT_NUM; outputIdx++) {
        auto outputDesc = tilingContext_->GetInputDesc(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputDesc);

        auto curDtype = outputDesc->GetDataType();
        OP_CHECK_IF(curDtype != varDtype_,
                        OP_LOGE(tilingContext_->GetNodeName(), "Output %d dtype not match with var dtype.", outputIdx),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

bool ApplyAdagradDTiling::CheckIsScalar(int32_t inputIdx) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    if (storageShape.IsScalar() || storageShape.GetShapeSize() == 1) {
        return true;
    }
    return false;
}

ge::graphStatus ApplyAdagradDTiling::CheckShape() {
    OP_CHECK_IF(!CheckIsScalar(LR_INDEX),
                    OP_LOGE(tilingContext_->GetNodeName(), "Input lr must be a scalar."),
                    return ge::GRAPH_FAILED);
    auto varStorageShape  = tilingContext_->GetInputShape(VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varStorageShape);
    auto accumStorageShape = tilingContext_->GetInputShape(ACCUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, accumStorageShape);
    auto gradStorageShape = tilingContext_->GetInputShape(GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, gradStorageShape);
    const gert::Shape& varShape = EnsureNotScalar(varStorageShape->GetStorageShape());
    const gert::Shape& accumShape = EnsureNotScalar(accumStorageShape->GetStorageShape());
    const gert::Shape& gradShape = EnsureNotScalar(gradStorageShape->GetStorageShape());

    OP_CHECK_IF(varShape != accumShape || varShape != gradShape,
                    OP_LOGE(tilingContext_->GetNodeName(), "shape of var, accum and grad are not same"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradDTiling::DoUpdateSlotsTiling() {
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    updateSlots = static_cast<uint64_t>(UPDATE_SLOTS_TPL_TRUE);
    if (this->varDtype_ == ge::DT_FLOAT16) {
        dType = static_cast<uint64_t>(APPLY_ADAGRAD_D_TPL_FP16);
        OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdagradDOp::ApplyAdagradDUpdateSlots<half>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed for fp16/bf16 with updates_slots"),
                        return ge::GRAPH_FAILED);
    } else if (this->varDtype_ == ge::DT_BF16) {
        dType = static_cast<uint64_t>(APPLY_ADAGRAD_D_TPL_BF16);
        OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdagradDOp::ApplyAdagradDUpdateSlots<bfloat16_t>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed for fp16/bf16 with updates_slots"),
                        return ge::GRAPH_FAILED);
    } else if (this->varDtype_ == ge::DT_FLOAT) {
        dType = static_cast<uint64_t>(APPLY_ADAGRAD_D_TPL_FP32);
        OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdagradDOp::ApplyAdagradDUpdateSlots<float>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed for fp32 with updates_slots"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext_->GetNodeName(), "Current dtype not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradDTiling::DoNonUpdateSlotsTiling() {
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    updateSlots = static_cast<uint64_t>(UPDATE_SLOTS_TPL_FALSE);
    if (this->varDtype_ == ge::DT_FLOAT16) {
        dType = static_cast<uint64_t>(APPLY_ADAGRAD_D_TPL_FP16);
        OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdagradDOp::ApplyAdagradD<half>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed for fp16."),
                        return ge::GRAPH_FAILED);
    } else if (this->varDtype_ == ge::DT_BF16) {
        dType = static_cast<uint64_t>(APPLY_ADAGRAD_D_TPL_BF16);
        OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdagradDOp::ApplyAdagradD<bfloat16_t>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed for bf16."),
                        return ge::GRAPH_FAILED);
    } else if (this->varDtype_ == ge::DT_FLOAT) {
        dType = static_cast<uint64_t>(APPLY_ADAGRAD_D_TPL_FP32);
        OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdagradDOp::ApplyAdagradD<float>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed for fp32."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext_->GetNodeName(), "Current dtype not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdagradDTiling::RunTiling() {
    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "Dtype check failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "Shape check failed."),
                    return ge::GRAPH_FAILED);

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const bool* updateSlotsAttr = attrs->GetAttrPointer<bool>(0);
    updateSlots_ = updateSlotsAttr != nullptr ? *updateSlotsAttr : true;
    tiling_ = tilingContext_->GetTilingData<ApplyAdagradDTilingDataStruct>();
    OP_CHECK_IF((tiling_ == nullptr), OP_LOGE(tilingContext_, "Get EleBaseTilingData from context failed"), return ge::GRAPH_FAILED);
    if (updateSlots_) {
        OP_CHECK_IF(DoUpdateSlotsTiling() != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(DoNonUpdateSlotsTiling() != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext_->GetNodeName(), "Do tiling failed."),
                        return ge::GRAPH_FAILED);
    }
    return SetTilingData();
}

static ge::graphStatus TilingForApplyAdagradD(gert::TilingContext *context)
{
    OP_LOGD("ApplyAdagradDTiling", "Enter TilingForApplyAdagradD");
    if (context == nullptr) {
        OP_LOGE("ApplyAdagradDTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const ApplyAdagradDCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("ApplyAdagradDTiling", "Enter ApplyAdagradD Elewise DAG Tiling");
    ApplyAdagradDTiling tiling(context);
    return tiling.RunTiling();
}

static ge::graphStatus TilingPrepareForElewiseApplyAdagradD(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ApplyAdagradDCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(ApplyAdagradD).Tiling(TilingForApplyAdagradD).TilingParse<ApplyAdagradDCompileInfo>(TilingPrepareForElewiseApplyAdagradD);
}