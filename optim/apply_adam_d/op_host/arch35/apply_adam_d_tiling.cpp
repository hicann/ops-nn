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
 * \file apply_adam_d_tiling.cpp
 * \brief
 */
#include "apply_adam_d_tiling.h"
#include <graph/utils/type_utils.h>
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/apply_adam_d_dag.h"
#include "../../op_kernel/arch35/apply_adam_d_tiling_struct.h"

#include <iostream>

using namespace ge;

namespace optiling {
const uint64_t SYS_WORKSPACE = 32; // 16M

constexpr int32_t VAR_INDEX = 0;
constexpr int32_t M_INDEX = 1;
constexpr int32_t V_INDEX = 2; 
constexpr int32_t BETA1_POWER_INDEX = 3;
constexpr int32_t BETA2_POWER_INDEX = 4;
constexpr int32_t LR_INDEX = 5; 
constexpr int32_t BETA1_INDEX = 6;
constexpr int32_t BETA2_INDEX = 7;
constexpr int32_t EPSILON_INDEX = 8;
constexpr int32_t GRAD_INDEX = 9;
constexpr int32_t INPUT_NUM = 10;
constexpr int32_t VAR_OUT_INDEX = 0;
constexpr int32_t M_OUT_INDEX = 1;
constexpr int32_t V_OUT_INDEX = 2;
constexpr int32_t OUTPUT_NUM = 3;

class ApplyAdamDTiling {
public:
    explicit ApplyAdamDTiling(gert::TilingContext *context) : tilingContext_(context) {};

    ge::graphStatus RunTiling();
    ApplyAdamDTilingData* tiling_ = nullptr;

protected:
    ge::graphStatus SetTilingData();
    bool CheckIsScalar(int32_t inputIdx);
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtype();

private:
    ge::DataType varDtype_;
    gert::TilingContext *tilingContext_;

    bool useLocking_;
    bool useNesterov_;
};

ge::graphStatus ApplyAdamDTiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");

    tiling_->useLocking = useLocking_;
    tiling_->useNesterov = useNesterov_;

    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);

    size_t *currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORKSPACE;

    if (this->varDtype_ == ge::DT_FLOAT16) {
        tilingContext_->SetTilingKey(101UL);
    } else if (this->varDtype_ == ge::DT_BF16) {
        tilingContext_->SetTilingKey(102UL);
    } else {
        tilingContext_->SetTilingKey(103UL);
    }

    tilingContext_->SetBlockDim(tiling_->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamDTiling::CheckDtype() {
    auto varDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, varDesc);

    this->varDtype_ = varDesc->GetDataType();

    for (int32_t inputIdx = M_INDEX; inputIdx < INPUT_NUM; inputIdx++) {
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

bool ApplyAdamDTiling::CheckIsScalar(int32_t inputIdx) {
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    if (storageShape.IsScalar() || storageShape.GetShapeSize() == 1) {
        return true;
    }
    return false;
}

ge::graphStatus ApplyAdamDTiling::CheckShape() {
    OP_CHECK_IF(!CheckIsScalar(BETA1_POWER_INDEX),
                    OP_LOGE(tilingContext_->GetNodeName(), "Input beta1_power must be scalar."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckIsScalar(BETA2_POWER_INDEX),
                    OP_LOGE(tilingContext_->GetNodeName(), "Input beta2_power must be scalar."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckIsScalar(LR_INDEX),
                    OP_LOGE(tilingContext_->GetNodeName(), "Input lr must be scalar."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckIsScalar(BETA1_INDEX),
                    OP_LOGE(tilingContext_->GetNodeName(), "Input beta1 must be scalar."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckIsScalar(BETA2_INDEX),
                    OP_LOGE(tilingContext_->GetNodeName(), "Input beta2 must be scalar."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckIsScalar(EPSILON_INDEX),
                    OP_LOGE(tilingContext_->GetNodeName(), "Input epsilon must be scalar."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamDTiling::RunTiling() {
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    tiling_ = tilingContext_->GetTilingData<ApplyAdamDTilingData>();

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "Dtype check failed."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "Shape check failed."),
                    return ge::GRAPH_FAILED);

    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const bool* useLockingAttr = attrs->GetAttrPointer<bool>(0);
    useLocking_ = useLockingAttr != nullptr ? *useLockingAttr : false;

    const bool* useNesterovAttr = attrs->GetAttrPointer<bool>(1);
    useNesterov_ = useNesterovAttr != nullptr ? *useNesterovAttr : false;

    if (useNesterov_) {
        if (this->varDtype_ == ge::DT_FLOAT16 || this->varDtype_ == ge::DT_BF16) {
            OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdamDDagFusionNesterov<half>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                            OP_LOGE(tilingContext_->GetNodeName(), "do tiling failed for fp16/bf16 with nesterov"),
                            return ge::GRAPH_FAILED);
        } else if (this->varDtype_ == ge::DT_FLOAT) {
            OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdamDDagFusionNesterov<float>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                            OP_LOGE(tilingContext_->GetNodeName(), "do tiling failed for fp32 with nesterov"),
                            return ge::GRAPH_FAILED);
        } else {
            OP_LOGE(tilingContext_->GetNodeName(), "current dtype not supported");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (this->varDtype_ == ge::DT_FLOAT16 || this->varDtype_ == ge::DT_BF16) {
            OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdamDDagFusion<half>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                            OP_LOGE(tilingContext_->GetNodeName(), "do tiling failed for fp16"),
                            return ge::GRAPH_FAILED);
        } else if (this->varDtype_ == ge::DT_FLOAT) {
            OP_CHECK_IF(eleBaseTiling.DoTiling<ApplyAdamDDagFusion<float>::OpDag>(tiling_->baseTiling) != ge::GRAPH_SUCCESS,
                            OP_LOGE(tilingContext_->GetNodeName(), "do tiling failed for fp32"),
                            return ge::GRAPH_FAILED);
        } else {
            OP_LOGE(tilingContext_->GetNodeName(), "current dtype not supported");
            return ge::GRAPH_FAILED;
        }
    }

    return SetTilingData();
}

static ge::graphStatus TilingPrepareForApplyAdamD(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ApplyAdamDCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForApplyAdamD(gert::TilingContext *context)
{
    OP_LOGD("ApplyAdamDTiling", "Enter TilingForApplyAdamD");
    if (context == nullptr) {
        OP_LOGE("ApplyAdamDTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const ApplyAdamDCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD("ApplyAdamDTiling", "Enter new ApplyAdamDTiling");
    ApplyAdamDTiling tiling(context);
    return tiling.RunTiling();
}

IMPL_OP_OPTILING(ApplyAdamD)
    .Tiling(TilingForApplyAdamD)
    .TilingParse<ApplyAdamDCompileInfo>(TilingPrepareForApplyAdamD);
}