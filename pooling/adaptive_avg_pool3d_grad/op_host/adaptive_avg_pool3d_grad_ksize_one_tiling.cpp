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
 * \file adaptive_avg_pool3d_grad_ksize_one_tiling.cpp
 * \brief
 */
#include "adaptive_avg_pool3d_grad_ksize_one_tiling.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t UB_RESERVED_SIZE = 1024;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t MIN_DATA_SIZE = 1024;
static constexpr int64_t ALIGN_LENTH = 512;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t MAX_INT32 = 2147483647;

bool AdaptiveAvgPool3dGradTilingKsizeOne::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }
    // kernel=1 in all dims means x_grad == y_grad, a pure memcpy
    if (inputData.dX != inputData.dGrad || inputData.hX != inputData.hGrad || inputData.wX != inputData.wGrad) {
        return false;
    }
    return true;
}

uint64_t AdaptiveAvgPool3dGradTilingKsizeOne::GetTilingKey() const
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    uint32_t idxDtype = outDataCount <= static_cast<int64_t>(MAX_INT32) ? TPL_INT32 : TPL_INT64;
    uint32_t isChannelLast = 0;
    return GET_TPL_TILING_KEY(TPL_KSIZE_ONE_KERNEL, idxDtype, isChannelLast);
}

void AdaptiveAvgPool3dGradTilingKsizeOne::DoUBTiling()
{
    int64_t inputBytes = (inputData.inputDtype == ge::DT_FLOAT) ? FLOAT32_SIZE : FLOAT16_SIZE;
    ubFactor_ = (ubSize_ - UB_RESERVED_SIZE) / BUFFER_NUM / inputBytes;
    int64_t alignSize = ALIGN_LENTH / inputBytes;
    int64_t coreData = Ops::Base::CeilDiv(inputData.gradShapeSize, coreNum_);
    coreData = Ops::Base::CeilAlign(coreData, alignSize);
    coreData = std::max(coreData, MIN_DATA_SIZE);
    usedCoreNum_ = Ops::Base::CeilDiv(inputData.gradShapeSize, coreData);
    // 512字节对齐
    blockFactor_ = coreData;
    tailBlockFactor_ = inputData.gradShapeSize - (usedCoreNum_ - 1) * blockFactor_;
    coreLoop_ = Ops::Base::CeilDiv(blockFactor_, ubFactor_);
    tailUbFactor_ = blockFactor_ - (coreLoop_ - 1) * ubFactor_;
    tailCoreLoop_ = Ops::Base::CeilDiv(tailBlockFactor_, ubFactor_);
    tailCoreTailUbFactor_ = tailBlockFactor_ - (tailCoreLoop_ - 1) * ubFactor_;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingKsizeOne::DoOpTiling()
{
    DoUBTiling();

    AdaptiveAvgPool3dGradKsizeOneTilingDataV35*
        tilingData = context_->GetTilingData<AdaptiveAvgPool3dGradKsizeOneTilingDataV35>();
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->blockFactor = blockFactor_;
    tilingData->tailBlockFactor = tailBlockFactor_;
    tilingData->coreLoop = coreLoop_;
    tilingData->tailCoreLoop = tailCoreLoop_;
    tilingData->ubFactor = ubFactor_;
    tilingData->tailUbFactor = tailUbFactor_;
    tilingData->tailCoreTailUbFactor = tailCoreTailUbFactor_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingKsizeOne::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingKsizeOne::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingKsizeOne::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingKsizeOne, 5);
} // namespace optiling
