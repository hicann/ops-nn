/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_v2_ar_full_load_tiling.cpp
 * \brief
 */
#include "softmax_v2_tiling.h"
#include "util/math_util.h"
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace Ops::Base;
using namespace ge;

namespace optiling
{
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr int64_t UB_RESERVED_BYTE = 1024;
static constexpr int64_t R_MAX_VALUE = 16384;
static constexpr int64_t BINARY_TMP_LOCAL_SHAPE = 512;

bool SoftmaxV2TilingAR::IsCapable()
{
    OP_CHECK_IF(a0_ != DIM_NUM_ONE, OP_LOGI(context_->GetNodeName(), "AR full load template is not capable. "),
                    return false);

    OP_CHECK_IF(r_ > R_MAX_VALUE,
                    OP_LOGI(context_->GetNodeName(),
                            "AR full load template is not capable. actual r is %ld, larger than %ld", r_, R_MAX_VALUE),
                    return false);
    return true;
}

ge::graphStatus SoftmaxV2TilingAR::DoOpTiling()
{
    int64_t rAligned = CeilAlign(r_, (BLOCK_SIZE / xDtypeSize_));

    int64_t ubFactor =
        (aicoreParams_.ubSize - UB_RESERVED_BYTE - BINARY_TMP_LOCAL_SHAPE) /
        (rAligned * FLOAT32_BYTES + DOUBLE_BUFFER * rAligned * xDtypeSize_ + DOUBLE_BUFFER * rAligned * yDtypeSize_);
    OP_CHECK_IF(
        (ubFactor <= 0),
        OP_LOGI(context_->GetNodeName(), "AR full load template is not capable. r is %ld, ubFactor %ld", r_, ubFactor),
        return ge::GRAPH_PARAM_INVALID);

    int64_t rLoopCount = CeilDiv(rAligned, vlFp32_);

    // 按a1分核
    int64_t aBlockFactor = CeilDiv(a1_, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = CeilDiv(a1_, aBlockFactor);

    ubFactor = ubFactor < aBlockFactor ? ubFactor : aBlockFactor;

    // tiling data设置
    tilingData_.set_a(a1_);
    tilingData_.set_r(r_);
    tilingData_.set_rAligned(rAligned);
    tilingData_.set_aBlockFactor(aBlockFactor);
    tilingData_.set_rLoopCount(rLoopCount);
    tilingData_.set_ubFactor(ubFactor);

    return ge::GRAPH_SUCCESS;
}

uint64_t SoftmaxV2TilingAR::GetTilingKey() const
{
    return TILINGKEY_AR;
}

ge::graphStatus SoftmaxV2TilingAR::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxV2, SoftmaxV2TilingAR, TEMPLATE_AR_FULL_LOAD_PRIORITY);

}  // namespace optiling
