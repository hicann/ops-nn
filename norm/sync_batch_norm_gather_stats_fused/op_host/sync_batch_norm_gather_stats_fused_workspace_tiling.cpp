/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /* !
 * \file sync_batch_norm_gather_stats_fused_workspace_tiling.cpp
 * \brief
 */

#include "sync_batch_norm_gather_stats_fused_tiling.h"

namespace optiling {

static const int64_t WORKSPACE_BLOCK_BYTES = 32;
static const int64_t WORKSPACE_B32_DTYPE_SIZE = 4;
static const int64_t WORKSPACE_B16_ALIGN_FACTOR = 16;
static const int64_t WORKSPACE_B32_ALIGN_FACTOR = 8;
static const int64_t WORKSPACE_MAX_BUFFER_NUM = 5;
static const int64_t WORKSPACE_MIN_C = 240000;
static const int64_t FLOAT_DTYPE_SIZE = 4;

static inline int64_t WorkspaceCeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

bool SyncBatchNormGatherStatsFusedWorkspaceTiling::IsCapable()
{
    if (commonParams.cLength < commonParams.nLength && commonParams.nLength > 255) {
        OP_LOGI(
            context_, "SyncBatchNormGatherStatsFusedWorkspaceTiling: cLength=%ld exceeds limit %ld",
            commonParams.cLength, WORKSPACE_MIN_C);
        return false;
    }
    return true;
}

uint64_t SyncBatchNormGatherStatsFusedWorkspaceTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(TemplateKey::WORKSPACE);
    return templateKey * TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

ge::graphStatus SyncBatchNormGatherStatsFusedWorkspaceTiling::DoOpTiling()
{
    td_.set_nLength(commonParams.nLength);
    td_.set_cLength(commonParams.cLength);

    int64_t blockFormer = WorkspaceCeilDiv(commonParams.cLength, static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = WorkspaceCeilDiv(commonParams.cLength, blockFormer);
    int64_t blockTail = commonParams.cLength - (blockNum - 1) * blockFormer;
    td_.set_blockFormer(blockFormer);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail);
    int64_t cFormerAlignV_ = WorkspaceCeilDiv(blockFormer, WORKSPACE_B16_ALIGN_FACTOR) * WORKSPACE_B16_ALIGN_FACTOR;
    int64_t cTailAlignV_ = WorkspaceCeilDiv(blockTail, WORKSPACE_B16_ALIGN_FACTOR) * WORKSPACE_B16_ALIGN_FACTOR;

    int64_t maxBufferSize = (commonParams.ubSizePlatForm - 1024) / WORKSPACE_MAX_BUFFER_NUM / FLOAT_DTYPE_SIZE /
                            WORKSPACE_B16_ALIGN_FACTOR * WORKSPACE_B16_ALIGN_FACTOR;

    int64_t ubFormerOfFormer = std::min(maxBufferSize, cFormerAlignV_);
    int64_t ubLoopOfFormer = WorkspaceCeilDiv(blockFormer, ubFormerOfFormer);
    int64_t ubTailOfFormer = blockFormer - (ubLoopOfFormer - 1) * ubFormerOfFormer;

    int64_t ubFormerOfTail = std::min(maxBufferSize, cTailAlignV_);
    int64_t ubLoopOfTail = WorkspaceCeilDiv(blockTail, ubFormerOfTail);
    int64_t ubTailOfTail = blockTail - (ubLoopOfTail - 1) * ubFormerOfTail;

    td_.set_ubFormerOfFormer(ubFormerOfFormer);
    td_.set_ubTailOfFormer(ubTailOfFormer);
    td_.set_ubLoopOfFormer(ubLoopOfFormer);
    td_.set_ubFormerOfTail(ubFormerOfTail);
    td_.set_ubTailOfTail(ubTailOfTail);
    td_.set_ubLoopOfTail(ubLoopOfTail);
    td_.set_momentum(commonParams.momentum);
    td_.set_eps(commonParams.eps);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedWorkspaceTiling::PostTiling()
{
    context_->SetBlockDim(commonParams.coreNum);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedWorkspaceTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = commonParams.coreNum * FLOAT_DTYPE_SIZE + 16 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("SyncBatchNormGatherStatsFused", SyncBatchNormGatherStatsFusedWorkspaceTiling, 2000);

} // namespace optiling