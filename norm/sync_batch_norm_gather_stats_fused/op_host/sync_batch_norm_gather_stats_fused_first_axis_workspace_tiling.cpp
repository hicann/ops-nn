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
 * \file sync_batch_norm_gather_stats_fused_first_axis_workspace_tiling.cpp
 * \brief
 */

#include "sync_batch_norm_gather_stats_fused_tiling.h"

namespace optiling {

static const int64_t FIRST_AXIS_WORKSPACE_BLOCK_BYTES = 32;
static const int64_t FIRST_AXIS_WORKSPACE_B32_DTYPE_SIZE = 4;
static const int64_t FIRST_AXIS_WORKSPACE_B16_ALIGN_FACTOR = 16;
static const int64_t FIRST_AXIS_WORKSPACE_B32_ALIGN_FACTOR = 8;
static const int64_t FIRST_AXIS_WORKSPACE_MAX_BUFFER_NUM = 3;
static const int64_t FIRST_AXIS_WORKSPACE_MIN_C = 4800;
static const int64_t FIRST_AXIS_FLOAT_DTYPE_SIZE = 4;
static const size_t FIRST_AXIS_WORKSPACE_RESERVED = 16 * 1024 * 1024;

static inline int64_t WorkspaceCeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

bool SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling::IsCapable() { return true; }

uint64_t SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(TemplateKey::FIRST_AXIS_WORKSPACE);
    return templateKey * TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

ge::graphStatus SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling::DoOpTiling()
{
    td_.set_nLength(commonParams.nLength);
    td_.set_cLength(commonParams.cLength);

    int64_t blockFormer = WorkspaceCeilDiv(commonParams.nLength, static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = WorkspaceCeilDiv(commonParams.nLength, blockFormer);
    int64_t blockTail = commonParams.nLength - (blockNum - 1) * blockFormer;
    td_.set_blockFormer(blockFormer);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail);
    int64_t cAlignV_ = WorkspaceCeilDiv(commonParams.cLength, FIRST_AXIS_WORKSPACE_B16_ALIGN_FACTOR) *
                       FIRST_AXIS_WORKSPACE_B16_ALIGN_FACTOR;

    int64_t maxBufferSize = (commonParams.ubSizePlatForm - 1024) / FIRST_AXIS_WORKSPACE_MAX_BUFFER_NUM /
                            FIRST_AXIS_FLOAT_DTYPE_SIZE / FIRST_AXIS_WORKSPACE_B16_ALIGN_FACTOR *
                            FIRST_AXIS_WORKSPACE_B16_ALIGN_FACTOR;

    int64_t ubFormer = std::min(maxBufferSize, cAlignV_);
    int64_t ubLoop = WorkspaceCeilDiv(commonParams.cLength, ubFormer);
    int64_t ubTail = commonParams.cLength - (ubLoop - 1) * ubFormer;

    td_.set_cAlignV(cAlignV_);
    td_.set_ubFormer(ubFormer);
    td_.set_ubTail(ubTail);
    td_.set_ubLoop(ubLoop);
    td_.set_momentum(commonParams.momentum);
    td_.set_eps(commonParams.eps);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling::PostTiling()
{
    context_->SetBlockDim(commonParams.coreNum);
    context_->SetScheduleMode(1);  // kernel 使用 SyncAll 全核同步，需设置为 BatchMode
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = (td_.get_blockNum() + 2) * td_.get_cAlignV() * FIRST_AXIS_FLOAT_DTYPE_SIZE +
                    FIRST_AXIS_WORKSPACE_RESERVED;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("SyncBatchNormGatherStatsFused", SyncBatchNormGatherStatsFusedFirstAxisWorkspaceTiling, 4000);

} // namespace optiling