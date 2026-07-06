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
 * \file sync_batch_norm_gather_stats_fused_first_axis_common_tiling.cpp
 * \brief
 */

#include "sync_batch_norm_gather_stats_fused_tiling.h"

namespace optiling {

static const int64_t FIRST_AXIS_COMMON_BLOCK_BYTES = 32;
static const int64_t FIRST_AXIS_COMMON_B32_DTYPE_SIZE = 4;
static const int64_t FIRST_AXIS_COMMON_B16_DTYPE_SIZE = 2;
static const int64_t FIRST_AXIS_COMMON_LNG_B16_ALIGN_FACTOR = 16;
static const int64_t FIRST_AXIS_COMMON_LNG_B32_ALIGN_FACTOR = 8;
static const int64_t FIRST_AXIS_COMMON_MAX_C = 5000;
static const int64_t FIRST_AXIS_FLOAT_DTYPE_SIZE = 4;
static const size_t FIRST_AXIS_COMMON_WORKSPACE_RESERVED = 16 * 1024 * 1024;

static inline int64_t CommonCeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

static inline int64_t CommonCeilAlign(int64_t value, int64_t alignment)
{
    return CommonCeilDiv(value, alignment) * alignment;
}

bool SyncBatchNormGatherStatsFusedFirstAxisCommonTiling::IsCapable()
{
    if (commonParams.cLength > FIRST_AXIS_COMMON_MAX_C) {
        OP_LOGI(context_, "SyncBatchNormGatherStatsFusedFirstAxisCommonTiling: cLength=%ld exceeds limit %ld",
                commonParams.cLength, FIRST_AXIS_COMMON_MAX_C);
        return false;
    }
    return true;
}

uint64_t SyncBatchNormGatherStatsFusedFirstAxisCommonTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(TemplateKey::FIRST_AXIS_COMMON);
    return templateKey * TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

int64_t SyncBatchNormGatherStatsFusedFirstAxisCommonTiling::CalculateubFormer()
{
    int64_t maxUBSize = commonParams.ubSizePlatForm - td_.get_cBufferByteSize() * 2 - 1024; // 除缓冲区外的内存
    int64_t coff = cAlignM_ * FIRST_AXIS_COMMON_B32_DTYPE_SIZE * 2 + 4 + 32;
    int64_t curUbFormer = maxUBSize / coff;
    for (; curUbFormer >= 0; curUbFormer--) { // 迭代搜优
        int64_t wholeBufferByteSize = curUbFormer * cAlignM_ * FIRST_AXIS_FLOAT_DTYPE_SIZE;
        int64_t nBufferByteSize = CommonCeilDiv(curUbFormer * FIRST_AXIS_FLOAT_DTYPE_SIZE,
                                                FIRST_AXIS_COMMON_BLOCK_BYTES) *
                                  FIRST_AXIS_COMMON_BLOCK_BYTES;
        int64_t nBrcbBufferByteSize = CommonCeilDiv(curUbFormer, FIRST_AXIS_COMMON_LNG_B32_ALIGN_FACTOR) *
                                      FIRST_AXIS_COMMON_LNG_B32_ALIGN_FACTOR * FIRST_AXIS_COMMON_BLOCK_BYTES; // BRCB
        int64_t curUBSize = wholeBufferByteSize * 2 + nBufferByteSize * 1 + nBrcbBufferByteSize;
        if (curUBSize <= maxUBSize && nBrcbBufferByteSize <= wholeBufferByteSize) {
            td_.set_wholeBufferByteSize(wholeBufferByteSize);
            td_.set_nBufferByteSize(nBufferByteSize);
            td_.set_nBrcbBufferByteSize(nBrcbBufferByteSize);
            break;
        }
    }
    return curUbFormer;
}

ge::graphStatus SyncBatchNormGatherStatsFusedFirstAxisCommonTiling::DoOpTiling()
{
    td_.set_nLength(commonParams.nLength);
    td_.set_cLength(commonParams.cLength);

    int64_t blockFormer = CommonCeilDiv(commonParams.nLength, static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = CommonCeilDiv(commonParams.nLength, blockFormer);
    int64_t blockTail = commonParams.nLength - (blockNum - 1) * blockFormer;
    td_.set_blockFormer(blockFormer);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail);

    cAlignV_ = CommonCeilDiv(commonParams.cLength, FIRST_AXIS_COMMON_LNG_B32_ALIGN_FACTOR) *
               FIRST_AXIS_COMMON_LNG_B32_ALIGN_FACTOR;
    cAlignM_ = cAlignV_;

    if (commonParams.dtype != ge::DataType::DT_FLOAT) {
        cAlignM_ = CommonCeilDiv(commonParams.cLength, FIRST_AXIS_COMMON_LNG_B16_ALIGN_FACTOR) *
                   FIRST_AXIS_COMMON_LNG_B16_ALIGN_FACTOR;
    }

    td_.set_cAlignV(cAlignV_);
    td_.set_cAlignM(cAlignM_);
    td_.set_cBufferByteSize(cAlignM_ * FIRST_AXIS_FLOAT_DTYPE_SIZE);
    int64_t ubFormer = CalculateubFormer();
    if (ubFormer <= 0) {
        OP_LOGE(context_, "SyncBatchNormGatherStatsFusedFirstAxisCommonTiling: Calculate ubFormer failed, ubFormer=%ld",
                ubFormer);
        return ge::GRAPH_FAILED;
    }

    int64_t ubLoopOfFormerBlock = CommonCeilDiv(blockFormer, ubFormer);
    int64_t ubLoopOfTailBlock = CommonCeilDiv(blockTail, ubFormer);

    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    td_.set_ubFormer(ubFormer);
    td_.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    td_.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    td_.set_ubTailOfFormerBlock(ubTailOfFormerBlock);
    td_.set_ubTailOfTailBlock(ubTailOfTailBlock);
    td_.set_wholeBufferElemNums(ubFormer * cAlignM_);

    td_.set_momentum(commonParams.momentum);
    td_.set_eps(commonParams.eps);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedFirstAxisCommonTiling::PostTiling()
{
    context_->SetBlockDim(commonParams.coreNum);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedFirstAxisCommonTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = (td_.get_blockNum() + 2) * td_.get_cAlignM() * FIRST_AXIS_FLOAT_DTYPE_SIZE +
                    FIRST_AXIS_FLOAT_DTYPE_SIZE * commonParams.coreNum + FIRST_AXIS_COMMON_WORKSPACE_RESERVED;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("SyncBatchNormGatherStatsFused", SyncBatchNormGatherStatsFusedFirstAxisCommonTiling, 3000);

} // namespace optiling