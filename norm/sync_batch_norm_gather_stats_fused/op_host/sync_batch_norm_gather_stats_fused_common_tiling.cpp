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
 * \file sync_batch_norm_gather_stats_fused_common_tiling.cpp
 * \brief
 */

#include "sync_batch_norm_gather_stats_fused_tiling.h"

namespace optiling {

static const int64_t COMMON_BLOCK_BYTES = 32;
static const int64_t COMMON_B32_DTYPE_SIZE = 4;
static const int64_t COMMON_B16_DTYPE_SIZE = 2;
static const int64_t COMMON_LNG_B16_ALIGN_FACTOR = 16;
static const int64_t COMMON_LNG_B32_ALIGN_FACTOR = 8;
static const int64_t COMMON_MAX_C = 240000;
static const int64_t FLOAT_DTYPE_SIZE = 4;

static inline int64_t CommonCeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

static inline int64_t CommonCeilAlign(int64_t value, int64_t alignment)
{
    return CommonCeilDiv(value, alignment) * alignment;
}

bool SyncBatchNormGatherStatsFusedCommonTiling::IsCapable()
{
    if (commonParams.cLength > COMMON_MAX_C ||
        (commonParams.cLength < commonParams.nLength && commonParams.nLength > 255)) {
        OP_LOGI(context_, "SyncBatchNormGatherStatsFusedCommonTiling: cLength=%ld exceeds limit %ld",
                commonParams.cLength, COMMON_MAX_C);
        return false;
    }
    return true;
}

uint64_t SyncBatchNormGatherStatsFusedCommonTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(TemplateKey::COMMON);
    return templateKey * TEMPLATE_KEY_WEIGHT + static_cast<uint64_t>(commonParams.dtypeKey);
}

int64_t SyncBatchNormGatherStatsFusedCommonTiling::CalculateubFormer()
{
    int64_t maxUBSize = commonParams.ubSizePlatForm - td_.get_cBufferByteSize() * 4 - 1024; // 除缓冲区外的内存
    int64_t coff = cFormerAlignM_ * COMMON_B32_DTYPE_SIZE * 2 + 4 + 32;
    int64_t curUBFormer = maxUBSize / coff;
    for (; curUBFormer >= 0; curUBFormer--) { // 迭代搜优
        int64_t wholeBufferByteSize = curUBFormer * cFormerAlignM_ * 4;
        int64_t nBufferByteSize = CommonCeilDiv(curUBFormer * 4, 32) * 32;
        int64_t nBrcbBufferByteSize = CommonCeilDiv(curUBFormer, 8) * 8 * 32; // BRCB
        int64_t curUBSize = wholeBufferByteSize * 2 + nBufferByteSize * 1 + nBrcbBufferByteSize;
        if (curUBSize <= maxUBSize && nBrcbBufferByteSize <= wholeBufferByteSize) {
            td_.set_wholeBufferByteSize(wholeBufferByteSize);
            td_.set_nBufferByteSize(nBufferByteSize);
            td_.set_nBrcbBufferByteSize(nBrcbBufferByteSize);
            break;
        }
    }
    return curUBFormer;
}

ge::graphStatus SyncBatchNormGatherStatsFusedCommonTiling::DoOpTiling()
{
    td_.set_nLength(commonParams.nLength);
    td_.set_cLength(commonParams.cLength);

    int64_t blockFormer = CommonCeilDiv(commonParams.cLength, static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = CommonCeilDiv(commonParams.cLength, blockFormer);
    int64_t blockTail = commonParams.cLength - (blockNum - 1) * blockFormer;
    td_.set_blockFormer(blockFormer);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail);

    cFormerAlignV_ = CommonCeilDiv(blockFormer, COMMON_LNG_B32_ALIGN_FACTOR) * COMMON_LNG_B32_ALIGN_FACTOR;
    cTailAlignV_ = CommonCeilDiv(blockTail, COMMON_LNG_B32_ALIGN_FACTOR) * COMMON_LNG_B32_ALIGN_FACTOR;
    cFormerAlignM_ = cFormerAlignV_;
    cTailAlignM_ = cTailAlignV_;
    if (commonParams.dtype != ge::DataType::DT_FLOAT) {
        cFormerAlignM_ = CommonCeilDiv(blockFormer, COMMON_LNG_B16_ALIGN_FACTOR) * COMMON_LNG_B16_ALIGN_FACTOR;
        cTailAlignM_ = CommonCeilDiv(blockTail, COMMON_LNG_B16_ALIGN_FACTOR) * COMMON_LNG_B16_ALIGN_FACTOR;
    }

    td_.set_cFormerAlignV_(cFormerAlignV_);
    td_.set_cFormerAlignM_(cFormerAlignM_);
    td_.set_cTailAlignV_(cTailAlignV_);
    td_.set_cTailAlignM_(cTailAlignM_);
    td_.set_cBufferByteSize(cFormerAlignM_ * COMMON_B32_DTYPE_SIZE);
    int64_t ubFormer = CalculateubFormer();
    if (ubFormer <= 0) {
        OP_LOGE(context_, "SyncBatchNormGatherStatsFusedCommonTiling: Calculate ubFormer failed, ubFormer=%ld",
                ubFormer);
        return ge::GRAPH_FAILED;
    }

    int64_t ubLoop = CommonCeilDiv(commonParams.nLength, ubFormer);
    int64_t ubTail = commonParams.nLength - (ubLoop - 1) * ubFormer;
    td_.set_wholeBufferElemNums(ubFormer * cFormerAlignM_);
    td_.set_ubFormer(ubFormer);
    td_.set_ubLoop(ubLoop);
    td_.set_ubTail(ubTail);

    td_.set_momentum(commonParams.momentum);
    td_.set_eps(commonParams.eps);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedCommonTiling::PostTiling()
{
    context_->SetBlockDim(commonParams.coreNum);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedCommonTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = commonParams.coreNum * FLOAT_DTYPE_SIZE + 16 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("SyncBatchNormGatherStatsFused", SyncBatchNormGatherStatsFusedCommonTiling, 1000);

} // namespace optiling