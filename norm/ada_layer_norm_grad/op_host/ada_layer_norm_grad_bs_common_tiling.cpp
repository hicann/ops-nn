/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ada_layer_norm_grad_bs_common_tiling.cpp
 * \brief
 */

#include "ada_layer_norm_grad_tiling.h"

namespace optiling {
static const int64_t MERGE_BS_COMMON_LNG_BLOCK_BYTES = 32;
static const int64_t MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE = 4;
static const int64_t MERGE_BS_COMMON_LNG_B16_ALIGN_FACTOR = 16;
static const int64_t MERGE_BS_COMMON_LNG_B32_ALIGN_FACTOR = 8;
static const int64_t MERGE_BS_COMMON_LNG_WHOLE_BUFFER_NUM = 4;
static const int64_t MERGE_BS_COMMON_LNG_NLAST_R_BUFFER_NUM = 4;
static const int64_t MERGE_BS_COMMON_LNG_LAST_R_BUFFER_NUM = 2;
static const int64_t MERGE_BS_COMMON_LNG_BRCB_ALIGN_FACTOR = 8;
static const int64_t MERGE_BS_COMMON_LNG_SCALE_BUFFER_NUM = 3;
static const int64_t MERGE_BS_COMMON_LNG_WORKSPACE_NUM = 2;
static const size_t MERGE_BS_COMMON_LNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;
static const int64_t MERGE_BS_COMMON_LNG_MAX_COL = 3440;

static inline int64_t CommonCeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

bool AdaLayerNormGradMergeBSCommonTiling::IsCapable()
{
    if (commonParams.colSize > MERGE_BS_COMMON_LNG_MAX_COL) {
        OP_LOGI(
            context_,
            "AdaLayerNormGradMergeBSCommonTiling: col value(=%ld) is not support now, "
            "please check.",
            commonParams.colSize);
        return false;
    }
    return true;
}

uint64_t AdaLayerNormGradMergeBSCommonTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNGTemplateKey::MERGE_BS_COMMON);
    return templateKey * LNG_TEMPLATE_KEY_WEIGHT + commonParams.isDeterministicKey * LNG_DETERMINISTIC_KEY_WEIGHT +
           static_cast<uint64_t>(commonParams.dtypeKey);
}

ge::graphStatus AdaLayerNormGradMergeBSCommonTiling::DoOpTiling()
{   
    batch_ = static_cast<int64_t>(commonParams.batchSize);
    seq_ = static_cast<int64_t>(commonParams.seqSize);
    row_ = static_cast<int64_t>(commonParams.rowSize);
    col_ = static_cast<int64_t>(commonParams.colSize);
    colAlignV_ = CommonCeilDiv(col_, MERGE_BS_COMMON_LNG_B32_ALIGN_FACTOR) * MERGE_BS_COMMON_LNG_B32_ALIGN_FACTOR;
    if (commonParams.dyDtype == ge::DataType::DT_FLOAT) {
        colAlignM_ = colAlignV_;
    } else {
        colAlignM_ = CommonCeilDiv(col_, MERGE_BS_COMMON_LNG_B16_ALIGN_FACTOR) * MERGE_BS_COMMON_LNG_B16_ALIGN_FACTOR;
    }

    td_.set_batch(batch_);
    td_.set_seq(seq_);
    td_.set_row(row_);
    td_.set_col(col_);
    td_.set_colAlignV(colAlignV_);
    td_.set_colAlignM(colAlignM_);
    // calculate block tiling, row split block
    int64_t blockFormer = CommonCeilDiv(row_, static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = CommonCeilDiv(row_, blockFormer);
    int64_t blockTail = row_ - (blockNum - 1) * blockFormer;
    td_.set_blockFormer(blockFormer);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail);
    td_.set_nlastRBufferBytes(colAlignM_ * MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE);

    // calculate ub tiling, row split ub
    int64_t ubFormer = CalculateUbFormer();
    OP_CHECK_IF(
        (ubFormer <= 0),
        OP_LOGE(
            context_,
            "TilingForAdaLayerNormGrad: ub former(=%ld) is not greater than 0 "
            "in common tiling, please check",
            ubFormer),
        return ge::GRAPH_FAILED);
    int64_t ubLoopOfFormerBlock = Ops::Base::CeilDiv(blockFormer, ubFormer);
    int64_t ubLoopOfTailBlock = Ops::Base::CeilDiv(blockTail, ubFormer);
    td_.set_ubFormer(ubFormer);
    td_.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
    td_.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
    td_.set_ubTailOfFormerBlock(blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer);
    td_.set_ubTailOfTailBlock(blockTail - (ubLoopOfTailBlock - 1) * ubFormer);
    td_.set_wholeBufferElemNums(colAlignM_ * ubFormer);

    return ge::GRAPH_SUCCESS;
}

int64_t AdaLayerNormGradMergeBSCommonTiling::CalculateUbFormer()
{
    int64_t maxUBSize = commonParams.ubSizePlatForm - td_.get_nlastRBufferBytes() * MERGE_BS_COMMON_LNG_NLAST_R_BUFFER_NUM - 1024;//除缓冲区外的内存
    int64_t coff = colAlignM_ * MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE * MERGE_BS_COMMON_LNG_WHOLE_BUFFER_NUM +
                   MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE * MERGE_BS_COMMON_LNG_LAST_R_BUFFER_NUM + MERGE_BS_COMMON_LNG_BLOCK_BYTES;
    int64_t curUbFormer = maxUBSize / coff;
    for (; curUbFormer >= 0; curUbFormer--) {//迭代搜优
        int64_t wholeBufferBytes = curUbFormer * colAlignM_ * MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE;
        int64_t blockFormerScaleBufferBytes = (CommonCeilDiv(curUbFormer, seq_) + 1) * colAlignM_ * MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE;
        int64_t lastRBufferBytes =
            CommonCeilDiv(curUbFormer * MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE, MERGE_BS_COMMON_LNG_BLOCK_BYTES) * MERGE_BS_COMMON_LNG_BLOCK_BYTES;
        int64_t lastBrcbBufferBytes = CommonCeilDiv(curUbFormer, MERGE_BS_COMMON_LNG_BRCB_ALIGN_FACTOR) *
                                      MERGE_BS_COMMON_LNG_BRCB_ALIGN_FACTOR * MERGE_BS_COMMON_LNG_BLOCK_BYTES;//BRCB
        int64_t curUBSize = wholeBufferBytes * MERGE_BS_COMMON_LNG_WHOLE_BUFFER_NUM + lastRBufferBytes * MERGE_BS_COMMON_LNG_LAST_R_BUFFER_NUM + lastBrcbBufferBytes + blockFormerScaleBufferBytes * MERGE_BS_COMMON_LNG_SCALE_BUFFER_NUM;
        if (curUBSize <= maxUBSize && lastBrcbBufferBytes <= wholeBufferBytes) {
            td_.set_wholeBufferBytes(wholeBufferBytes);
            td_.set_blockFormerScaleBufferBytes(blockFormerScaleBufferBytes);
            td_.set_blockFormerScaleBufferElemNums((CommonCeilDiv(curUbFormer, seq_) + 1) * colAlignM_);
            td_.set_lastRBufferBytes(lastRBufferBytes);
            td_.set_lastBrcbBufferBytes(lastBrcbBufferBytes);
            break;
        }
    }
    return curUbFormer;
}

ge::graphStatus AdaLayerNormGradMergeBSCommonTiling::PostTiling()
{
    context_->SetBlockDim(commonParams.coreNum);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradMergeBSCommonTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    // workspace size is coreNum * colAlignV * 4 * 2
    workspaces[0] = td_.get_blockNum() * colAlignV_ * MERGE_BS_COMMON_LNG_B32_DTYPE_SIZE * (MERGE_BS_COMMON_LNG_WORKSPACE_NUM + 2 * td_.get_batch()) + MERGE_BS_COMMON_LNG_WORKSPACE_RESERVED;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("AdaLayerNormGrad", AdaLayerNormGradMergeBSCommonTiling, 2000);
} // namespace optiling