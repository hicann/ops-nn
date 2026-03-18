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
 * \file ada_layer_norm_grad_workspace_tiling.cc
 * \brief
 */
#include "ada_layer_norm_grad_tiling.h"

namespace optiling {
static const int64_t LNG_TMP_BUFFER_SIZE_0 = 64;
static const int64_t LNG_B32_DTYPE_SIZE = 4;
static const int64_t LNG_B16_ALIGN_FACTOR = 16;
static const int64_t LNG_B32_ALIGN_FACTOR = 8;
static const int64_t LNG_MAX_BUFFER_NUM = 6;
static const int64_t LNG_CONSTANT_THREE = 3;
static const size_t LNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;
static const int64_t LNG_MAX_COL = 12176;

static inline int64_t CeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

bool AdaLayerNormGradWorkspaceTiling::IsCapable()
{
    if (commonParams.isRegBase) {
        return false;
    }
    return true;
}

uint64_t AdaLayerNormGradWorkspaceTiling::GetTilingKey() const
{
    uint64_t templateKey = static_cast<uint64_t>(LNGTemplateKey::WORKSPACE);
    return templateKey * LNG_TEMPLATE_KEY_WEIGHT + commonParams.isDeterministicKey * LNG_DETERMINISTIC_KEY_WEIGHT +
           static_cast<uint64_t>(commonParams.dtypeKey);
}

ge::graphStatus AdaLayerNormGradWorkspaceTiling::PostTiling()
{
    context_->SetBlockDim(commonParams.coreNum);
    td_.SaveToBuffer(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());                             

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradWorkspaceTiling::DoOpTiling() 
{
    int64_t colAlignV = CeilDiv(static_cast<int64_t>(commonParams.colSize), LNG_B16_ALIGN_FACTOR) *
                        LNG_B16_ALIGN_FACTOR;
    int64_t colAlignM = 0;
    if (commonParams.dyDtype == ge::DataType::DT_FLOAT) {
        colAlignM = colAlignV;
    } else {
        colAlignM = CeilDiv(static_cast<int64_t>(commonParams.colSize), LNG_B16_ALIGN_FACTOR) * LNG_B16_ALIGN_FACTOR;
    }
    int64_t batch_ = static_cast<int64_t>(commonParams.batchSize);
    int64_t seq_ = static_cast<int64_t>(commonParams.seqSize);
    int64_t row_ = static_cast<int64_t>(commonParams.rowSize);
    int64_t col_ = static_cast<int64_t>(commonParams.colSize);

    td_.set_batch(batch_);
    td_.set_seq(seq_);
    td_.set_row(row_); 
    td_.set_col(col_);
    td_.set_colAlignV(colAlignV);
    td_.set_colAlignM(colAlignM);
    // calculate block tiling, batch split block

    int64_t blockFormer = CeilDiv(batch_, static_cast<int64_t>(commonParams.coreNum));
    int64_t blockNum = CeilDiv(batch_, blockFormer); 
    int64_t blockTail = batch_ - (blockNum - 1) * blockFormer;             
    td_.set_blockFormer(blockFormer * seq_);
    td_.set_blockNum(blockNum);
    td_.set_blockTail(blockTail * seq_);
    int64_t maxBufferSize = (commonParams.ubSizePlatForm - LNG_TMP_BUFFER_SIZE_0 * LNG_CONSTANT_THREE) /
                            LNG_MAX_BUFFER_NUM / LNG_B32_DTYPE_SIZE / LNG_B16_ALIGN_FACTOR *
                            LNG_B16_ALIGN_FACTOR;                                   
    int64_t ubFormer = std::min(maxBufferSize, colAlignV);                          
    int64_t ubLoop = CeilDiv(static_cast<int64_t>(commonParams.colSize), ubFormer); 
    int64_t ubTail = commonParams.colSize - (ubLoop - 1) * ubFormer;                
    td_.set_ubFormer(ubFormer);
    td_.set_ubLoop(ubLoop);
    td_.set_ubTail(ubTail);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradWorkspaceTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = td_.get_blockNum() * td_.get_colAlignV() * LNG_B32_DTYPE_SIZE * (4 + 2 * td_.get_blockFormer() / td_.get_seq()) + LNG_WORKSPACE_RESERVED;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("AdaLayerNormGrad", AdaLayerNormGradWorkspaceTiling, 4000); //
} // namespace optiling