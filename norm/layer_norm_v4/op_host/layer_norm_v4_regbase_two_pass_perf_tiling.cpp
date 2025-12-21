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
 * \file layer_norm_v4_regbase_two_pass_perf_tiling.cpp
 * \brief
 */

#include "layer_norm_v4_tiling.h"

namespace optiling {
static constexpr int64_t LN_DOUBLE_BUFFER = 2;
static constexpr uint32_t LN_MINIMAL_WORKSPACE = 32;
static constexpr int64_t LN_NUM_TWO = 2;
static constexpr int64_t LN_BLOCK_SIZE = 32;
static constexpr int64_t LN_B32_ALIGN_NUM = LN_BLOCK_SIZE / sizeof(float);
static constexpr int64_t LN_ROW_THRESHOLD = 4096;
static constexpr int64_t LN_COL_THRESHOLD = 8192;

int64_t LayerNormV4RegBaseTwoPassPerfTiling::GetUBCanUseSize()
{
    int64_t binaryAddTmpSize = commonParams.vlFp32 * sizeof(float) * LN_NUM_TWO * LN_NUM_TWO;
    return commonParams.ubSizePlatForm - binaryAddTmpSize;
}

int64_t LayerNormV4RegBaseTwoPassPerfTiling::GetRowWeight()
{
    int64_t xElemSize = sizeof(float);
    if (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) {
        xElemSize = xElemSize / LN_NUM_TWO;
    }
    int64_t betaElemSize = sizeof(float);
    if (commonParams.paramDtype == ge::DT_FLOAT16 || commonParams.paramDtype == ge::DT_BF16) {
        betaElemSize = betaElemSize / LN_NUM_TWO;
    }

    return LN_NUM_TWO * betaElemSize + LN_DOUBLE_BUFFER * xElemSize + LN_DOUBLE_BUFFER * xElemSize + sizeof(float);
}

bool LayerNormV4RegBaseTwoPassPerfTiling::CanFitInBuffer(int64_t curA)
{
    int64_t curAAlign = (curA + LN_B32_ALIGN_NUM - 1) / LN_B32_ALIGN_NUM * LN_B32_ALIGN_NUM;
    int64_t ubCanUseSize = GetUBCanUseSize();
    int64_t rowWeight = GetRowWeight();

    return curA * static_cast<int64_t>(commonParams.rowAlign) * rowWeight <=
           ubCanUseSize - LN_NUM_TWO * LN_DOUBLE_BUFFER * curAAlign;
}

bool LayerNormV4RegBaseTwoPassPerfTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }

    if (static_cast<int64_t>(commonParams.rowAlign) > LN_NUM_TWO * commonParams.vlFp32 * commonParams.vlFp32 * LN_NUM_TWO) {
        return false;
    }

    // tile size is not large enough
    if (commonParams.colSize >= LN_ROW_THRESHOLD && commonParams.rowAlign >= LN_COL_THRESHOLD) {
        return false;
    }

    if (!CanFitInBuffer(1)) {
        return false;
    }

    return true;
}

uint64_t LayerNormV4RegBaseTwoPassPerfTiling::GetTilingKey() const
{
    uint64_t tilingKey = -1;
    if (commonParams.tensorDtype == ge::DT_FLOAT && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT32_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT16_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT16) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT16_FLOAT16);
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_PERF_BFLOAT16_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_BF16) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_REGBASE_TWO_PASS_PERF_BFLOAT16_BFLOAT16);
    }
    return tilingKey;
}

ge::graphStatus LayerNormV4RegBaseTwoPassPerfTiling::DoOpTiling()
{
    // optional input
    td_.set_nullptrGamma(static_cast<int8_t>(commonParams.gammaNullPtr));
    td_.set_nullptrBeta(static_cast<int8_t>(commonParams.betaNullPtr));
    // attr
    td_.set_epsilon(commonParams.eps);

    // dim
    int64_t a = commonParams.colSize;
    int64_t r = commonParams.rowSize;
    int64_t rAlign = commonParams.rowAlign;
    td_.set_a(a);
    td_.set_r(r);
    td_.set_rAlign(rAlign);

    int64_t aBlockFactor = (a + commonParams.coreNum - 1) / commonParams.coreNum;
    blockNum_ = (a + aBlockFactor - 1) / aBlockFactor;
    td_.set_aBlockFactor(aBlockFactor);

    int64_t aUbFactor = GetUBCanUseSize() / (commonParams.rowAlign * GetRowWeight());
    while (!CanFitInBuffer(aUbFactor)) {
        aUbFactor--;
    }
    td_.set_aUbFactor(aUbFactor);
    int64_t aUbFactorAlignB32 = (aUbFactor + LN_B32_ALIGN_NUM - 1) / LN_B32_ALIGN_NUM * LN_B32_ALIGN_NUM;
    td_.set_aUbFactorAlignB32(aUbFactorAlignB32);

    int64_t formerBlockUbLoops = (aBlockFactor + aUbFactor - 1) / aUbFactor;
    int64_t tailBlockUbLoops = (a - aBlockFactor * (blockNum_ - 1) + aUbFactor - 1) / aUbFactor;
    td_.set_formerBlockUbLoops(formerBlockUbLoops);
    td_.set_tailBlockUbLoops(tailBlockUbLoops);

    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < r) {
        powerOfTwoForR *= LN_NUM_TWO;
    }
    td_.set_powerOfTwoForR(powerOfTwoForR);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4RegBaseTwoPassPerfTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4RegBaseTwoPassPerfTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = LN_MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV4RegBaseTwoPassPerfTiling, 150);
REGISTER_OPS_TILING_TEMPLATE(LayerNormV4, LayerNormV4RegBaseTwoPassPerfTiling, 150);
} // namespace optiling
