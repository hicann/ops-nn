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
 * \file layer_norm_v3_regbase_no_reduce_tiling.cpp
 * \brief
 */

#include "layer_norm_v3_tiling.h"
#include "op_common/op_host/util/math_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
static constexpr int64_t LNV3_DOUBLE_BUFFER = 2;
static constexpr uint32_t LNV3_MINIMAL_WORKSPACE = 32;
static constexpr int64_t LNV3_NUM_TWO = 2;
static constexpr int64_t LNV3_BLOCK_SIZE = 32;
static constexpr int64_t MIN_TILING_BITS_SIZE_PER_CORE = 32768; // 4KB
static constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;

bool LayerNormV3RegBaseNoReduceTiling::IsCapable()
{
    if (!commonParams.isRegBase) {
        return false;
    }

    if (static_cast<int64_t>(commonParams.rowSize) != 1) {
        return false;
    }

    return true;
}

uint64_t LayerNormV3RegBaseNoReduceTiling::GetTilingKey() const
{
    uint64_t tilingKey = -1;
    if (commonParams.tensorDtype == ge::DT_FLOAT && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV3TilingKey::LAYER_NORM_REGBASE_NO_REDUCE_FLOAT32_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV3TilingKey::LAYER_NORM_REGBASE_NO_REDUCE_FLOAT16_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT16) {
        tilingKey = static_cast<uint64_t>(LayerNormV3TilingKey::LAYER_NORM_REGBASE_NO_REDUCE_FLOAT16_FLOAT16);
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV3TilingKey::LAYER_NORM_REGBASE_NO_REDUCE_BFLOAT16_FLOAT32);
    }
    if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_BF16) {
        tilingKey = static_cast<uint64_t>(LayerNormV3TilingKey::LAYER_NORM_REGBASE_NO_REDUCE_BFLOAT16_BFLOAT16);
    }
    return tilingKey;
}

ge::graphStatus LayerNormV3RegBaseNoReduceTiling::DoOpTiling()
{
    // optional input
    td_.set_nullptrGamma(static_cast<int8_t>(commonParams.gammaNullPtr));
    td_.set_nullptrBeta(static_cast<int8_t>(commonParams.betaNullPtr));
    // attr
    td_.set_epsilon(commonParams.eps);

    // dim
    int64_t a = commonParams.colSize;
    td_.set_a(a);

    int64_t xElemSize = sizeof(float);
    int64_t betaElemSize = sizeof(float);
    int64_t tmpSize = sizeof(float);
    if (commonParams.tensorDtype == ge::DT_FLOAT16 || commonParams.tensorDtype == ge::DT_BF16) {
        xElemSize = xElemSize / LNV3_NUM_TWO;
    }
    if (commonParams.paramDtype == ge::DT_FLOAT16 || commonParams.paramDtype == ge::DT_BF16) {
        betaElemSize = betaElemSize / LNV3_NUM_TWO;
    }
    uint64_t coreNum = (a * xElemSize + MIN_TILING_BITS_SIZE_PER_CORE - 1) / MIN_TILING_BITS_SIZE_PER_CORE;
    if (coreNum > commonParams.coreNum) {
        coreNum = commonParams.coreNum;
    }
    int64_t aBlockFactor = (((a + coreNum - 1) / coreNum) * xElemSize + CACHE_LINE_BYTE_LENGTH - 1) /
                           CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH / xElemSize;
    blockNum_ = (a + aBlockFactor - 1) / aBlockFactor;
    td_.set_aBlockFactor(aBlockFactor);

    int64_t aUbFactor = (commonParams.ubSizePlatForm - LNV3_BLOCK_SIZE * LNV3_NUM_TWO) /
                        (LNV3_DOUBLE_BUFFER * (xElemSize * LNV3_NUM_TWO + betaElemSize * LNV3_NUM_TWO) + tmpSize);
    aUbFactor = (aUbFactor * xElemSize) / CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH / xElemSize;
    td_.set_aUbFactor(aUbFactor);
    int64_t formerBlockUbLoops = (aBlockFactor + aUbFactor - 1) / aUbFactor;
    int64_t tailBlockUbLoops = (a - aBlockFactor * (blockNum_ - 1) + aUbFactor - 1) / aUbFactor;
    td_.set_formerBlockUbLoops(formerBlockUbLoops);
    td_.set_tailBlockUbLoops(tailBlockUbLoops);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3RegBaseNoReduceTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3RegBaseNoReduceTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = LNV3_MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV3RegBaseNoReduceTiling, 50);
REGISTER_OPS_TILING_TEMPLATE(LayerNorm, LayerNormV3RegBaseNoReduceTiling, 50);
} // namespace optiling