/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "layer_norm_v4_tiling.h"

namespace optiling {
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

namespace {
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t BFP32_SIZE = sizeof(float);
constexpr uint64_t B16_SIZE = BFP32_SIZE / NUM_TWO;
constexpr uint64_t TRANSPOSE_R_LIMIT = 16;
constexpr uint64_t TRANSPOSE_MIN_VL_PER_CORE = 8;
constexpr uint64_t PACK_OUTPUT_MIN_A = 150000;
constexpr uint64_t TRANSPOSE_ROWS = 16;
constexpr uint64_t TRANSPOSE_BLOCK_BYTES = 32;
constexpr uint64_t TRANSPOSE_MAX_REPEATS = 255;
constexpr uint64_t TRANSPOSE_KEY_BASE = 800;
constexpr uint64_t BUFFER_COUNT = NUM_TWO;

} // namespace

bool LayerNormV4RegBaseTransposeTiling::IsCapable()
{
    return commonParams.isRegBase && commonParams.rowSize <= TRANSPOSE_R_LIMIT &&
           CeilDiv(commonParams.colSize, commonParams.coreNum) > TRANSPOSE_MIN_VL_PER_CORE * commonParams.vlFp32;
}

uint64_t LayerNormV4RegBaseTransposeTiling::GetTilingKey() const
{
    return TRANSPOSE_KEY_BASE + GetDTypeKey(commonParams.tensorDtype, commonParams.paramDtype);
}

ge::graphStatus LayerNormV4RegBaseTransposeTiling::DoOpTiling()
{
    const auto xType = commonParams.tensorDtype;
    const auto paramType = commonParams.paramDtype;
    if (paramType != ge::DT_FLOAT && paramType != xType) {
        return ge::GRAPH_PARAM_INVALID;
    }
    const uint64_t xBytes = xType == ge::DT_FLOAT ? BFP32_SIZE : B16_SIZE;
    const uint64_t paramBytes = paramType == ge::DT_FLOAT ? BFP32_SIZE : B16_SIZE;
    const uint64_t r = commonParams.rowSize;
    const uint64_t blockBytes = commonParams.blockSize;
    // TransDataTo5HD consumes 16 rows and 16 B16 / 8 B32 columns per repeat.
    const uint64_t rAlignment = TRANSPOSE_BLOCK_BYTES / xBytes;
    const uint64_t rAlign = CeilAlign(r, rAlignment);
    const uint64_t paramAlign = CeilAlign(r * paramBytes, blockBytes) / paramBytes;
    uint64_t aAlignment = commonParams.vlFp32;
    uint64_t tempBytesPerA = rAlign * BFP32_SIZE;
    if ((r == 1 || r == NUM_TWO) && commonParams.colSize >= PACK_OUTPUT_MIN_A) {
        // Align both the packed group length and each 5HD source address.
        aAlignment = TRANSPOSE_ROWS * std::max(TRANSPOSE_ROWS / r, rAlignment);
        tempBytesPerA = r * (BFP32_SIZE + xBytes);
    }
    // FP32 input/centered data, AR output double buffers, shared reduce/RA-output storage,
    // and two slots for each FP32 statistic. Keep this budget consistent with Init.
    const uint64_t bytesPerA = BUFFER_COUNT * (r * BFP32_SIZE + rAlign * xBytes) + tempBytesPerA +
                               NUM_TWO * BUFFER_COUNT * BFP32_SIZE;
    const uint64_t paramsSize = NUM_TWO * paramAlign * paramBytes;
    const uint64_t rowsPerCore = CeilDiv(commonParams.colSize, commonParams.coreNum);
    const uint64_t maxTileA = std::min((commonParams.ubSizePlatForm - paramsSize) / bytesPerA,
                                       TRANSPOSE_ROWS * TRANSPOSE_MAX_REPEATS);
    const uint64_t tileA = std::min(maxTileA / aAlignment * aAlignment, CeilAlign(rowsPerCore, aAlignment));
    if (tileA == 0) {
        return ge::GRAPH_PARAM_INVALID;
    }
    td_.set_a(commonParams.colSize);
    td_.set_aBlockFactor(rowsPerCore);
    td_.set_tileA(tileA);
    td_.set_r(r);
    td_.set_rAlign(rAlign);
    td_.set_paramAlign(paramAlign);
    td_.set_usedCoreNum(CeilDiv(commonParams.colSize, rowsPerCore));
    td_.set_nullptrGamma(commonParams.gammaNullPtr);
    td_.set_nullptrBeta(commonParams.betaNullPtr);
    td_.set_epsilon(commonParams.eps);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4RegBaseTransposeTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_usedCoreNum());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

// Priority 100 belongs to the legacy transpose template; keep both registrations.
REGISTER_OPS_TILING_TEMPLATE(LayerNormV4, LayerNormV4RegBaseTransposeTiling, 125);
} // namespace optiling
