/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_tiling_simd.cpp
 * \brief SIMD tiling implementation for Embedding operator.
 */
#include "embedding_tiling_simt.h"
#include "embedding_tiling_simd.h"

namespace optiling {

const static int64_t SIMD_BUFFER_NUM = 2;
const static int64_t SIMD_INDICES_SIZE = 8192;

const static uint64_t SIMD_TWO_DIM_TILING_KEY_INT32 = 1000000299UL;
const static uint64_t SIMD_TWO_DIM_TILING_KEY_INT64 = 1000000300UL;
const static uint64_t SIMT_TWO_DIM_BASE_KEY = 2000000000UL;

static bool IsPcieThroughImpl(gert::TilingContext* context)
{
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90200000
    return context->GetPcieThroughFlag();
#else
    return false;
#endif
}

bool EmbeddingTilingBase::IsPcieThrough() { return IsPcieThroughImpl(context_); }

ge::graphStatus EmbeddingTilingBase::DoOpTilingForSimd()
{
    tilingMode_ = TILING_SIMD_TWO_DIM;
    return SimdTwoDimTiling();
}

uint64_t EmbeddingTilingBase::GetTilingKey() const
{
    uint64_t tilingKey = 0UL;
    if (tilingMode_ == TILING_SIMD_TWO_DIM) {
        tilingKey = (indicesDtype_ == ge::DT_INT64) ? SIMD_TWO_DIM_TILING_KEY_INT64 : SIMD_TWO_DIM_TILING_KEY_INT32;
    } else if (tilingMode_ == TILING_SIMT_TWO_DIM) {
        tilingKey = SIMT_TWO_DIM_BASE_KEY + static_cast<uint64_t>(improveDtypeSize_);
    }
    return tilingKey;
}

ge::graphStatus EmbeddingTilingBase::SimdTwoDimTiling()
{
    if (gatherSize_ == 0) {
        needCoreNum_ = 1;
        simdTwoDimTilingData_.set_needCoreNum(static_cast<int16_t>(needCoreNum_));
        simdTwoDimTilingData_.set_indiceFactor(0);
        simdTwoDimTilingData_.set_dtypeSize(improveDtypeSize_);
        simdTwoDimTilingData_.set_gatherDimSize(gatherDimSize_);
        simdTwoDimTilingData_.set_gatherSize(gatherSize_);
        simdTwoDimTilingData_.set_innerSize(innerSize_);
        simdTwoDimTilingData_.set_blockFactor(0);
        simdTwoDimTilingData_.set_tailBlockFactor(0);
        simdTwoDimTilingData_.set_maxElement(0);
        return ge::GRAPH_SUCCESS;
    }
    int64_t blockFactor = gatherSize_ / aivNum_;
    int64_t tailBlockFactor = gatherSize_ - blockFactor * aivNum_;
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t ubAviable = (ubSize_ - SIMD_INDICES_SIZE) / ubBlockSize * ubBlockSize / improveDtypeSize_ / SIMD_BUFFER_NUM;
    int32_t indiceFactor = SIMD_INDICES_SIZE / indicesDtypeSize_;
    needCoreNum_ = blockFactor > 0 ? aivNum_ : tailBlockFactor;

    simdTwoDimTilingData_.set_needCoreNum(static_cast<int16_t>(needCoreNum_));
    simdTwoDimTilingData_.set_indiceFactor(indiceFactor);
    simdTwoDimTilingData_.set_dtypeSize(improveDtypeSize_);
    simdTwoDimTilingData_.set_gatherDimSize(gatherDimSize_);
    simdTwoDimTilingData_.set_gatherSize(gatherSize_);
    simdTwoDimTilingData_.set_innerSize(innerSize_);
    simdTwoDimTilingData_.set_blockFactor(blockFactor);
    simdTwoDimTilingData_.set_tailBlockFactor(tailBlockFactor);
    simdTwoDimTilingData_.set_maxElement(ubAviable);
    return ge::GRAPH_SUCCESS;
}

void EmbeddingTilingBase::ShowSimdTilingData()
{
    OP_LOGI(opName_,
            "simdTwoDimTilingData is needCoreNum: %d, indiceFactor: %d, dtypeSize: %d, gatherDimSize: %ld,"
            "gatherSize: %ld, innerSize: %ld, blockFactor: %ld, tailBlockFactor: %ld, maxElement: %ld",
            simdTwoDimTilingData_.get_needCoreNum(), simdTwoDimTilingData_.get_indiceFactor(),
            simdTwoDimTilingData_.get_dtypeSize(), simdTwoDimTilingData_.get_gatherDimSize(),
            simdTwoDimTilingData_.get_gatherSize(), simdTwoDimTilingData_.get_innerSize(),
            simdTwoDimTilingData_.get_blockFactor(), simdTwoDimTilingData_.get_tailBlockFactor(),
            simdTwoDimTilingData_.get_maxElement());
}

} // namespace optiling
