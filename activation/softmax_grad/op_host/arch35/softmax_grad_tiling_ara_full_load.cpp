/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softmax_grad_tiling.h"

using namespace ge;

namespace optiling {
bool SoftmaxGradARATiling::IsCapable()
{
    // a0_为1的场景走AR模板
    OP_CHECK_IF(a0_ == DIM_NUM_ONE,
                OP_LOGI(context_->GetNodeName(),
                        "ARA full load template is not capable. merged shape is (%ld, %ld, %ld).", a1_, r_, a0_),
                return false);
    return true;
}

ge::graphStatus SoftmaxGradARATiling::DoOpTiling()
{
    int64_t a0TileBase = xDtype_ == ge::DT_FLOAT ? vlFp32_ : vlFp16_;

    int64_t factor0 = r_ * DOUBLE_BUFFER * (FLOAT32_BYTES + xDtypeSize_ * CONST_TWO);

    int64_t factorMax = aicoreParams_.ubSize / a0TileBase / (factor0 + FLOAT32_BYTES);

    OP_CHECK_IF(factorMax <= 0,
                OP_LOGI(context_->GetNodeName(),
                        "ARA full load template is not capable. merged shape is (%ld, %ld, %ld), ub size: %ldB, "
                        "tileBase: %ld, ub factor: %ld.",
                        a1_, r_, a0_, aicoreParams_.ubSize, a0TileBase, factorMax),
                return ge::GRAPH_PARAM_INVALID);

    int64_t a0FactorMax;
    int64_t a0Inner;
    int64_t tileA0Len;
    int64_t a0Outer;
    int64_t tileA0Tail;
    int64_t a1Inner;
    int64_t a1Outer;
    int64_t a1Tail;
    int64_t totalTiles;

    if (isDav920x_) {
        // target-soc: flatten paradigm — actively shrink a0 tile to fill more cores (refer to the
        // softmax_v2 ar_small_r tiling flatten paradigm). Treat a1*a0FactorMax as the finest-grained
        // tile count and derive per-core tile via CeilDiv(totalTilesMax, numBlocks).
        a0FactorMax = Ops::Base::CeilDiv(a0_, a0TileBase);
        int64_t totalTilesMax = a1_ * a0FactorMax;
        int64_t a0InnerMax = Ops::Base::CeilDiv(totalTilesMax, static_cast<int64_t>(aicoreParams_.numBlocks));
        a0Inner = a0InnerMax < factorMax ? a0InnerMax : factorMax;
        a0Inner = a0Inner < a0FactorMax ? a0Inner : a0FactorMax;
        tileA0Len = a0Inner * a0TileBase;
        a0Outer = Ops::Base::CeilDiv(a0_, tileA0Len);
        tileA0Tail = a0_ - tileA0Len * (a0Outer - 1);

        // a1: keep a1Inner=1 to preserve full a1 parallelism (aligned with softmax_v2 side).
        a1Inner = 1;
        a1Outer = a1_;
        a1Tail = a1Inner;
        totalTiles = a1Outer * a0Outer;
    } else {
        // non-target-soc: original greedy logic (keep legacy platform behaviour unchanged).
        // 不切r轴，先切a0
        a0FactorMax = Ops::Base::CeilDiv(a0_, a0TileBase);
        a0Inner = factorMax < a0FactorMax ? factorMax : a0FactorMax;
        tileA0Len = a0Inner * a0TileBase;
        a0Outer = Ops::Base::CeilDiv(a0_, tileA0Len);
        tileA0Tail = a0_ - tileA0Len * (a0Outer - 1);

        // 再切a1
        int64_t a1FactorMax = (aicoreParams_.ubSize - tileA0Len * FLOAT32_BYTES) / (tileA0Len * factor0);
        a1Inner = a1_ < a1FactorMax ? a1_ : a1FactorMax;
        a1Outer = Ops::Base::CeilDiv(a1_, a1Inner);
        a1Tail = a1_ - a1Inner * (a1Outer - 1);
        totalTiles = a1Outer * a0Outer;
    }

    // 然后分核
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData_.set_totalTiles(totalTiles);
    tilingData_.set_tilesPerCore(tilesPerCore);
    tilingData_.set_totalRLen(r_);
    tilingData_.set_totalA0Len(a0_);
    tilingData_.set_tileA0Outer(a0Outer);
    tilingData_.set_tileA0Len(tileA0Len);
    tilingData_.set_tileA0Tail(tileA0Tail);
    tilingData_.set_a1Outer(a1Outer);
    tilingData_.set_a1Inner(a1Inner);
    tilingData_.set_a1Tail(a1Tail);

    OP_LOGI(context_->GetNodeName(),
            "Do tiling success, tileA0Len: %ld, a0Outer: %ld, tileA0Tail: %ld, a1Outer: %ld, a1Inner: %ld, a1Tail: "
            "%ld, totalTiles: %ld, tilesPerCore: %ld, usedCoreNums_:%ld",
            tileA0Len, a0Outer, tileA0Tail, a1Outer, a1Inner, a1Tail, totalTiles, tilesPerCore, usedCoreNums_);

    return ge::GRAPH_SUCCESS;
}

uint64_t SoftmaxGradARATiling::GetTilingKey() const { return TILINGKEY_ARA; }

ge::graphStatus SoftmaxGradARATiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxGrad, SoftmaxGradARATiling, TEMPLATE_ARA_FULL_LOAD_PRIORITY);
} // namespace optiling
