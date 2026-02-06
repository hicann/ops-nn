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

namespace optiling
{

constexpr int64_t DEFAULT_BIN_ADD_R_FACTOR = 128;

bool SoftmaxGradARARecomputeTiling::IsCapable()
{
    // a0_为1的场景走AR模板
    OP_CHECK_IF(a0_ == DIM_NUM_ONE,
                    OP_LOGI(context_->GetNodeName(),
                            "ARA recompute template is not capable. merged shape is (%ld, %ld, %ld).", a1_, r_, a0_),
                    return false);
    return true;
}

int64_t SoftmaxGradARARecomputeTiling::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

ge::graphStatus SoftmaxGradARARecomputeTiling::ComputeBinaryAddParams()
{
    binAddRFactor_ = DEFAULT_BIN_ADD_R_FACTOR;
    binAddRLoop_ = r_ / binAddRFactor_;
    binAddRTotalLoop_ = Ops::Base::CeilDiv(r_, binAddRFactor_);
    binAddRTail_ = r_ - (binAddRTotalLoop_ - 1) * binAddRFactor_;
    // 计算最接近binAddRTotalLoop_的2^k
    binAddBasicBlockLoop_ =
        binAddRTotalLoop_ > 1 ? (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binAddRTotalLoop_ - 1))) : 0;
    mainFoldCount_ = binAddRLoop_ - binAddBasicBlockLoop_;
    if (binAddBasicBlockLoop_ == 0) {
        binAddCacheBufferCount_ = 1;
        binAddResultCacheID_ = 0;
    } else {
        binAddCacheBufferCount_ = ULONG_BIT_LEN - __builtin_clzl(binAddBasicBlockLoop_);
        binAddResultCacheID_ = GetCacheID(binAddBasicBlockLoop_ - 1);
    }

    OP_LOGI(context_->GetNodeName(),
            "Binary add rFactor: %ld, rLoop: %ld, binAddRTotalLoop_: %ld, rTail: %ld, basicBlockLoop: %ld, "
            "mainFoldCount:%ld,cacheBufferCount: "
            "%ld, resultCacheId:%ld ",
            binAddRFactor_, binAddRLoop_, binAddRTotalLoop_, binAddRTail_, binAddBasicBlockLoop_, mainFoldCount_,
            binAddCacheBufferCount_, binAddResultCacheID_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradARARecomputeTiling::DoOpTiling()
{
    OP_CHECK_IF(
        ComputeBinaryAddParams() != ge::GRAPH_SUCCESS,
        OP_LOGI(context_->GetNodeName(), "ARA recompute template is not capable, compute bianary add tiling not ok."),
        return ge::GRAPH_PARAM_INVALID);

    a0TileBase_ = xDtype_ == ge::DT_FLOAT ? FP32_BLOCK_ALIGN_NUM : FP16_BLOCK_ALIGN_NUM;

    int64_t factorMax = aicoreParams_.ubSize / a0TileBase_ /
                        (binAddRFactor_ * DOUBLE_BUFFER * (FLOAT32_BYTES + xDtypeSize_ * CONST_TWO) +
                         FLOAT32_BYTES * binAddCacheBufferCount_ + FLOAT32_BYTES);

    OP_CHECK_IF(factorMax <= 0,
                    OP_LOGI(context_->GetNodeName(),
                            "ARA recompute template is not capable. merged shape is (%ld, %ld, %ld), ub size: %ldB, "
                            "tileBase: %ld, ub factor: %ld.",
                            a1_, r_, a0_, aicoreParams_.ubSize, a0TileBase_, factorMax),
                    return ge::GRAPH_PARAM_INVALID);

    // 尽量占多核
    int64_t a0FactorMax = Ops::Base::CeilDiv(a0_, a0TileBase_);
    int64_t totalTilesMax = a1_ * a0FactorMax;
    int64_t a0InnerMax = Ops::Base::CeilDiv(totalTilesMax, static_cast<int64_t>(aicoreParams_.numBlocks));
    int64_t a0Inner = a0InnerMax < factorMax ? a0InnerMax : factorMax;
    a0Inner = a0Inner < a0FactorMax ? a0Inner : a0FactorMax;

    tileA0Len_ = a0Inner * a0TileBase_;
    a0Outer_ = Ops::Base::CeilDiv(a0_, tileA0Len_);
    tileA0Tail_ = a0_ - tileA0Len_ * (a0Outer_ - 1);

    // 不切R轴
    totalTiles_ = a1_ * a0Outer_;
    tilesPerCore_ = Ops::Base::CeilDiv(totalTiles_, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles_, tilesPerCore_);

    tilingData_.set_totalTiles(totalTiles_);
    tilingData_.set_tilesPerCore(tilesPerCore_);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_totalRLen(r_);
    tilingData_.set_totalA0Len(a0_);
    tilingData_.set_tileA0Outer(a0Outer_);
    tilingData_.set_tileA0Len(tileA0Len_);
    tilingData_.set_tileA0Tail(tileA0Tail_);

    // binary add
    tilingData_.set_binAddRFactor(binAddRFactor_);
    tilingData_.set_binAddRLoop(binAddRLoop_);
    tilingData_.set_binAddRTotalLoop(binAddRTotalLoop_);
    tilingData_.set_binAddRTail(binAddRTail_);
    tilingData_.set_binAddBasicBlockLoop(binAddBasicBlockLoop_);
    tilingData_.set_binAddMainFoldCount(mainFoldCount_);
    tilingData_.set_binAddCacheBufferCount(binAddCacheBufferCount_);
    tilingData_.set_binAddResultCacheID(binAddResultCacheID_);

    OP_LOGI(context_->GetNodeName(),
            "Do tiling success, tileA0Len: %ld, a0Outer_: %ld, tileA0Tail_: %ld, totalTiles_: %ld, tilesPerCore_: "
            "%ld, usedCoreNums_:%ld",
            tileA0Len_, a0Outer_, tileA0Tail_, totalTiles_, tilesPerCore_, usedCoreNums_);

    return ge::GRAPH_SUCCESS;
}

uint64_t SoftmaxGradARARecomputeTiling::GetTilingKey() const
{
    return TILINGKEY_ARA_RECOMPUTE;
}

ge::graphStatus SoftmaxGradARARecomputeTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxGrad, SoftmaxGradARARecomputeTiling, TEMPLATE_ARA_RECOMPUTE_PRIORITY);
}  // namespace optiling
