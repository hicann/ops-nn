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
 * \file softmax_v2_ara_full_load_tiling.cpp
 * \brief
 */

#include "softmax_v2_tiling.h"
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"

using namespace ge;

using namespace Ops::Base;
namespace optiling
{

bool SoftmaxV2ARATiling::IsCapable()
{
    // a0_为1的场景走AR模板
    OP_CHECK_IF(a0_ == DIM_NUM_ONE,
                    OP_LOGI(context_->GetNodeName(),
                            "ARA full load template is not capable. merged shape is (%ld, %ld, %ld).", a1_, r_, a0_),
                    return false);
    return true;
}

ge::graphStatus SoftmaxV2ARATiling::BinaryAddTiling()
{
    binaryAddQuotient_ = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r_ - 1)));
    int64_t binaryAddNum = binaryAddQuotient_ / CONST_EIGHT;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= SCALE_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum * SCALE_COEF_TWO) {
        binaryAddK -= 1;
        binaryAddLast = 1;
    } else if (curBinaryAddNum != binaryAddNum) {
        OP_LOGE(context_->GetNodeName(), "Binary add calculate error.");
        return ge::GRAPH_FAILED;
    }

    uint16_t remainderLoopCount = (r_ - binaryAddQuotient_ + SCALE_COEF_EIGHT - 1) / SCALE_COEF_EIGHT;
    uint16_t quotientLoopCount = (binaryAddQuotient_ / SCALE_COEF_EIGHT) - remainderLoopCount;
    uint32_t remainderOffset = binaryAddQuotient_ * tileA0Len_;
    uint32_t baseLineOffset = SCALE_COEF_EIGHT * tileA0Len_;
    uint16_t binaryAddInnerLoop = binaryAddQuotient_ / SCALE_COEF_EIGHT;
    uint32_t validNumInXUb = r_ * tileA0Len_;
    uint16_t remainderTailCount = (r_ - binaryAddQuotient_) - (remainderLoopCount - 1) * SCALE_COEF_EIGHT;
    uint32_t quotientTailOffset = (remainderLoopCount - 1) * baseLineOffset;
    uint32_t remainderTailOffset = quotientTailOffset + remainderOffset;
    uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderTailOffset;
    uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderTailOffset + tileA0Len_;
    uint32_t remainderTailOffset2 =
        (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_TWO_OFFSET * tileA0Len_;
    uint32_t remainderTailOffset3 =
        (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_THREE_OFFSET * tileA0Len_;
    uint32_t remainderTailOffset4 =
        (ROW_FOUR > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FOUR_OFFSET * tileA0Len_;
    uint32_t remainderTailOffset5 =
        (ROW_FIVE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FIVE_OFFSET * tileA0Len_;
    uint32_t remainderTailOffset6 =
        (ROW_SIX > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SIX_OFFSET * tileA0Len_;
    uint32_t remainderTailOffset7 =
        (ROW_SEVEN > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SEVEN_OFFSET * tileA0Len_;

    tilingData_.set_binaryAddK(binaryAddK);
    tilingData_.set_binaryAddLast(binaryAddLast);
    tilingData_.set_remainderLoopCount(remainderLoopCount);
    tilingData_.set_quotientLoopCount(quotientLoopCount);
    tilingData_.set_remainderOffset(remainderOffset);
    tilingData_.set_baseLineOffset(baseLineOffset);
    tilingData_.set_binaryAddInnerLoop(binaryAddInnerLoop);
    tilingData_.set_validNumInXUb(validNumInXUb);
    tilingData_.set_remainderTailCount(remainderTailCount);
    tilingData_.set_quotientTailOffset(quotientTailOffset);
    tilingData_.set_remainderTailOffset(remainderTailOffset);
    tilingData_.set_remainderTailOffset0(remainderTailOffset0);
    tilingData_.set_remainderTailOffset1(remainderTailOffset1);
    tilingData_.set_remainderTailOffset2(remainderTailOffset2);
    tilingData_.set_remainderTailOffset3(remainderTailOffset3);
    tilingData_.set_remainderTailOffset4(remainderTailOffset4);
    tilingData_.set_remainderTailOffset5(remainderTailOffset5);
    tilingData_.set_remainderTailOffset6(remainderTailOffset6);
    tilingData_.set_remainderTailOffset7(remainderTailOffset7);

    OP_LOGI(context_->GetNodeName(), "Binary add K: %ld, binary add quotient: %ld, binary addLast: %d", binaryAddK,
            binaryAddQuotient_, curBinaryAddNum == binaryAddNum ? 0 : 1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxV2ARATiling::DoOpTiling()
{
    a0TileBase_ = xDtype_ == ge::DT_FLOAT ? vlFp32_ : vlFp16_;

    // 切a0, 尽量占多核
    int64_t factorMax = aicoreParams_.ubSize /
                        ((xDtypeSize_ * DOUBLE_BUFFER + FLOAT32_BYTES * DOUBLE_BUFFER + FLOAT32_BYTES) * r_ +
                         FLOAT32_BYTES + FLOAT32_BYTES) /
                        a0TileBase_;
    OP_CHECK_IF(factorMax <= 0,
                    OP_LOGI(context_->GetNodeName(),
                            "ARA full load template is not capable. merged shape is (%ld, %ld, %ld), ub size: %ldB, "
                            "tileBase: %ld, ub factor: %ld.",
                            a1_, r_, a0_, aicoreParams_.ubSize, a0TileBase_, factorMax),
                    return ge::GRAPH_PARAM_INVALID);

    int64_t a0FactorMax = CeilDiv(a0_, a0TileBase_);
    int64_t totalTilesMax = a1_ * a0FactorMax;
    int64_t a0InnerMax = CeilDiv(totalTilesMax, static_cast<int64_t>(aicoreParams_.blockDim));
    int64_t a0Inner = a0InnerMax < factorMax ? a0InnerMax : factorMax;
    a0Inner = a0Inner < a0FactorMax ? a0Inner : a0FactorMax;

    tileA0Len_ = a0Inner * a0TileBase_;
    int64_t a0Outer = CeilDiv(a0_, tileA0Len_);
    int64_t tileA0Tail_ = a0_ - tileA0Len_ * (a0Outer - 1);

    int64_t a1Inner = 1;

    // 分核
    totalTiles_ = a1_ * a0Outer;
    tilesPerCore_ = CeilDiv(totalTiles_, static_cast<int64_t>(aicoreParams_.blockDim));
    usedCoreNums_ = CeilDiv(totalTiles_, tilesPerCore_);

    tilingData_.set_totalTiles(totalTiles_);
    tilingData_.set_tilesPerCore(tilesPerCore_);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_totalA1Len(a1_);
    tilingData_.set_totalRLen(r_);
    tilingData_.set_totalA0Len(a0_);
    tilingData_.set_a1Outer(a1_);
    tilingData_.set_a0Outer(a0Outer);
    tilingData_.set_tileA1Len(a1Inner);
    tilingData_.set_tileA1Tail(a1_);
    tilingData_.set_tileA0Len(tileA0Len_);
    tilingData_.set_tileA0Tail(tileA0Tail_);

    if (r_ <= CONST_EIGHT) {
        return ge::GRAPH_SUCCESS;
    }

    return BinaryAddTiling();
}

uint64_t SoftmaxV2ARATiling::GetTilingKey() const
{
    return TILINGKEY_ARA;
}

ge::graphStatus SoftmaxV2ARATiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxV2, SoftmaxV2ARATiling, TEMPLATE_ARA_FULL_LOAD_PRIORITY);
}  // namespace optiling
