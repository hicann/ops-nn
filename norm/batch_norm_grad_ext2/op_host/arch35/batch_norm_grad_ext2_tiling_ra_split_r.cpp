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
 * \file batch_norm_grad_ext2_tiling_ra_split_r.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "batch_norm_grad_ext2_tiling_ra_split_r.h"

using namespace AscendC;

namespace optiling {

constexpr uint64_t STAGE0_R_ELEM_NUM = 4; // mainX, foldX, mainDy, foldDy
constexpr uint64_t STAGE0_A_ELEM_NUM = 4; // mean, rstd, dbeta, dgamma
constexpr uint64_t STAGE2_A_ELEM_NUM = 5; // mean, rstd, dbeta, dgamma, gamma
constexpr uint64_t STAGE2_R_ELEM_NUM = 3; // dy, x, dx
constexpr uint64_t DOUBLE_BUFF = 2;
constexpr uint64_t WORKSPACE_NUM = 2;
constexpr uint64_t ULONG_BIT_LEN = 64;
constexpr int64_t R_LOOP_FACTOR = 64;
constexpr int64_t LIMIT_ADIM_FACTOR = 16;
constexpr uint64_t FLOAT_BYTE_SIZE = sizeof(float);
constexpr uint64_t BNG_RA_SPLIT_R_TK_BASE = 11000000;
constexpr uint64_t BNG_RA_SPLIT_R_TILING_KEY = 50000000;
constexpr size_t BNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;
constexpr int64_t CONST_ONE = 1;
constexpr int64_t CONST_FUOR = 4;
constexpr int64_t RECOMPUTE_OR_SPLITR_SHAPE = 1024;

bool BatchNormGradExt2TilingRASplitR::IsCapable()
{
    if (r0Dim != 1 || r1Dim < R_LOOP_FACTOR || r1Dim < aDim) {
        return false;
    }

    // 当r1Dim大于1024且aDim小于1024，且r1Dim大于aDim的4倍时，使能此模板。
    if (r1Dim > RECOMPUTE_OR_SPLITR_SHAPE && aDim < RECOMPUTE_OR_SPLITR_SHAPE && r1Dim > aDim * CONST_FUOR) {
        return true;
    }

    return false;
}

ge::graphStatus BatchNormGradExt2TilingRASplitR::DoOpTiling()
{
    dyTypeSize_ = ge::GetSizeByDataType(dyDtype);
    int64_t blockFactor = Ops::Base::CeilDiv(r1Dim, static_cast<int64_t>(coreNum));
    blockFactor_ = std::max(R_LOOP_FACTOR, blockFactor);
    usedCoreNum_ = Ops::Base::CeilDiv(r1Dim, blockFactor_);
    tailBlockFactor_ = (r1Dim % blockFactor_ == 0) ? blockFactor_ : r1Dim % blockFactor_;
    aFactor_ = std::min(Ops::Base::GetVRegSize(context_) / dyTypeSize_, aDim);
    aFactorAlign_ = Ops::Base::CeilAlign(aFactor_, static_cast<int64_t>(blockSize / dyTypeSize_));
    aLoopTimes_ = Ops::Base::CeilDiv(aDim, aFactorAlign_);
    aFactorTail_ = (aDim % aFactorAlign_ == 0) ? aFactorAlign_ : aDim % aFactorAlign_;

    OP_CHECK_IF(ge::GRAPH_SUCCESS != Stage0Stage1UbTiling(),
                OP_LOGE_WITHOUT_REPORT(context_, "failed Stage0Stage1UbTiling."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != Stage2UbTiling(), OP_LOGE_WITHOUT_REPORT(context_, "failed Stage2UbTiling."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradExt2TilingRASplitR::Stage0Stage1UbTiling()
{
    // 一次计算一个tiling块
    rLoopFactor_ = std::min(R_LOOP_FACTOR, blockFactor_);
    binaryBlockCnt_ = Ops::Base::CeilDiv(blockFactor_, rLoopFactor_);
    binaryFoldPoint_ = (binaryBlockCnt_ <= 1) ? 1 : 1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binaryBlockCnt_ - 1));
    cacheBuffCnt_ = ULONG_BIT_LEN - __builtin_clzl(binaryBlockCnt_);
    binaryBlockTail_ = (blockFactor_ % rLoopFactor_) == 0 ? rLoopFactor_ : blockFactor_ % rLoopFactor_;
    lastCoreBlockCnt_ = Ops::Base::CeilDiv(tailBlockFactor_, rLoopFactor_);
    lastCoreFoldPoint_ = (lastCoreBlockCnt_ <= 1) ? 1 :
                                                    1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(lastCoreBlockCnt_ - 1));
    lastCoreLoopTail_ = (tailBlockFactor_ % rLoopFactor_) == 0 ? rLoopFactor_ : tailBlockFactor_ % rLoopFactor_;

    // 校验UB是否越界
    uint64_t rElemUbSize = Ops::Base::CeilAlign(
        aFactorAlign_ * rLoopFactor_ * STAGE0_R_ELEM_NUM * dyTypeSize_ * DOUBLE_BUFF, blockSize / dyTypeSize_);
    uint64_t cacheBuffSize = Ops::Base::CeilAlign(cacheBuffCnt_ * aFactorAlign_ * FLOAT_BYTE_SIZE,
                                                  blockSize / FLOAT_BYTE_SIZE);
    uint64_t aElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE0_A_ELEM_NUM * DOUBLE_BUFF * FLOAT_BYTE_SIZE,
                                                blockSize / FLOAT_BYTE_SIZE);
    uint64_t oneStepUbSize = rElemUbSize + cacheBuffSize + aElemUbSize;
    OP_CHECK_IF(ubSize < oneStepUbSize,
                OP_LOGE_WITHOUT_REPORT(context_, "ubSize %lu less than oneStepUbSize: %lu.", ubSize, oneStepUbSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradExt2TilingRASplitR::Stage2UbTiling()
{
    uint64_t aElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE2_A_ELEM_NUM * FLOAT_BYTE_SIZE * DOUBLE_BUFF,
                                                blockSize / FLOAT_BYTE_SIZE);
    uint64_t rElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE2_R_ELEM_NUM * dyTypeSize_ * DOUBLE_BUFF,
                                                blockSize / dyTypeSize_);
    OP_CHECK_IF(
        ubSize < aElemUbSize + rElemUbSize,
        OP_LOGE_WITHOUT_REPORT(context_, "ubSize %lu less than oneTileUbSize: %lu.", ubSize, aElemUbSize + rElemUbSize),
        return ge::GRAPH_FAILED);
    int64_t dxLoopFactor = Ops::Base::FloorDiv(ubSize - aElemUbSize, rElemUbSize);
    dxLoopFactor_ = std::min(blockFactor_, dxLoopFactor);
    dxLoopTimes_ = Ops::Base::CeilDiv(blockFactor_, dxLoopFactor_),
    dxLoopTail_ = (blockFactor_ % dxLoopFactor_ == 0) ? dxLoopFactor_ : blockFactor_ % dxLoopFactor_;
    dxLastCoreFactor_ = std::min(tailBlockFactor_, dxLoopFactor);
    dxLastCoreTimes_ = Ops::Base::CeilDiv(tailBlockFactor_, dxLastCoreFactor_),
    dxLastCoreTail_ = (tailBlockFactor_ % dxLastCoreFactor_ == 0) ? dxLastCoreFactor_ :
                                                                    tailBlockFactor_ % dxLastCoreFactor_;
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradExt2TilingRASplitR::GetTilingKey() const { return BNG_RA_SPLIT_R_TILING_KEY; }

ge::graphStatus BatchNormGradExt2TilingRASplitR::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED + usedCoreNum_ * aDim * FLOAT_BYTE_SIZE * WORKSPACE_NUM;
    OP_LOGI(context_->GetNodeName(), "Workspace size: %ld", workspaceSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradExt2TilingRASplitR::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "BatchNormGradExt2TilingRASplitR tilingData: useCoreNum is %ld, rDim is %ld, aDim is %ld, "
            "blockFactor is %ld, tailBlockFactor %ld, rLoopFactor is %ld, binaryBlockCnt is %ld, "
            "binaryFoldPoint is %ld, binaryBlockTail is %ld, lastCoreBlockCnt is %ld, lastCoreFoldPoint is %ld, "
            "lastCoreLoopTail is %ld, aFactor %ld, aFactorAlign is %ld, aFactorTail is %ld, aLoopTimes is %ld, "
            "dxLoopFactor %ld, dxLoopTail is %ld, dxLoopTimes %ld, dxLastCoreFactor %ld, dxLastCoreTail is %ld, "
            "dxLastCoreTimes %ld, cacheBuffCnt is %ld, tilingKey is %ld",
            usedCoreNum_, tilingData_.rDim, tilingData_.aDim, tilingData_.blockFactor, tilingData_.tailBlockFactor,
            tilingData_.rLoopFactor, tilingData_.binaryBlockCnt, tilingData_.binaryFoldPoint,
            tilingData_.binaryBlockTail, tilingData_.lastCoreBlockCnt, tilingData_.lastCoreFoldPoint,
            tilingData_.lastCoreLoopTail, tilingData_.aFactor, tilingData_.aFactorAlign, tilingData_.aFactorTail,
            tilingData_.aLoopTimes, tilingData_.dxLoopFactor, tilingData_.dxLoopTail, tilingData_.dxLoopTimes,
            tilingData_.dxLastCoreFactor, tilingData_.dxLastCoreTail, tilingData_.dxLastCoreTimes,
            tilingData_.cacheBuffCnt, GetTilingKey());
    return;
}

ge::graphStatus BatchNormGradExt2TilingRASplitR::PostTiling()
{
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.rDim = r1Dim;
    tilingData_.aDim = aDim;
    tilingData_.blockFactor = blockFactor_;
    tilingData_.tailBlockFactor = tailBlockFactor_;
    tilingData_.rLoopFactor = rLoopFactor_;
    tilingData_.binaryBlockCnt = binaryBlockCnt_;
    tilingData_.binaryFoldPoint = binaryFoldPoint_;
    tilingData_.binaryBlockTail = binaryBlockTail_;
    tilingData_.lastCoreBlockCnt = lastCoreBlockCnt_;
    tilingData_.lastCoreFoldPoint = lastCoreFoldPoint_;
    tilingData_.lastCoreLoopTail = lastCoreLoopTail_;
    tilingData_.aFactor = aFactor_;
    tilingData_.aFactorAlign = aFactorAlign_;
    tilingData_.aFactorTail = aFactorTail_;
    tilingData_.aLoopTimes = aLoopTimes_;
    tilingData_.dxLoopFactor = dxLoopFactor_;
    tilingData_.dxLoopTail = dxLoopTail_;
    tilingData_.dxLoopTimes = dxLoopTimes_;
    tilingData_.dxLastCoreFactor = dxLastCoreFactor_;
    tilingData_.dxLastCoreTail = dxLastCoreTail_;
    tilingData_.dxLastCoreTimes = dxLastCoreTimes_;
    tilingData_.cacheBuffCnt = cacheBuffCnt_;

    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(CONST_ONE);
    context_->SetBlockDim(usedCoreNum_);
    auto* tilingDataOut = context_->GetTilingData<std::decay_t<decltype(tilingData_)>>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = tilingData_;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGradExt2, BatchNormGradExt2TilingRASplitR, BNG_RA_SPLIT_R_TK_BASE);

} // namespace optiling
