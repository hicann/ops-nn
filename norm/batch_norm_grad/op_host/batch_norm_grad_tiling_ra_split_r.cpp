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
 * \file batch_norm_grad_tiling_ra_split_r.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "batch_norm_grad_tiling_ra_split_r.h"

using namespace AscendC;

namespace optiling {

constexpr uint64_t STAGE0_R_ELEM_NUM = 4;     // mainX, foldX, mainDy, foldDy
constexpr uint64_t STAGE0_A_ELEM_NUM = 4;     // mean, rstd, dbeta, dgamma
constexpr uint64_t STAGE2_A_ELEM_NUM = 5;     // mean, rstd, dbeta, dgamma, gamma
constexpr uint64_t STAGE2_R_ELEM_NUM = 3;     // dy, x, dx
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

bool BatchNormGradTilingRASplitR::IsCapable()
{
    if (r0Dim != 1 || r1Dim < R_LOOP_FACTOR || r1Dim < aDim) {
        return false;
    }
    // r1Dim大于8192,并且aDim小于1024范围时，调度到切R轴模版
    if ((r1Dim > R_LOOP_FACTOR * coreNum * 2) && (aDim < coreNum * LIMIT_ADIM_FACTOR)) {
        return true;
    }
    return false;
}

ge::graphStatus BatchNormGradTilingRASplitR::DoOpTiling()
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

    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != Stage0Stage1UbTiling(), OP_LOGE(context_, "failed Stage0Stage1UbTiling."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != Stage2UbTiling(), OP_LOGE(context_, "failed Stage2UbTiling."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingRASplitR::Stage0Stage1UbTiling()
{
    // 一次计算一个tiling块
    rLoopFactor_ = std::min(R_LOOP_FACTOR, blockFactor_);
    binaryBlockCnt_ = Ops::Base::CeilDiv(blockFactor_, rLoopFactor_);
    binaryFoldPoint_ = (binaryBlockCnt_ <= 1) ? 1
                       : 1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binaryBlockCnt_ - 1));
    cacheBuffCnt_ = ULONG_BIT_LEN -  __builtin_clzl(binaryBlockCnt_);
    binaryBlockTail_ = (blockFactor_ % rLoopFactor_) == 0 ? rLoopFactor_ : blockFactor_ % rLoopFactor_;
    lastCoreBlockCnt_ = Ops::Base::CeilDiv(tailBlockFactor_, rLoopFactor_);
    lastCoreFoldPoint_ = (lastCoreBlockCnt_ <= 1) ? 1
                       : 1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(lastCoreBlockCnt_ - 1));
    lastCoreLoopTail_ =  (tailBlockFactor_ % rLoopFactor_) == 0 ? rLoopFactor_ : tailBlockFactor_ % rLoopFactor_;

    // 校验UB是否越界
    uint64_t rElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * rLoopFactor_ * STAGE0_R_ELEM_NUM * dyTypeSize_ * DOUBLE_BUFF,
                                          blockSize / dyTypeSize_);
    uint64_t cacheBuffSize =
            Ops::Base::CeilAlign(cacheBuffCnt_ * aFactorAlign_ * FLOAT_BYTE_SIZE, blockSize / FLOAT_BYTE_SIZE);
    uint64_t aElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE0_A_ELEM_NUM  * DOUBLE_BUFF * FLOAT_BYTE_SIZE,
                                          blockSize / FLOAT_BYTE_SIZE);
    uint64_t oneStepUbSize = rElemUbSize + cacheBuffSize + aElemUbSize;
    OP_CHECK_IF(
        ubSize < oneStepUbSize, OP_LOGE(context_, "ubSize %lu less than oneStepUbSize: %lu.", ubSize, oneStepUbSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradTilingRASplitR::Stage2UbTiling()
{
    uint64_t aElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE2_A_ELEM_NUM * FLOAT_BYTE_SIZE * DOUBLE_BUFF,
                                          blockSize / FLOAT_BYTE_SIZE);
    uint64_t rElemUbSize = Ops::Base::CeilAlign(aFactorAlign_ * STAGE2_R_ELEM_NUM * dyTypeSize_ * DOUBLE_BUFF,
                                          blockSize / dyTypeSize_);
    OP_CHECK_IF(
        ubSize < aElemUbSize + rElemUbSize,
        OP_LOGE(context_, "ubSize %lu less than oneTileUbSize: %lu.", ubSize, aElemUbSize + rElemUbSize),
        return ge::GRAPH_FAILED);
    int64_t dxLoopFactor = Ops::Base::FloorDiv(ubSize - aElemUbSize, rElemUbSize);
    dxLoopFactor_ = std::min(blockFactor_, dxLoopFactor);
    dxLoopTimes_ = Ops::Base::CeilDiv(blockFactor_, dxLoopFactor_),
    dxLoopTail_ = (blockFactor_ % dxLoopFactor_ == 0) ? dxLoopFactor_ : blockFactor_ % dxLoopFactor_;
    dxLastCoreFactor_ = std::min(tailBlockFactor_, dxLoopFactor);
    dxLastCoreTimes_ = Ops::Base::CeilDiv(tailBlockFactor_, dxLastCoreFactor_),
    dxLastCoreTail_ = (tailBlockFactor_ % dxLastCoreFactor_ == 0) ? dxLastCoreFactor_ : tailBlockFactor_ % dxLastCoreFactor_;
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradTilingRASplitR::GetTilingKey() const
{
    return BNG_RA_SPLIT_R_TILING_KEY;
}

ge::graphStatus BatchNormGradTilingRASplitR::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED + usedCoreNum_ * aDim * FLOAT_BYTE_SIZE * WORKSPACE_NUM;
    OP_LOGI(context_->GetNodeName(), "Workspace size: %ld", workspaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradTilingRASplitR::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
        "BatchNormGradTilingRASplitR tilingData: useCoreNum is %ld, rDim is %ld, aDim is %ld, "
        "blockFactor is %ld, tailBlockFactor %ld, rLoopFactor is %ld, binaryBlockCnt is %ld, "
        "binaryFoldPoint is %ld, binaryBlockTail is %ld, lastCoreBlockCnt is %ld, lastCoreFoldPoint is %ld, "
        "lastCoreLoopTail is %ld, aFactor %ld, aFactorAlign is %ld, aFactorTail is %ld, aLoopTimes is %ld, "
        "dxLoopFactor %ld, dxLoopTail is %ld, dxLoopTimes %ld, dxLastCoreFactor %ld, dxLastCoreTail is %ld, "
        "dxLastCoreTimes %ld, cacheBuffCnt is %ld, tilingKey is %ld",
        usedCoreNum_,
        tilingData_.get_rDim(),
        tilingData_.get_aDim(),
        tilingData_.get_blockFactor(),
        tilingData_.get_tailBlockFactor(),
        tilingData_.get_rLoopFactor(),
        tilingData_.get_binaryBlockCnt(),
        tilingData_.get_binaryFoldPoint(),
        tilingData_.get_binaryBlockTail(),
        tilingData_.get_lastCoreBlockCnt(),
        tilingData_.get_lastCoreFoldPoint(),
        tilingData_.get_lastCoreLoopTail(),
        tilingData_.get_aFactor(),
        tilingData_.get_aFactorAlign(),
        tilingData_.get_aFactorTail(),
        tilingData_.get_aLoopTimes(),
        tilingData_.get_dxLoopFactor(),
        tilingData_.get_dxLoopTail(),
        tilingData_.get_dxLoopTimes(),
        tilingData_.get_dxLastCoreFactor(),
        tilingData_.get_dxLastCoreTail(),
        tilingData_.get_dxLastCoreTimes(),
        tilingData_.get_cacheBuffCnt(),
        GetTilingKey());
    return;
}

ge::graphStatus BatchNormGradTilingRASplitR::PostTiling()
{
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_rDim(r1Dim);
    tilingData_.set_aDim(aDim);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_tailBlockFactor(tailBlockFactor_);
    tilingData_.set_rLoopFactor(rLoopFactor_);
    tilingData_.set_binaryBlockCnt(binaryBlockCnt_);
    tilingData_.set_binaryFoldPoint(binaryFoldPoint_);
    tilingData_.set_binaryBlockTail(binaryBlockTail_);
    tilingData_.set_lastCoreBlockCnt(lastCoreBlockCnt_);
    tilingData_.set_lastCoreFoldPoint(lastCoreFoldPoint_);
    tilingData_.set_lastCoreLoopTail(lastCoreLoopTail_);
    tilingData_.set_aFactor(aFactor_);
    tilingData_.set_aFactorAlign(aFactorAlign_);
    tilingData_.set_aFactorTail(aFactorTail_);
    tilingData_.set_aLoopTimes(aLoopTimes_);
    tilingData_.set_dxLoopFactor(dxLoopFactor_);
    tilingData_.set_dxLoopTail(dxLoopTail_);
    tilingData_.set_dxLoopTimes(dxLoopTimes_);
    tilingData_.set_dxLastCoreFactor(dxLastCoreFactor_);
    tilingData_.set_dxLastCoreTail(dxLastCoreTail_);
    tilingData_.set_dxLastCoreTimes(dxLastCoreTimes_);
    tilingData_.set_cacheBuffCnt(cacheBuffCnt_);

    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(CONST_ONE);
    context_->SetBlockDim(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradTilingRASplitR, BNG_RA_SPLIT_R_TK_BASE);

} // namespace optiling
