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
 * \file layer_norm_grad_grouped_reduce_big_m_tiling.cc
 * \brief
 */

#include <iostream>
#include <vector>
#include "layer_norm_grad_tiling.h"
namespace optiling
{
constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_FIVE = 5;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_EIGHT = 8;
constexpr static int64_t CONST_NINE = 9;
constexpr static int64_t MAX_CORE_NUM = 64;


bool LayerNormGradGroupedReduceBigMTiling::IsCapable()
{
    constexpr static int64_t ROW_THRESHOLD_MAX = 4096;
    constexpr static int64_t COL_THRESHOLD_MIN = 528;
    if (commonParams.rowSize > ROW_THRESHOLD_MAX && commonParams.colSize < COL_THRESHOLD_MIN) {
        return true;
    }
    return false;
}

ge::graphStatus LayerNormGradGroupedReduceBigMTiling::GammaBetaKernelTiling()
{
    constexpr static int64_t gammaBetaMfactor = 64;
    constexpr static int64_t gammaBetaNfactor = 64;

    int64_t row = static_cast<int64_t>(commonParams.rowSize);
    int64_t col = static_cast<int64_t>(commonParams.colSize);
    int64_t blockNum = static_cast<int64_t>(commonParams.coreNum);
    // 受二分累加存活空间计算限制
    blockNum = std::min(MAX_CORE_NUM, blockNum);
    int64_t blocksNeeded = ops::CeilDiv(row, gammaBetaMfactor);
    int64_t usableBlocks = std::min(blocksNeeded, blockNum);

    int64_t mPerBlock = ops::FloorDiv(row, usableBlocks);
    int64_t remainder = row - usableBlocks * mPerBlock;

    int64_t nLoop = ops::FloorDiv(col, gammaBetaNfactor);
    int64_t nTail = col - nLoop * gammaBetaNfactor;

    int64_t nFactorBlockAligned = ops::CeilAlign(static_cast<int64_t>(gammaBetaNfactor * sizeof(float)),
                                                 static_cast<int64_t>(commonParams.blockSize)) /
                                  sizeof(float);
    int64_t mFactorBlockAligned = ops::CeilAlign(static_cast<int64_t>(gammaBetaMfactor * sizeof(float)),
                                                 static_cast<int64_t>(commonParams.blockSize)) /
                                  sizeof(float);

    int64_t mToProcessMainBlock = mPerBlock + 1;
    int64_t mToProcessTailBlock = mPerBlock;

    int64_t mLoopMainBlock = ops::FloorDiv(mToProcessMainBlock, gammaBetaMfactor);
    int64_t mTotalLoopMainBlock = ops::CeilDiv(mToProcessMainBlock, gammaBetaMfactor);
    int64_t mTailMainBlock = mToProcessMainBlock - mLoopMainBlock * gammaBetaMfactor;
    int64_t basicBlockLoopMainBlock = FindNearestPower2(mTotalLoopMainBlock);
    int64_t mainFoldCountMainBlock = mLoopMainBlock - basicBlockLoopMainBlock;

    int64_t cacheBufferCountMainBlock = 1;
    int64_t resultCacheIDMainBlock = 0;
    if (basicBlockLoopMainBlock != 0) {
        cacheBufferCountMainBlock = ULONG_BIT_LEN - static_cast<int64_t>(__builtin_clzl(static_cast<uint64_t>(basicBlockLoopMainBlock)));
        resultCacheIDMainBlock = GetCacheID(basicBlockLoopMainBlock - 1);
    }

    int64_t mLoopTailBlock = ops::FloorDiv(mToProcessTailBlock, gammaBetaMfactor);
    int64_t mTotalLoopTailBlock = ops::CeilDiv(mToProcessTailBlock, gammaBetaMfactor);
    int64_t mTailTailBlock = mToProcessTailBlock - mLoopTailBlock * gammaBetaMfactor;
    int64_t basicBlockLoopTailBlock = FindNearestPower2(mTotalLoopTailBlock);
    int64_t mainFoldCountTailBlock = mLoopTailBlock - basicBlockLoopTailBlock;
    int64_t cacheBufferCountTailBlock = 1;
    int64_t resultCacheIDTailBlock = 0;
    if (basicBlockLoopTailBlock != 0) {
        cacheBufferCountTailBlock = ULONG_BIT_LEN - static_cast<int64_t>(__builtin_clzl(static_cast<uint64_t>(basicBlockLoopTailBlock)));
        resultCacheIDTailBlock = GetCacheID(basicBlockLoopTailBlock - 1);
    }

    int64_t ubSizeNeed = (gammaBetaMfactor * gammaBetaNfactor * CONST_SIX + gammaBetaMfactor * CONST_TWO +
                          gammaBetaNfactor * CONST_THREE + gammaBetaNfactor * cacheBufferCountMainBlock * CONST_TWO) *
                         sizeof(float);
    OP_TILING_CHECK(ubSizeNeed > static_cast<int64_t>(commonParams.ubSizePlatForm),
                    OP_LOGI(context_->GetNodeName(),
                            "Big M template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
                            "ubSizeNeed: %ldB for gamma beta compute.",
                            commonParams.rowSize, commonParams.colSize, commonParams.ubSizePlatForm, ubSizeNeed),
                    return ge::GRAPH_PARAM_INVALID);

    // core 0做核间结果二分
    int64_t mLoopStg2 = ops::FloorDiv(usableBlocks, gammaBetaMfactor);
    int64_t mTotalLoopStg2 = ops::CeilDiv(usableBlocks, gammaBetaMfactor);
    int64_t mTailStg2 = usableBlocks - mLoopStg2 * gammaBetaMfactor;
    int64_t mBasicBlockLoopStg2 = FindNearestPower2(mTotalLoopStg2);
    int64_t mMainFoldCountStg2 = mTotalLoopStg2 - mBasicBlockLoopStg2;
    int64_t mResultCacheIDStg2 = 0;
    if (mBasicBlockLoopStg2 != 0) {
        mResultCacheIDStg2 = GetCacheID(mBasicBlockLoopStg2 - 1);
    }
    // stage2 核间二分计算复用核内二分存活空间大小
    OP_TILING_CHECK(cacheBufferCountMainBlock < mBasicBlockLoopStg2,
                    OP_LOGI(context_->GetNodeName(),
                            "Big M template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
                            "betaGammaCacheBufferCount: %ld, betaGammaCacheBufferCountStg2: %ld.",
                            commonParams.rowSize, commonParams.colSize, commonParams.ubSizePlatForm, cacheBufferCountMainBlock,
                            mBasicBlockLoopStg2),
                    return ge::GRAPH_PARAM_INVALID);

    td_.set_gammaBetaUsableBlocks(usableBlocks);
    td_.set_gammaBetaMPerBlock(mPerBlock);
    td_.set_gammaBetaMReminder(remainder);
    td_.set_gammaBetaNloop(nLoop);
    td_.set_gammaBetaNtail(nTail);
    td_.set_gammaBetaMfactorBlockAligned(mFactorBlockAligned);
    td_.set_gammaBetaNfactorBlockAligned(nFactorBlockAligned);

    td_.set_gammabetaMToProcessMainBlock(mToProcessMainBlock);
    td_.set_gammabetaMLoopMainBlock(mLoopMainBlock);
    td_.set_gammabetaMTotalLoopMainBlock(mTotalLoopMainBlock);
    td_.set_gammabetaMTailMainBlock(mTailMainBlock);
    td_.set_gammabetaBasicBlockLoopMainBlock(basicBlockLoopMainBlock);
    td_.set_gammabetaMainFoldCountMainBlock(mainFoldCountMainBlock);
    td_.set_gammabetaCacheBufferCountMainBlock(cacheBufferCountMainBlock);
    td_.set_gammabetaResultCacheIDMainBlock(resultCacheIDMainBlock);

    td_.set_gammabetaMToProcessTailBlock(mToProcessTailBlock);
    td_.set_gammabetaMLoopTailBlock(mLoopTailBlock);
    td_.set_gammabetaMTotalLoopTailBlock(mTotalLoopTailBlock);
    td_.set_gammabetaMTailTailBlock(mTailTailBlock);
    td_.set_gammabetaBasicBlockLoopTailBlock(basicBlockLoopTailBlock);
    td_.set_gammabetaMainFoldCountTailBlock(mainFoldCountTailBlock);
    td_.set_gammabetaCacheBufferCountTailBlock(cacheBufferCountTailBlock);
    td_.set_gammabetaResultCacheIDTailBlock(resultCacheIDTailBlock);

    td_.set_gammaBetaMTailStg2(mTailStg2);
    td_.set_gammaBetaMBasicBlockLoopStg2(mBasicBlockLoopStg2);
    td_.set_gammaBetaMMainFoldCountStg2(mMainFoldCountStg2);
    td_.set_gammaBetaMResultCacheIDStg2(mResultCacheIDStg2);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradGroupedReduceBigMTiling::BackwardKernelTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradGroupedReduceBigMTiling::DoOpTiling()
{
    td_.set_row(static_cast<int64_t>(commonParams.rowSize));
    td_.set_col(static_cast<int64_t>(commonParams.colSize));
    td_.set_pdxIsRequire(static_cast<int32_t>(commonParams.pdxIsRequire));
    td_.set_pdgammaIsRequire(static_cast<int32_t>(commonParams.pdgammaIsRequire));
    td_.set_pdbetaIsRequire(static_cast<int32_t>(commonParams.pdbetaIsRequire));
    td_.set_epsilon(commonParams.epsilon);
    ge::graphStatus statusGammaBeta = GammaBetaKernelTiling();
    OP_TILING_CHECK(statusGammaBeta != ge::GRAPH_SUCCESS,,
            return statusGammaBeta);
    ge::graphStatus statusBackward = BackwardKernelTiling();
    OP_TILING_CHECK(statusBackward != ge::GRAPH_SUCCESS,,
            return statusBackward);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradGroupedReduceBigMTiling::GetWorkspaceSize()
{
    constexpr int64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
    constexpr int64_t NUM_TWO = 2;
    int64_t wsSize = commonParams.coreNum * commonParams.colSize * NUM_TWO * sizeof(float) + DEFAULT_WORKSPACE_SIZE;
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(wsSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradGroupedReduceBigMTiling::PostTiling()
{
    int64_t numBlocks = commonParams.coreNum;
    context_->SetBlockDim(numBlocks);
    context_->SetScheduleMode(1); // Set to batch mode, all cores start simultaneously
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradGroupedReduceBigMTiling::GetTilingKey() const
{
    constexpr uint64_t LNG_BIGM_TILINGKEY = 600;
    return LNG_BIGM_TILINGKEY;
}

REGISTER_TILING_TEMPLATE("LayerNormGrad", LayerNormGradGroupedReduceBigMTiling, 5000);
} // namespace optiling
