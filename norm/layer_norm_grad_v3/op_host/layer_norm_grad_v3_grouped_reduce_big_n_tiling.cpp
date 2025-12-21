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
 * \file layer_norm_grad_v3_grouped_reduce_big_n_tiling.cc
 * \brief
 */

#include "layer_norm_grad_v3_tiling.h"
namespace optiling
{
constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_SEVEN = 7;
constexpr static int64_t CONST_EIGHT = 8;
constexpr static int64_t CONST_NINE = 9;

constexpr static int64_t GAMMA_BETA_DEFAULT_BIN_ADD_R_FACTOR = 64;
constexpr static int64_t BACKWARD_DEFAULT_BIN_ADD_R_FACTOR_128 = 128;
constexpr static int64_t BACKWARD_DEFAULT_BIN_ADD_R_FACTOR_64 = 64;
constexpr static int64_t BACKWARD_BIG_R_FACTOR_THRES = 16;

bool LayerNormGradV3GroupedReduceBigNTiling::IsCapable()
{
    constexpr static int64_t ROW_THRESHOLD_MIN = 512;
    constexpr static int64_t COL_THRESHOLD_MAX = 8192;
    if (commonParams.rowSize < ROW_THRESHOLD_MIN && commonParams.colSize > COL_THRESHOLD_MAX) {
        return true;
    }
    return false;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::GammaBetaKernelTiling()
{
    // 沿M轴做reduce（输入形状为(M, N)）
    constexpr int64_t mFactor = GAMMA_BETA_DEFAULT_BIN_ADD_R_FACTOR;
    int64_t rowSize = static_cast<int64_t>(commonParams.rowSize);
    int64_t colSize = static_cast<int64_t>(commonParams.colSize);

    // M方向分块参数计算（二分累加）
    int64_t mLoop = ops::FloorDiv(rowSize, mFactor);
    int64_t mTotalLoop = ops::CeilDiv(rowSize, mFactor);
    int64_t mTail = rowSize - mLoop * mFactor;
    int64_t basicBlockLoop = FindNearestPower2(mTotalLoop);
    int64_t mainFoldCount = mLoop - basicBlockLoop;
    int64_t cacheBufferCount = CONST_ONE;
    int64_t resultCacheID = 0;
    if (basicBlockLoop != 0) {
        cacheBufferCount = ULONG_BIT_LEN - __builtin_clzl(basicBlockLoop);
        resultCacheID = GetCacheID(basicBlockLoop - 1);
    }

    // N方向分块参数计算（分核处理）
    int64_t nFactorBase = commonParams.blockSize / HALF_SIZE;
    int64_t mainBlockFactor = ops::CeilDiv(colSize, static_cast<int64_t>(commonParams.coreNum));
    mainBlockFactor = std::max(mainBlockFactor, nFactorBase);
    int64_t blockDim = ops::CeilDiv(colSize, mainBlockFactor);
    int64_t tailBlockFactor = colSize - (blockDim - CONST_ONE) * mainBlockFactor;
    int64_t nFactorMax = (commonParams.ubSizePlatForm / sizeof(float) - mFactor * CONST_TWO) /
                         (mFactor * CONST_SIX + CONST_THREE + cacheBufferCount * CONST_TWO);
    OP_TILING_CHECK(nFactorMax < nFactorBase,
                    OP_LOGI(context_->GetNodeName(),
                            "Big N template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
                            "nFactorMax: %ld.",
                            commonParams.rowSize, commonParams.colSize, commonParams.ubSizePlatForm, nFactorMax),
                    return ge::GRAPH_PARAM_INVALID);

    int64_t nFactor = ops::FloorAlign(nFactorMax, nFactorBase);
    int64_t nLoopMain = ops::CeilDiv(mainBlockFactor, nFactor);
    int64_t nTailMain = mainBlockFactor - (nLoopMain - CONST_ONE) * nFactor;
    int64_t nLoopTail = ops::CeilDiv(tailBlockFactor, nFactor);
    int64_t nTailTail = tailBlockFactor - (nLoopTail - CONST_ONE) * nFactor;

    td_.set_gammaBetaMainBlockFactor(mainBlockFactor);
    td_.set_gammaBetaBlockDim(blockDim);
    td_.set_gammaBetaNloopMainBlock(nLoopMain);
    td_.set_gammaBetaNtailMainBlock(nTailMain);
    td_.set_gammaBetaNloopTailBlock(nLoopTail);
    td_.set_gammaBetaNtailTailBlock(nTailTail);
    td_.set_gammaBetaMtail(mTail);
    td_.set_gammaBetaBasicBlockLoop(basicBlockLoop);
    td_.set_gammaBetaMainFoldCount(mainFoldCount);
    td_.set_gammaBetaCacheBufferCount(cacheBufferCount);
    td_.set_gammaBetaResultCacheID(resultCacheID);
    td_.set_gammaBetaNfactor(nFactor);
    td_.set_gammaBetaMfactor(mFactor);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::BackwardKernelTiling()
{
    // 输入形状为(M, N)
    int64_t colSize = static_cast<int64_t>(commonParams.colSize);
    int64_t rowSize = static_cast<int64_t>(commonParams.rowSize);
    int64_t coreNum = static_cast<int64_t>(commonParams.coreNum);

    // 1. N轴分核
    int64_t nFactor = rowSize > BACKWARD_BIG_R_FACTOR_THRES ? BACKWARD_DEFAULT_BIN_ADD_R_FACTOR_64
                                                            : BACKWARD_DEFAULT_BIN_ADD_R_FACTOR_128;
    // kernel二分计算部分限制核数不超过nfactor
    coreNum = std::min(coreNum, nFactor);
    int64_t mainBlockFactor = ops::CeilDiv(colSize, coreNum);
    mainBlockFactor = std::max(mainBlockFactor, nFactor);
    int64_t blockDim = ops::CeilDiv(colSize, mainBlockFactor);
    int64_t nPerBlock = ops::FloorDiv(colSize, blockDim);
    int64_t nRem = colSize - nPerBlock * blockDim;
    int64_t nToProcessMain = nPerBlock + CONST_ONE;
    int64_t nToProcessTail = nPerBlock;

    // 2. N方向二分累加参数
    // 2.1 主块（Main）的二分累加参数
    int64_t nLoopMain = ops::FloorDiv(nToProcessMain, nFactor);
    int64_t nTotalLoopMain = ops::CeilDiv(nToProcessMain, nFactor);
    int64_t nTailMain = nToProcessMain - nLoopMain * nFactor;
    int64_t basicBlockLoopMain = FindNearestPower2(nTotalLoopMain);
    int64_t mainFoldCountMain = nLoopMain - basicBlockLoopMain;
    int64_t cacheBufferCountMain = CONST_ONE;
    int64_t resultCacheIDMain = 0;
    if (basicBlockLoopMain != 0) {
        cacheBufferCountMain = ULONG_BIT_LEN - __builtin_clzl(basicBlockLoopMain);
        resultCacheIDMain = GetCacheID(basicBlockLoopMain - 1);
    }

    // 2.2 尾部（Tail）的二分累加参数（与主块逻辑完全一致）
    int64_t nLoopTail = ops::FloorDiv(nToProcessTail, nFactor);
    int64_t nTotalLoopTail = ops::CeilDiv(nToProcessTail, nFactor);
    int64_t nTailTail = nToProcessTail - nLoopTail * nFactor;
    int64_t basicBlockLoopTail = FindNearestPower2(nTotalLoopTail);
    int64_t mainFoldCountTail = nLoopTail - basicBlockLoopTail;
    int64_t cacheBufferCountTail = CONST_ONE;
    int64_t resultCacheIDTail = 0;
    if (basicBlockLoopTail != 0) {
        cacheBufferCountTail = ULONG_BIT_LEN - __builtin_clzl(basicBlockLoopTail);
        resultCacheIDTail = GetCacheID(basicBlockLoopTail - CONST_ONE);
    }

    // 3. M方向分块参数
    int64_t mFactorAlignedMax = (commonParams.ubSizePlatForm / sizeof(float) - nFactor * CONST_TWO) /
                                (nFactor * CONST_NINE + CONST_NINE + cacheBufferCountMain * CONST_TWO);
    mFactorAlignedMax =
        ops::FloorAlign(mFactorAlignedMax, static_cast<int64_t>(commonParams.blockSize / sizeof(float)));
    OP_TILING_CHECK(mFactorAlignedMax <= 0,
                    OP_LOGI(context_->GetNodeName(),
                            "Big N template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
                            "backwardMfactorAlignedMax: %ld.",
                            commonParams.rowSize, commonParams.colSize, commonParams.ubSizePlatForm, mFactorAlignedMax),
                    return ge::GRAPH_PARAM_INVALID);

    int64_t mFactor = (rowSize < mFactorAlignedMax) ? rowSize : mFactorAlignedMax;
    int64_t mFactorAligned = ops::CeilAlign(mFactor, static_cast<int64_t>(commonParams.blockSize / sizeof(float)));
    int64_t mTotalLoop = ops::CeilDiv(rowSize, mFactor);
    int64_t mTail = rowSize - (mTotalLoop - CONST_ONE) * mFactor;

    // 4. 保存所有参数到td_结构
    td_.set_backwardBlockDim(blockDim);
    td_.set_backwardNPerBlock(nPerBlock);
    td_.set_backwardNRem(nRem);
    td_.set_nToProcessMain(nToProcessMain);
    td_.set_nToProcessTail(nToProcessTail);
    td_.set_backwardMTotalLoop(mTotalLoop);
    td_.set_backwardMtail(mTail);
    td_.set_backwardNloopMain(nLoopMain);
    td_.set_backwardNtailMain(nTailMain);
    td_.set_backwardBasicBlockLoopMain(basicBlockLoopMain);
    td_.set_backwardMainFoldCountMain(mainFoldCountMain);
    td_.set_backwardNfactorBlockAligned(nFactor);
    td_.set_backwardMfactor(mFactor);
    td_.set_backwardMfactorBlockAligned(mFactorAligned);
    td_.set_backwardCacheBufferCountMain(cacheBufferCountMain);
    td_.set_backwardResultCacheIDMain(resultCacheIDMain);
    td_.set_backwardNloopTail(nLoopTail);
    td_.set_backwardNtailTail(nTailTail);
    td_.set_backwardBasicBlockLoopTail(basicBlockLoopTail);
    td_.set_backwardMainFoldCountTail(mainFoldCountTail);
    td_.set_backwardCacheBufferCountTail(cacheBufferCountTail);
    td_.set_backwardResultCacheIDTail(resultCacheIDTail);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::DoOpTiling()
{
    td_.set_row(static_cast<int64_t>(commonParams.rowSize));
    td_.set_col(static_cast<int64_t>(commonParams.colSize));
    td_.set_pdxIsRequire(static_cast<int32_t>(commonParams.pdxIsRequire));
    td_.set_pdgammaIsRequire(static_cast<int32_t>(commonParams.pdgammaIsRequire));
    td_.set_pdbetaIsRequire(static_cast<int32_t>(commonParams.pdbetaIsRequire));
    ge::graphStatus statusGammaBeta = GammaBetaKernelTiling();
    OP_TILING_CHECK(statusGammaBeta != ge::GRAPH_SUCCESS, , return statusGammaBeta);
    ge::graphStatus statusBackward = BackwardKernelTiling();
    OP_TILING_CHECK(statusBackward != ge::GRAPH_SUCCESS, , return statusBackward);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t ascendcWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024;
    constexpr int64_t NUM_THREE = 3;
    int64_t wsSize = commonParams.coreNum * commonParams.rowSize * NUM_THREE + DEFAULT_WORKSPACE_SIZE;
    wsSize += ascendcWorkspaceSize;
    size_t* workspaces = context_->GetWorkspaceSizes(CONST_ONE);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = wsSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3GroupedReduceBigNTiling::PostTiling()
{
    int64_t blockDim = commonParams.coreNum;
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(CONST_ONE);  // Set to batch mode, all cores start simultaneously
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradV3GroupedReduceBigNTiling::GetTilingKey() const
{
    constexpr static uint64_t bigNTilingKey = 700;
    return bigNTilingKey;
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3GroupedReduceBigNTiling, 6000);
}  // namespace optiling
