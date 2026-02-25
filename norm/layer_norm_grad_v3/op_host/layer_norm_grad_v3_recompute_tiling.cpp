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
 * \file layer_norm_grad_v3_recompute_tiling.cc
 * \brief
 */

#include <iostream>
#include <vector>
#include "layer_norm_grad_v3_tiling.h"

namespace optiling {
constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_EIGHT = 8;

constexpr int64_t DEFAULT_GAMMA_BETA_M_FACTOR = 128;
#ifdef DAVID_FPGA
constexpr int64_t DEFAULT_GAMMA_BETA_N_FACTOR = 16;
#else
constexpr int64_t DEFAULT_GAMMA_BETA_N_FACTOR = 64;
#endif

constexpr int64_t DEFAULT_BACKWARD_M_FACTOR = 1;
#ifdef DAVID_FPGA
constexpr int64_t DEFAULT_BACKWARD_N_FACTOR = 1024;
#else
constexpr int64_t DEFAULT_BACKWARD_N_FACTOR = 1024 * 6;
#endif

bool LayerNormGradV3RecomputeTiling::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::GammaBetaKernelTiling()
{
    // 沿M轴做reduce（输入形状为(M, N)）
    constexpr int64_t mFactor = DEFAULT_GAMMA_BETA_M_FACTOR;
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
        cacheBufferCount = ULONG_BIT_LEN - static_cast<int64_t>(__builtin_clzl(static_cast<uint64_t>(basicBlockLoop)));
        resultCacheID = GetCacheID(basicBlockLoop - 1);
    }

    // N方向分块参数计算（分核处理）
    int64_t nFactorBase = DEFAULT_GAMMA_BETA_N_FACTOR;
    int64_t mainBlockFactor = ops::CeilDiv(colSize, static_cast<int64_t>(commonParams.coreNum));
    mainBlockFactor = std::max(mainBlockFactor, nFactorBase);
    int64_t numBlocks = ops::CeilDiv(colSize, mainBlockFactor);
    int64_t tailBlockFactor = colSize - (numBlocks - CONST_ONE) * mainBlockFactor;
    int64_t nFactorMax = (commonParams.ubSizePlatForm / sizeof(float) - mFactor * CONST_TWO) /
                         (mFactor * CONST_SIX + CONST_THREE + cacheBufferCount * CONST_TWO);
    OP_TILING_CHECK(nFactorMax < nFactorBase,
                    OP_LOGI(context_->GetNodeName(),
                            "Recompute template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
                            "nFactorMax: %ld.",
                            commonParams.rowSize, commonParams.colSize, commonParams.ubSizePlatForm, nFactorMax),
                    return ge::GRAPH_PARAM_INVALID);

    int64_t nFactor = ops::FloorAlign(nFactorMax, nFactorBase);
    int64_t nLoopMain = ops::CeilDiv(mainBlockFactor, nFactor);
    int64_t nTailMain = mainBlockFactor - (nLoopMain - CONST_ONE) * nFactor;
    int64_t nLoopTail = ops::CeilDiv(tailBlockFactor, nFactor);
    int64_t nTailTail = tailBlockFactor - (nLoopTail - CONST_ONE) * nFactor;

    td_.set_gammaBetaMainBlockFactor(mainBlockFactor);
    td_.set_gammaBetaBlockDim(static_cast<int32_t>(numBlocks));
    td_.set_gammaBetaNloopMainBlock(nLoopMain);
    td_.set_gammaBetaNtailMainBlock(nTailMain);
    td_.set_gammaBetaNloopTailBlock(nLoopTail);
    td_.set_gammaBetaNtailTailBlock(nTailTail);
    td_.set_gammaBetaMtail(mTail);
    td_.set_gammaBetaBasicBlockLoop(basicBlockLoop);
    td_.set_gammaBetaMainFoldCount(mainFoldCount);
    td_.set_gammaBetaCacheBufferCount(static_cast<int32_t>(cacheBufferCount));
    td_.set_gammaBetaResultCacheID(static_cast<int32_t>(resultCacheID));
    td_.set_gammaBetaNfactor(static_cast<int32_t>(nFactor));
    td_.set_gammaBetaMfactor(static_cast<int32_t>(mFactor));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::BackwardKernelTiling()
{
    int64_t backwardMfactor = DEFAULT_BACKWARD_M_FACTOR;
    int64_t backwardNfactor = DEFAULT_BACKWARD_N_FACTOR;
    if (static_cast<int64_t>(commonParams.colSize) < backwardNfactor) {
        backwardNfactor = ops::CeilAlign(static_cast<int64_t>(commonParams.colSize),
            static_cast<int64_t>(commonParams.vlFp32));
    }
    int64_t backwardNfactorBlockAligned = ops::CeilAlign(
        static_cast<int64_t>(backwardNfactor * sizeof(float)), static_cast<int64_t>(commonParams.blockSize)) / sizeof(float);
    int64_t backwardMainBlockFactor =
        ops::CeilDiv(static_cast<int64_t>(commonParams.rowSize), static_cast<int64_t>(commonParams.coreNum));
    int64_t backwardBlockDim = ops::CeilDiv(static_cast<int64_t>(commonParams.rowSize), backwardMainBlockFactor);
    int64_t backwardMainBlockCount =
        ops::FloorDiv(static_cast<int64_t>(commonParams.rowSize), static_cast<int64_t>(backwardMainBlockFactor));
    int64_t backwardTailBlockCount = backwardBlockDim - backwardMainBlockCount;
    int64_t backwardTailBlockFactor = commonParams.rowSize - backwardMainBlockCount * backwardMainBlockFactor;

    int64_t backwardNLoopMain = ops::FloorDiv(
        static_cast<int64_t>(commonParams.colSize), static_cast<int64_t>(backwardNfactor)); 
    int64_t backwardNTotalLoopMain = ops::CeilDiv(
        static_cast<int64_t>(commonParams.colSize), static_cast<int64_t>(backwardNfactor));
    int64_t backwardNLoopTail = commonParams.colSize - backwardNLoopMain * backwardNfactor;
    int64_t backwardBasicBlockLoop = FindNearestPower2(backwardNTotalLoopMain); 
    int64_t backwardMainFoldCount = backwardNLoopMain - backwardBasicBlockLoop;

    int64_t backwardCacheBufferCountMain = 1;
    int64_t backwardResultCacheIDMain = 0;
    if (backwardBasicBlockLoop != 0) {
        backwardCacheBufferCountMain = ULONG_BIT_LEN - static_cast<int64_t>(__builtin_clzl(
            static_cast<uint64_t>(backwardBasicBlockLoop))); 
        backwardResultCacheIDMain = GetCacheID(backwardBasicBlockLoop - 1);
    }

    int64_t backwardCeilVLCount = ops::CeilDiv(
        static_cast<int64_t>(backwardNfactorBlockAligned), static_cast<int64_t>(commonParams.vlFp32));
    int64_t backwardFoldPoint = FindNearestPower2(backwardCeilVLCount);
    int64_t backwardFoldSize = ops::CeilAlign(
        static_cast<int64_t>(backwardFoldPoint), static_cast<int64_t>(commonParams.blockSize / sizeof(float)));
    
    int64_t backwardMFactorMax = (commonParams.ubSizePlatForm / sizeof(float) - CONST_TWO * backwardNfactor) /
        ((CONST_TWO * CONST_THREE + CONST_ONE) * backwardNfactor + CONST_FOUR + CONST_ONE +
        CONST_TWO * backwardCacheBufferCountMain + backwardFoldSize);

     OP_TILING_CHECK(backwardMFactorMax <= 0,
                    OP_LOGI(context_->GetNodeName(),
                            "Recompute template is not capable. merged shape is (%lu, %lu), ub size: %luB, "
                            "backwardMfactorMax: %ld.",
                            commonParams.rowSize, commonParams.colSize, commonParams.ubSizePlatForm, backwardMFactorMax),
                    return ge::GRAPH_PARAM_INVALID);
    backwardMfactor = static_cast<int64_t>(commonParams.rowSize) < backwardMFactorMax
                    ? static_cast<int64_t>(commonParams.rowSize)
                    : backwardMFactorMax;

    int64_t backwardMLoopMain = ops::FloorDiv(
        static_cast<int64_t>(backwardMainBlockFactor), static_cast<int64_t>(backwardMfactor));
    int64_t backwardMLoopTail = backwardMainBlockFactor - backwardMLoopMain * backwardMfactor;
    
    // 尾块处理
    int64_t backwardMLoopTailTail = ops::FloorDiv(
        static_cast<int64_t>(backwardTailBlockFactor), static_cast<int64_t>(backwardMfactor));
    int64_t backwardMTailTail = backwardTailBlockFactor - backwardMLoopTailTail * backwardMfactor;
    
    int64_t backwardMfactorBlockAligned = ops::CeilAlign(
        static_cast<int64_t>(backwardMfactor), static_cast<int64_t>(
        static_cast<int64_t>(commonParams.blockSize) / static_cast<int64_t>(sizeof(float))));

    td_.set_backwardMfactor(static_cast<int32_t>(backwardMfactor)); 
    td_.set_backwardNfactor(static_cast<int32_t>(backwardNfactor)); 
    td_.set_backwardMainBlockFactor(backwardMainBlockFactor);
    td_.set_backwardBlockDim(static_cast<int32_t>(backwardBlockDim));

    td_.set_backwardMainBlockCount(backwardMainBlockCount); 
    td_.set_backwardTailBlockCount(backwardTailBlockCount); 
    td_.set_backwardTailBlockFactor(backwardTailBlockFactor); 

    td_.set_backwardMLoopMain(backwardMLoopMain); 
    td_.set_backwardMLoopTail(backwardMLoopTail); 

    td_.set_backwardMLoopTailTail(backwardMLoopTailTail); 
    td_.set_backwardMTailTail(backwardMTailTail); 

    td_.set_backwardNLoopMain(backwardNLoopMain); 
    td_.set_backwardNTotalLoopMain(backwardNTotalLoopMain); 
    td_.set_backwardNLoopTail(backwardNLoopTail);  
    td_.set_backwardBasicBlockLoop(backwardBasicBlockLoop); 
    td_.set_backwardMainFoldCount(backwardMainFoldCount); 

    td_.set_backwardNfactorBlockAligned(backwardNfactorBlockAligned); 
    td_.set_backwardMfactorBlockAligned(backwardMfactorBlockAligned); 

    td_.set_backwardCacheBufferCountMain(static_cast<int32_t>(backwardCacheBufferCountMain));  
    td_.set_backwardResultCacheIDMain(static_cast<int32_t>(backwardResultCacheIDMain)); 

    td_.set_backwardCeilVLCount(backwardCeilVLCount);
    td_.set_backwardFoldPoint(backwardFoldPoint);  
    td_.set_backwardFoldSize(backwardFoldSize);  

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::DoOpTiling()
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

ge::graphStatus LayerNormGradV3RecomputeTiling::GetWorkspaceSize()
{
    constexpr int64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradV3RecomputeTiling::PostTiling()
{
    int64_t numBlocks = td_.get_gammaBetaBlockDim() > td_.get_backwardBlockDim() ? td_.get_gammaBetaBlockDim() :
                                                                                   td_.get_backwardBlockDim();
    context_->SetBlockDim(numBlocks);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormGradV3RecomputeTiling::GetTilingKey() const
{
    constexpr uint64_t recomputeTemplateKey = 500;
    return recomputeTemplateKey;
}

REGISTER_TILING_TEMPLATE("LayerNormGradV3", LayerNormGradV3RecomputeTiling, 7000);
} // namespace optiling
