/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_segment_mean_tiling_simd_tiling.cpp
 * \brief
 */
#include "sparse_segment_mean_tiling_simd_tiling.h"

namespace optiling {
static constexpr int64_t SIMD_TILING_KEY = 100;
static constexpr int64_t WS_SYS_SIZE = 16 * 1024 * 1024;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t GM_NUM_OFFSET = 128;
static constexpr int64_t SINGLE_CORE_THRESHOLD = 4 * 1024;
static constexpr int64_t MIN_BATCH_BINAEY_ACC = 64;
static constexpr int64_t BASE_BLOCK_ALIGN = 512;
static constexpr int64_t WORKSPACE_BUFFER_NUM = 2;

std::set<int64_t> ComputeFactors(int64_t usedCoreNum)
{
    std::set<int64_t> result;
    int64_t upbound = std::ceil(std::sqrt(usedCoreNum) + 1);

    for (int64_t m = 1; m < upbound; m++) {
        int64_t y = static_cast<int64_t>(usedCoreNum) / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

void AutoTilingRowCol(int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum, int64_t rowTotalNum, int64_t colTotalNum)
{
    int64_t tmpEleNum = BASE_BLOCK_ALIGN / sizeof(float);
    int64_t colBlockTotalNum = (colTotalNum + tmpEleNum - 1) / tmpEleNum;
    usedCoreNum = std::min(usedCoreNum, std::max(int64_t(1), rowTotalNum * colBlockTotalNum * tmpEleNum / (SINGLE_CORE_THRESHOLD)));

    std::set<int64_t> cutSet = ComputeFactors(usedCoreNum);
    std::vector<std::vector<int64_t>> allTiling;

    for (int64_t m : cutSet) {
        if (m > rowTotalNum) {
            continue;
        }

        int64_t n = usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > colBlockTotalNum) {
            continue;
        }

        int64_t rowNormalBlock = Ops::Base::CeilDiv(rowTotalNum, m);
        int64_t mReal = Ops::Base::CeilDiv(rowTotalNum, rowNormalBlock);
        int64_t rowTailBlock = rowTotalNum - (mReal - 1) * rowNormalBlock;

        int64_t colNormalBlock = Ops::Base::CeilDiv(colBlockTotalNum, n);
        int64_t nReal = Ops::Base::CeilDiv(colBlockTotalNum, colNormalBlock);
        int64_t colTailBlock = colBlockTotalNum - (nReal - 1) * colNormalBlock;

        int64_t blockNormal = rowNormalBlock * colNormalBlock;
        int64_t blockTail = rowTailBlock * colTailBlock;
        int64_t delta = blockNormal - blockTail;
        allTiling.push_back({m, n, m * n, delta});
    }

    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int NIndex = 1;
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[NIndex]) < std::make_pair(b[DeltaIndex], b[NIndex]);
    });

    rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    colTileNum = static_cast<uint16_t>(allTiling[0][1]);
}

bool SparseSegmentMeanSimdTiling::IsCapable()
{
    return true;
}

uint64_t SparseSegmentMeanSimdTiling::GetTilingKey() const
{
    uint64_t tilingKey = SIMD_TILING_KEY;
    return tilingKey;
}

void SparseSegmentMeanSimdTiling::DoUBTiling()
{
    splitData.xBufferSize = hardwareData.ubSize / DOUBLE;
    int64_t indicesSingleLoopPro = std::min(MIN_BATCH_BINAEY_ACC, inputData.outterSize);
    int64_t innersizeSingleLoopPro = std::min(splitData.xBufferSize / indicesSingleLoopPro, inputData.innerSize);

    auto shape = ge::Shape({indicesSingleLoopPro, innersizeSingleLoopPro});
    uint32_t maxValue = 0;
    uint32_t minValue = 0;

    AscendC::GetReduceSumMaxMinTmpSize(
        shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::RA, true, true, maxValue, minValue);
    splitData.sharedTmpBufferSize = maxValue;
    
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    splitData.workspaceBufferSize = Ops::Base::CeilAlign(GM_NUM_OFFSET + static_cast<int64_t>(sizeof(int64_t)), ubBlockSize);
    splitData.yBufferSize = (hardwareData.ubSize - splitData.sharedTmpBufferSize - splitData.xBufferSize - splitData.workspaceBufferSize * WORKSPACE_BUFFER_NUM) / DOUBLE;
   
    splitData.inBufferSize = hardwareData.ubSize / DOUBLE;
    splitData.outBufferSize = (hardwareData.ubSize - splitData.sharedTmpBufferSize - splitData.inBufferSize) / DOUBLE;
}

void SparseSegmentMeanSimdTiling::DoBlockTiling()
{
    int64_t rowTileNum = 0;
    int64_t colTileNum = 0;
    int64_t usedCoreNum = hardwareData.coreNum;
    int64_t rowTotalNum = inputData.outterSize;
    int64_t colTotalNum = inputData.innerSize;

    AutoTilingRowCol(rowTileNum, colTileNum, usedCoreNum, rowTotalNum, colTotalNum);

    splitData.normalCoreIndicesNum = Ops::Base::CeilDiv(rowTotalNum, rowTileNum);
    splitData.indicesOuter = Ops::Base::CeilDiv(rowTotalNum, splitData.normalCoreIndicesNum);
    splitData.tailCoreIndicesNum = rowTotalNum - (splitData.indicesOuter - 1) * splitData.normalCoreIndicesNum;

    splitData.normalCoreInnerNum = Ops::Base::CeilDiv(colTotalNum, colTileNum);
    splitData.innerOuter = Ops::Base::CeilDiv(colTotalNum, splitData.normalCoreInnerNum);
    splitData.tailCoreInnerNum = colTotalNum - (splitData.innerOuter - 1) * splitData.normalCoreInnerNum;

    splitData.usedCoreNum = splitData.indicesOuter * splitData.innerOuter;

    int64_t totalOutputSize = inputData.segmentNum * inputData.innerSize;
    splitData.normalCoreProcessNumForClear = Ops::Base::CeilDiv(totalOutputSize, splitData.usedCoreNum);
    splitData.usedCoreNumForClear = Ops::Base::CeilDiv(totalOutputSize, splitData.normalCoreProcessNumForClear);
    splitData.tailCoreProcessNumForClear =
        totalOutputSize - (splitData.usedCoreNumForClear - 1) * splitData.normalCoreProcessNumForClear;

    splitData.perCoreInnerElements = Ops::Base::CeilDiv(inputData.innerSize, splitData.usedCoreNum);
    splitData.usedCoreNumForMulCore = Ops::Base::CeilDiv(inputData.innerSize, splitData.perCoreInnerElements);
    splitData.tailCoreInnerElements = inputData.innerSize - (splitData.usedCoreNumForMulCore - 1) * splitData.perCoreInnerElements;
}

void SparseSegmentMeanSimdTiling::PrintSplitData() const
{
    OP_LOGD("SparseSegmentMeanSimd", "[SparseSegmentMeanSimd] PrintSplitData start running");

    std::ostringstream info;
    info << "splitData.usedCoreNum: " << splitData.usedCoreNum << std::endl;

    info << "splitData.normalCoreInnerNum: " << splitData.normalCoreInnerNum << std::endl;
    info << "splitData.innerOuter: " << splitData.innerOuter << std::endl;
    info << "splitData.tailCoreInnerNum: " << splitData.tailCoreInnerNum << std::endl;

    info << "splitData.normalCoreIndicesNum: " << splitData.normalCoreIndicesNum << std::endl;
    info << "splitData.indicesOuter: " << splitData.indicesOuter << std::endl;
    info << "splitData.tailCoreIndicesNum: " << splitData.tailCoreIndicesNum << std::endl;

    info << "splitData.xBufferSize: " << splitData.xBufferSize << std::endl;
    info << "splitData.yBufferSize: " << splitData.yBufferSize << std::endl;
    info << "splitData.sharedTmpBufferSize: " << splitData.sharedTmpBufferSize << std::endl;

    info << "splitData.normalCoreProcessNumForClear: " << splitData.normalCoreProcessNumForClear << std::endl;
    info << "splitData.tailCoreProcessNumForClear: " << splitData.tailCoreProcessNumForClear << std::endl;
    info << "splitData.usedCoreNumForClear: " << splitData.usedCoreNumForClear << std::endl;

    info << "splitData.perCoreInnerElements: " << splitData.perCoreInnerElements << std::endl;
    info << "splitData.tailCoreInnerElements: " << splitData.tailCoreInnerElements << std::endl;
    info << "splitData.usedCoreNumForMulCore: " << splitData.usedCoreNumForMulCore << std::endl;
    info << "splitData.inBufferSize: " << splitData.inBufferSize << std::endl;
    info << "splitData.outBufferSize: " << splitData.outBufferSize << std::endl;
    info << "splitData.workspaceBufferSize: " << splitData.workspaceBufferSize << std::endl;

    OP_LOGI("SparseSegmentMeanSimd", "%s", info.str().c_str());
}

void SparseSegmentMeanSimdTiling::SetTilingData()
{
    SparseSegmentMeanSimdTilingData* tilingData = context_->GetTilingData<SparseSegmentMeanSimdTilingData>();
    tilingData->tilingkey = GetTilingKey();
    tilingData->usedCoreNum = splitData.usedCoreNum;
    tilingData->innerSize = inputData.innerSize;
    tilingData->gatherSize = inputData.gatherSize;

    tilingData->xBufferSize = splitData.xBufferSize;
    tilingData->yBufferSize = splitData.yBufferSize;
    tilingData->sharedTmpBufferSize = splitData.sharedTmpBufferSize;

    tilingData->normalCoreInnerNum = splitData.normalCoreInnerNum;
    tilingData->tailCoreInnerNum = splitData.tailCoreInnerNum;
    tilingData->innerOuter = splitData.innerOuter;

    tilingData->normalCoreIndicesNum = splitData.normalCoreIndicesNum;
    tilingData->tailCoreIndicesNum = splitData.tailCoreIndicesNum;
    tilingData->indicesOuter = splitData.indicesOuter;

    tilingData->perCoreInnerElements = splitData.perCoreInnerElements;
    tilingData->tailCoreInnerElements = splitData.tailCoreInnerElements;
    tilingData->usedCoreNumForMulCore = splitData.usedCoreNumForMulCore;
    tilingData->inBufferSize = splitData.inBufferSize;
    tilingData->outBufferSize = splitData.outBufferSize;
    tilingData->workspaceBufferSize = splitData.workspaceBufferSize;
    
    tilingData->normalCoreProcessNumForClear = splitData.normalCoreProcessNumForClear;
    tilingData->tailCoreProcessNumForClear = splitData.tailCoreProcessNumForClear;
    tilingData->usedCoreNumForClear = splitData.usedCoreNumForClear;
}

ge::graphStatus SparseSegmentMeanSimdTiling::DoOpTiling()
{
    DoBlockTiling();
    DoUBTiling();
    SetTilingData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSegmentMeanSimdTiling::GetWorkspaceSize()
{
    auto sysWorkspace = WS_SYS_SIZE;
    sysWorkspace += inputData.innerSize * splitData.indicesOuter * DOUBLE * sizeof(float);
    sysWorkspace += GM_NUM_OFFSET;
    sysWorkspace += splitData.indicesOuter * DOUBLE * DOUBLE * GM_NUM_OFFSET;
    sysWorkspace += GM_NUM_OFFSET;  // 预留 防止越界访问
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspace);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSegmentMeanSimdTiling::PostTiling()
{
    context_->SetBlockDim(splitData.usedCoreNum);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SparseSegmentMean, SparseSegmentMeanSimdTiling, 2);

} // namespace optiling