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
 * \file scatter_add_with_sorted_simd_tiling.cpp
 * \brief scatter_add_with_sorted_simd_tiling
 */

#include "scatter_add_with_sorted_simd_tiling.h"

namespace optiling {

static constexpr int64_t SIMD_INNER_THRES = 128;
static constexpr int64_t BASE_BLOCK_ALIGN = 512;
static constexpr int64_t SINGLE_CORE_THRESHOLD = 4 * 1024;
static constexpr int64_t CACHE_ALIGN_SIZE = 128;

static constexpr int64_t NUM_FOUR = 4;
static constexpr int64_t COL_TILING_THRES = 8 * 1024;
constexpr uint64_t BUFFER_NUM = 2;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = static_cast<int64_t>(16) * 1024 * 1024;

bool ScatterAddWithSortedSimdTiling::IsCapable()
{
    bool isSimd = varShape_[1] * updatesDtypeSize_ >= SIMD_INNER_THRES;
    return isSimd;
}

std::set<int64_t> FindFactor(int64_t usedCoreNum)
{
    std::set<int64_t> result;
    uint64_t upbound = std::ceil(std::sqrt(usedCoreNum) + 1);

    for (uint64_t m = 1; m < upbound; m++) {
        uint64_t y = usedCoreNum / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

void ScatterAddWithSortedSimdTiling::AutoTilingRowCol(int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum, int64_t rowTotalNum, int64_t colTotalNum)
{
    int64_t tmpEleNum = BASE_BLOCK_ALIGN / updatesDtypeSize_;
    int64_t colBlockTotalNum = (colTotalNum + tmpEleNum - 1) / tmpEleNum;
    usedCoreNum = std::min(usedCoreNum, std::max<int64_t>(1, rowTotalNum * colBlockTotalNum * tmpEleNum / (SINGLE_CORE_THRESHOLD)));


    std::set<int64_t> cutSet = FindFactor(usedCoreNum);
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
        return std::make_pair(a[NIndex], a[DeltaIndex]) < std::make_pair(b[NIndex], b[DeltaIndex]);
    });

    int64_t allTilingSize = static_cast<int64_t>(allTiling.size());
    while (allTilingSize > 1 && static_cast<int64_t>(indicesNum_) / allTiling[0][0] < std::min<int64_t>(NUM_FOUR, indicesNum_)) {
        allTiling.erase(allTiling.begin());
        allTilingSize = static_cast<int64_t>(allTiling.size());
    }
    rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    colTileNum = static_cast<uint16_t>(allTiling[0][1]);
}

void ScatterAddWithSortedSimdTiling::DoBlockTiling()
{
    ubBlock_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    availableUbsize = ubSize_;
    availableUbsize = Ops::Base::FloorAlign(static_cast<int64_t>(availableUbsize / BUFFER_NUM), ubBlock_);
    FrontAndBackIndex = Ops::Base::CeilAlign(
        static_cast<int64_t>(2 * indicesDtypeSize_), ubBlock_);
    resUb = availableUbsize - 3 * static_cast<int64_t>(Ops::Base::CeilAlign(static_cast<int64_t>(varShape_[1] * updatesDtypeSize_), ubBlock_));
    resUb = isDeterminTemplate_ ? std::max<int64_t>(0, resUb - FrontAndBackIndex) : resUb;
    if (resUb >= COL_TILING_THRES) {
        coreNumInCol_ = 1;
        normalCoreColNum_ = varShape_[1];
        tailCoreColNum_ = varShape_[1];
        coreNumInRow_ = std::min(totalCoreNum_, indicesNum_);
        normalCoreRowNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), coreNumInRow_);
        coreNumInRow_ = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), normalCoreRowNum_);
        tailCoreRowNum_ = indicesNum_ - (coreNumInRow_ - 1) * normalCoreRowNum_;
    } else {
        int64_t rowTileNum = 0;
        int64_t colTileNum = 0;
        AutoTilingRowCol(rowTileNum, colTileNum, totalCoreNum_, indicesNum_, varShape_[1]);

        normalCoreRowNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), rowTileNum);
        coreNumInRow_ = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), normalCoreRowNum_);
        tailCoreRowNum_ = indicesNum_ - (coreNumInRow_ - 1) * normalCoreRowNum_;

        normalCoreColNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(varShape_[1]), colTileNum);
        coreNumInCol_ = Ops::Base::CeilDiv(static_cast<int64_t>(varShape_[1]), normalCoreColNum_);
        tailCoreColNum_ = varShape_[1] - (coreNumInCol_ - 1) * normalCoreColNum_;
    }
    needCoreNum_ = coreNumInCol_ * coreNumInRow_;
}

void ScatterAddWithSortedSimdTiling::DoUBTiling()
{
    int64_t rowNumInUb = 0;
    if (resUb >= COL_TILING_THRES) {
        int64_t colSizeAlign = Ops::Base::CeilAlign(
            static_cast<int64_t>(normalCoreColNum_ * updatesDtypeSize_), ubBlock_);
        int64_t tmpRowNum = Ops::Base::FloorAlign(resUb / 2, ubBlock_) / indicesDtypeSize_;
        rowNumInUb = std::min(tmpRowNum, normalCoreRowNum_);
        rowNumInUb = std::max<int64_t>(1, rowNumInUb);
        updatesBufferSize_ = colSizeAlign;
        outBufferSize_ = colSizeAlign;
        normalCoreColUbLoop_ = 1;
        normalCoreNormalLoopCols_ = normalCoreColNum_;
        normalCoreTailLoopCols_ = normalCoreColNum_;
        tailCoreColUbLoop_ = 1;
        tailCoreNormalLoopCols_ = tailCoreColNum_;
        tailCoreTailLoopCols_ = tailCoreColNum_;
    } else {
        int64_t tmpRowNum = COL_TILING_THRES / 2 / indicesDtypeSize_;
        rowNumInUb = std::min(tmpRowNum, normalCoreRowNum_);
        int64_t colSizeInUb = Ops::Base::FloorAlign((availableUbsize - COL_TILING_THRES) / 3, ubBlock_);
        updatesBufferSize_ = colSizeInUb;
        outBufferSize_ = colSizeInUb;
        int64_t colNumInUb = colSizeInUb / updatesDtypeSize_;
        colNumInUb = std::max<int64_t>(1, colNumInUb);
        normalCoreColUbLoop_ = Ops::Base::CeilDiv(normalCoreColNum_, colNumInUb);
        normalCoreNormalLoopCols_ = Ops::Base::CeilDiv(normalCoreColNum_, normalCoreColUbLoop_);
        normalCoreTailLoopCols_ = normalCoreColNum_ - (normalCoreColUbLoop_ - 1) * normalCoreNormalLoopCols_;
        tailCoreColUbLoop_ = Ops::Base::CeilDiv(tailCoreColNum_, colNumInUb);
        tailCoreNormalLoopCols_ = Ops::Base::CeilDiv(tailCoreColNum_, tailCoreColUbLoop_);
        tailCoreTailLoopCols_ = tailCoreColNum_ - (tailCoreColUbLoop_ - 1) * tailCoreNormalLoopCols_;
    }
    indicesBufferSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(rowNumInUb * indicesDtypeSize_), ubBlock_);
    posBufferSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(rowNumInUb * indicesDtypeSize_), ubBlock_);
    normalCoreRowUbLoop_ = Ops::Base::CeilDiv(normalCoreRowNum_, rowNumInUb);
    normalCoreNormalLoopRows_ = Ops::Base::CeilDiv(normalCoreRowNum_, normalCoreRowUbLoop_);
    normalCoreTailLoopRows_ = normalCoreRowNum_ - (normalCoreRowUbLoop_ - 1) * normalCoreNormalLoopRows_;
    tailCoreRowUbLoop_ = Ops::Base::CeilDiv(tailCoreRowNum_, rowNumInUb);
    tailCoreNormalLoopRows_ = Ops::Base::CeilDiv(tailCoreRowNum_, tailCoreRowUbLoop_);
    tailCoreTailLoopRows_ = tailCoreRowNum_ - (tailCoreRowUbLoop_ - 1) * tailCoreNormalLoopRows_;
}

void ScatterAddWithSortedSimdTiling::DeterminTemplateUbTiling()
{
    vecAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(varShape_[1] * updatesDtypeSize_), BASE_BLOCK_ALIGN);
    int64_t resUbForDetermin = ubSize_ - 3 * static_cast<int64_t>(Ops::Base::CeilAlign(
                                                 static_cast<int64_t>(varShape_[1] * updatesDtypeSize_), ubBlock_));
    // do UBTILING
    if (resUbForDetermin >=
        (coreNumInRow_ * CACHE_ALIGN_SIZE)) {
        int64_t colSizeAlignDetermin =
            Ops::Base::CeilAlign(static_cast<int64_t>(varShape_[1] * updatesDtypeSize_), ubBlock_);
        updatesDeterminBufferSize_ = colSizeAlignDetermin;
        outBufferDeterminSize_ = colSizeAlignDetermin;
        normalCoreColDetermNum_ = varShape_[1];
        colNumInUbDeterm = varShape_[1];
        colNumInUbDeterm = std::max<int64_t>(1, colNumInUbDeterm);
        coreNumInColDeterm_ = 1;
        tailCoreColNumDeterm_ = varShape_[1];

    } else {
        int64_t resUbForUpdates = ubSize_ - Ops::Base::FloorAlign((coreNumInRow_ * CACHE_ALIGN_SIZE), ubBlock_);
        int64_t copyUpdatesInUb = Ops::Base::FloorAlign(resUbForUpdates / 3, ubBlock_);
        updatesDeterminBufferSize_ = copyUpdatesInUb;
        outBufferDeterminSize_ = copyUpdatesInUb;
        normalCoreColDetermNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(varShape_[1]), coreNumInCol_);

        colNumInUbDeterm = copyUpdatesInUb / updatesDtypeSize_;
        colNumInUbDeterm = std::max<int64_t>(1, colNumInUbDeterm);

        coreNumInColDeterm_ = Ops::Base::CeilDiv(static_cast<int64_t>(varShape_[1]), normalCoreColDetermNum_);
        tailCoreColNumDeterm_ = varShape_[1] - (coreNumInColDeterm_ - 1) * normalCoreColDetermNum_;
    }

    indicesWorkspaceBufferSize_ = Ops::Base::FloorAlign((coreNumInRow_ * CACHE_ALIGN_SIZE), ubBlock_);
    normalCoreColUbDetermLoop_ = Ops::Base::CeilDiv(normalCoreColDetermNum_, colNumInUbDeterm);
    normalCoreNormalLoopDetermCols_ = Ops::Base::CeilDiv(normalCoreColDetermNum_, normalCoreColUbDetermLoop_);
    normalCoreTailLoopDetermCols_ =
        normalCoreColDetermNum_ - (normalCoreColUbDetermLoop_ - 1) * normalCoreNormalLoopDetermCols_;

    tailCoreColUbDetermLoop_ = Ops::Base::CeilDiv(tailCoreColNumDeterm_, colNumInUbDeterm);
    tailCoreNormalLoopDetermCols_ = Ops::Base::CeilDiv(tailCoreColNumDeterm_, tailCoreColUbDetermLoop_);
    tailCoreTailLoopDetermCols_ =
        tailCoreColNumDeterm_ - (tailCoreColUbDetermLoop_ - 1) * tailCoreNormalLoopDetermCols_;
}

void ScatterAddWithSortedSimdTiling::SetTilingData()
{
    tilingData_ = context_->GetTilingData<ScatterAddWithSortedSimdTilingData>();
    tilingData_->needCoreNum = needCoreNum_;
    tilingData_->indicesNum = indicesNum_;
    tilingData_->updatesInner = varShape_[1];
    tilingData_->withPos = hasPos_;

    tilingData_->updatesBufferSize = updatesBufferSize_;
    tilingData_->outBufferSize = outBufferSize_;
    tilingData_->indicesBufferSize = indicesBufferSize_;
    tilingData_->posBufferSize = posBufferSize_;
    tilingData_->FrontAndBackIndexSize = FrontAndBackIndex;

    tilingData_->coreNumInRow = coreNumInRow_;
    tilingData_->coreNumInCol = coreNumInCol_;

    tilingData_->normalCoreColNum = normalCoreColNum_;
    tilingData_->tailCoreColNum = tailCoreColNum_;
    tilingData_->normalCoreRowNum = normalCoreRowNum_;
    tilingData_->tailCoreRowNum = tailCoreRowNum_;

    tilingData_->normalCoreRowUbLoop = normalCoreRowUbLoop_;
    tilingData_->normalCoreNormalLoopRows = normalCoreNormalLoopRows_;
    tilingData_->normalCoreTailLoopRows = normalCoreTailLoopRows_;
    tilingData_->tailCoreRowUbLoop = tailCoreRowUbLoop_;
    tilingData_->tailCoreNormalLoopRows = tailCoreNormalLoopRows_;
    tilingData_->tailCoreTailLoopRows = tailCoreTailLoopRows_;

    tilingData_->normalCoreColUbLoop = normalCoreColUbLoop_;
    tilingData_->normalCoreNormalLoopCols = normalCoreNormalLoopCols_;
    tilingData_->normalCoreTailLoopCols = normalCoreTailLoopCols_;
    tilingData_->tailCoreColUbLoop = tailCoreColUbLoop_;
    tilingData_->tailCoreNormalLoopCols = tailCoreNormalLoopCols_;
    tilingData_->tailCoreTailLoopCols = tailCoreTailLoopCols_;

    tilingData_->vecAlignSize = vecAlignSize_;
    tilingData_->indicesWorkspaceBufferSize = indicesWorkspaceBufferSize_;
    tilingData_->coreNumInColDeterm = coreNumInColDeterm_;
    tilingData_->tailCoreColUbDetermLoop = tailCoreColUbDetermLoop_;
    tilingData_->normalCoreColUbDetermLoop = normalCoreColUbDetermLoop_;
    tilingData_->tailCoreNormalLoopDetermCols = tailCoreNormalLoopDetermCols_;
    tilingData_->normalCoreNormalLoopDetermCols = normalCoreNormalLoopDetermCols_;
    tilingData_->tailCoreTailLoopDetermCols = tailCoreTailLoopDetermCols_;
    tilingData_->normalCoreTailLoopDetermCols = normalCoreTailLoopDetermCols_;
    tilingData_->updatesDeterminBufferSize = updatesDeterminBufferSize_;
    tilingData_->outBufferDeterminSize = outBufferDeterminSize_;
    tilingData_->normalCoreColDetermNum = normalCoreColDetermNum_;
    tilingData_->tailCoreColNumDeterm = tailCoreColNumDeterm_;
    tilingData_->ubBlock = ubBlock_;
    tilingData_->tilingKey = GetTilingKey();
    return;
}

ge::graphStatus ScatterAddWithSortedSimdTiling::DoOpTiling()
{
    if (varShape_[0] * varShape_[1] * indicesNum_ == 0) {
        needCoreNum_ = 1;
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }

    DoBlockTiling();
    DoUBTiling();

    if (isDeterminTemplate_) {
        DeterminTemplateUbTiling();
    }

    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterAddWithSortedSimdTiling::GetTilingKey() const
{
    if (varShape_[0] * varShape_[1] * indicesNum_ == 0) {
        return GET_TPL_TILING_KEY(TPL_MODE_EMPTY, TPL_SCALAR_FALSE, TPL_DETERM_FALSE, TPL_ADDR_B32);
    }

    uint64_t isScalar = isUpdateScalar_ ? TPL_SCALAR_TRUE : TPL_SCALAR_FALSE;
    uint64_t isDeterm = isDeterminTemplate_ ? TPL_DETERM_TRUE : TPL_DETERM_FALSE;
    return GET_TPL_TILING_KEY(TPL_MODE_SIMD, isScalar, isDeterm, TPL_ADDR_B32);
}

ge::graphStatus ScatterAddWithSortedSimdTiling::GetWorkspaceSize()
{
    size_t useWorkspace = 0;
    if (isDeterminTemplate_) {
        useWorkspace += coreNumInRow_ * 2 * vecAlignSize_ + coreNumInRow_ * CACHE_ALIGN_SIZE;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = useWorkspace + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedSimdTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "ScatterAddWithSortedSimdTiling simd PostTiling enter.");
    context_->SetBlockDim(needCoreNum_);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void ScatterAddWithSortedSimdTiling::DumpTilingInfo()
{
    std::ostringstream info1;
    info1 << "tilingKey: " << GetTilingKey();
    info1 << ", UB Size: " << ubSize_;
    info1 << ", needCoreNum: " << tilingData_->needCoreNum;
    info1 << ", indicesNum: " << tilingData_->indicesNum;
    info1 << ", updatesInner: " << tilingData_->updatesInner;

    info1 << ", updatesBufferSize: " << tilingData_->updatesBufferSize;
    info1 << ", outBufferSize: " << tilingData_->outBufferSize;
    info1 << ", indicesBufferSize: " << tilingData_->indicesBufferSize;
    info1 << ", posBufferSize: " << tilingData_->posBufferSize;

    info1 << ", coreNumInRow: " << tilingData_->coreNumInRow;
    info1 << ", coreNumInCol: " << tilingData_->coreNumInCol;

    info1 << ", normalCoreColNum: " << tilingData_->normalCoreColNum;
    info1 << ", normalCoreRowNum: " << tilingData_->normalCoreRowNum;
    info1 << ", tailCoreRowNum: " << tilingData_->tailCoreRowNum;

    info1 << ", normalCoreRowUbLoop: " << tilingData_->normalCoreRowUbLoop;
    info1 << ", tailCoreColNum: " << tilingData_->tailCoreColNum;
    info1 << ", normalCoreTailLoopRows: " << tilingData_->normalCoreTailLoopRows;
    info1 << ", tailCoreRowUbLoop: " << tilingData_->tailCoreRowUbLoop;
    info1 << ", tailCoreNormalLoopRows: " << tilingData_->tailCoreNormalLoopRows;
    info1 << ", tailCoreTailLoopRows: " << tilingData_->tailCoreTailLoopRows;
    OP_LOGI(context_->GetNodeName(), "%s", info1.str().c_str());

    std::ostringstream info2;

    info2 << ", normalCoreColUbLoop: " << tilingData_->normalCoreColUbLoop;
    info2 << ", normalCoreNormalLoopCols: " << tilingData_->normalCoreNormalLoopCols;
    info2 << ", normalCoreTailLoopCols: " << tilingData_->normalCoreTailLoopCols;
    info2 << ", tailCoreColUbLoop: " << tilingData_->tailCoreColUbLoop;
    info2 << ", tailCoreNormalLoopCols: " << tilingData_->tailCoreNormalLoopCols;
    info2 << ", tailCoreTailLoopCols: " << tilingData_->tailCoreTailLoopCols;

    info2 << ", vecAlignSize: " << tilingData_->vecAlignSize;
    info2 << ", indicesWorkspaceBufferSize: " << tilingData_->indicesWorkspaceBufferSize;
    info2 << ", coreNumInColDeterm: " << tilingData_->coreNumInColDeterm;
    info2 << ", tailCoreColUbDetermLoop: " << tilingData_->tailCoreColUbDetermLoop;
    info2 << ", normalCoreColUbDetermLoop: " << tilingData_->normalCoreColUbDetermLoop;
    info2 << ", tailCoreNormalLoopDetermCols: " << tilingData_->tailCoreNormalLoopDetermCols;
    info2 << ", normalCoreNormalLoopDetermCols: " << tilingData_->normalCoreNormalLoopDetermCols;
    info2 << ", tailCoreTailLoopDetermCols: " << tilingData_->tailCoreTailLoopDetermCols;
    info2 << ", normalCoreTailLoopDetermCols: " << tilingData_->normalCoreTailLoopDetermCols;
    info2 << ", updatesDeterminBufferSize: " << tilingData_->updatesDeterminBufferSize;
    info2 << ", outBufferDeterminSize: " << tilingData_->outBufferDeterminSize;
    info2 << ", normalCoreColDetermNum: " << tilingData_->normalCoreColDetermNum;
    info2 << ", tailCoreColNumDeterm: " << tilingData_->tailCoreColNumDeterm;

    OP_LOGI(context_->GetNodeName(), "%s", info2.str().c_str());
}

REGISTER_TILING_TEMPLATE("ScatterAddWithSorted", ScatterAddWithSortedSimdTiling, 0);
} // namespace optiling