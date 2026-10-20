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
constexpr int64_t FRONT_AND_BACK_INDEX_COUNT = 2;
constexpr int64_t UB_THREE_BUFFER_COUNT = 3;
constexpr uint64_t BUFFER_NUM_SIMD_SORT = 1;
static constexpr uint64_t BASE_A_SIZE = 4096;
static constexpr uint64_t BASE_BLOCK_SIZE = 4096;
static constexpr uint64_t COL_LIMIT_SIZE = 4096;
static constexpr int64_t DMA_CORE_INNER_SIZE = 6144;
static constexpr int64_t DMA_MIN_CORE_ROW = 16;
static constexpr int64_t MIN_INDICES_NUM_DMA = 24;
static constexpr uint64_t DUP_NUM_DMA = 20;
static constexpr uint32_t SORT_STAT_PADDING = 64;
constexpr int64_t SCATTER_DMA_MAX_ROW = 4095;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t NUM_THREE = 3;

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

void ScatterAddWithSortedSimdTiling::AutoTilingRowCol(int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum,
                                                      int64_t rowTotalNum, int64_t colTotalNum)
{
    int64_t tmpEleNum = BASE_BLOCK_ALIGN / updatesDtypeSize_;
    int64_t colBlockTotalNum = (colTotalNum + tmpEleNum - 1) / tmpEleNum;
    usedCoreNum = std::min(usedCoreNum,
                           std::max<int64_t>(1, rowTotalNum * colBlockTotalNum * tmpEleNum / (SINGLE_CORE_THRESHOLD)));

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
    while (allTilingSize > 1 &&
           static_cast<int64_t>(indicesNum_) / allTiling[0][0] < std::min<int64_t>(NUM_FOUR, indicesNum_)) {
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
    FrontAndBackIndex = Ops::Base::CeilAlign(static_cast<int64_t>(FRONT_AND_BACK_INDEX_COUNT * indicesDtypeSize_),
                                             ubBlock_);
    resUb = availableUbsize -
            UB_THREE_BUFFER_COUNT * static_cast<int64_t>(Ops::Base::CeilAlign(
                                        static_cast<int64_t>(varShape_[1] * updatesDtypeSize_), ubBlock_));
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
        int64_t colSizeAlign = Ops::Base::CeilAlign(static_cast<int64_t>(normalCoreColNum_ * updatesDtypeSize_),
                                                    ubBlock_);
        int64_t tmpRowNum = Ops::Base::FloorAlign(static_cast<int64_t>(resUb / BUFFER_NUM), ubBlock_) /
                            indicesDtypeSize_;
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
        int64_t tmpRowNum = COL_TILING_THRES / BUFFER_NUM / indicesDtypeSize_;
        rowNumInUb = std::min(tmpRowNum, normalCoreRowNum_);
        int64_t colSizeInUb = Ops::Base::FloorAlign((availableUbsize - COL_TILING_THRES) / UB_THREE_BUFFER_COUNT,
                                                    ubBlock_);
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
    int64_t resUbForDetermin = ubSize_ - UB_THREE_BUFFER_COUNT *
                                             static_cast<int64_t>(Ops::Base::CeilAlign(
                                                 static_cast<int64_t>(varShape_[1] * updatesDtypeSize_), ubBlock_));
    // do UBTILING
    if (resUbForDetermin >= (coreNumInRow_ * CACHE_ALIGN_SIZE)) {
        int64_t colSizeAlignDetermin = Ops::Base::CeilAlign(static_cast<int64_t>(varShape_[1] * updatesDtypeSize_),
                                                            ubBlock_);
        updatesDeterminBufferSize_ = colSizeAlignDetermin;
        outBufferDeterminSize_ = colSizeAlignDetermin;
        normalCoreColDetermNum_ = varShape_[1];
        colNumInUbDeterm = varShape_[1];
        colNumInUbDeterm = std::max<int64_t>(1, colNumInUbDeterm);
        coreNumInColDeterm_ = 1;
        tailCoreColNumDeterm_ = varShape_[1];
    } else {
        int64_t resUbForUpdates = ubSize_ - Ops::Base::FloorAlign((coreNumInRow_ * CACHE_ALIGN_SIZE), ubBlock_);
        int64_t copyUpdatesInUb = Ops::Base::FloorAlign(resUbForUpdates / UB_THREE_BUFFER_COUNT, ubBlock_);
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
    normalCoreTailLoopDetermCols_ = normalCoreColDetermNum_ -
                                    (normalCoreColUbDetermLoop_ - 1) * normalCoreNormalLoopDetermCols_;

    tailCoreColUbDetermLoop_ = Ops::Base::CeilDiv(tailCoreColNumDeterm_, colNumInUbDeterm);
    tailCoreNormalLoopDetermCols_ = Ops::Base::CeilDiv(tailCoreColNumDeterm_, tailCoreColUbDetermLoop_);
    tailCoreTailLoopDetermCols_ = tailCoreColNumDeterm_ -
                                  (tailCoreColUbDetermLoop_ - 1) * tailCoreNormalLoopDetermCols_;
}

void ScatterAddWithSortedSimdTiling::SetTilingDataDMA()
{
    tilingDataDMA_ = context_->GetTilingData<ScatterAddWithSortedSimdDMATilingData>();
    tilingDataDMA_->varShape[0] = varShape_[0];
    tilingDataDMA_->varShape[1] = varShape_[1];
    tilingDataDMA_->indicesNum = indicesNum_;

    tilingDataDMA_->rowTileNum = coreNumInRow_;
    tilingDataDMA_->colTileNum = coreNumInCol_;
    tilingDataDMA_->needCoreNum = needCoreNum_;
    tilingDataDMA_->normBlockRow = normalCoreRowNum_;
    tilingDataDMA_->tailBlockRow = tailCoreRowNum_;
    tilingDataDMA_->normBlockCol = normalCoreColNum_;
    tilingDataDMA_->tailBlockCol = tailCoreColNum_;
    tilingDataDMA_->ubFactorRow = normalCoreNormalLoopRows_;
    tilingDataDMA_->ubFactorCol = normalCoreNormalLoopCols_;

    tilingDataDMA_->vecAlignSize = vecAlignSize_;
    tilingDataDMA_->indicesWorkspaceBufferSize = indicesWorkspaceBufferSize_;
    tilingDataDMA_->coreNumInColDeterm = coreNumInColDeterm_;
    tilingDataDMA_->tailCoreColUbDetermLoop = tailCoreColUbDetermLoop_;
    tilingDataDMA_->normalCoreColUbDetermLoop = normalCoreColUbDetermLoop_;
    tilingDataDMA_->tailCoreNormalLoopDetermCols = tailCoreNormalLoopDetermCols_;
    tilingDataDMA_->normalCoreNormalLoopDetermCols = normalCoreNormalLoopDetermCols_;
    tilingDataDMA_->tailCoreTailLoopDetermCols = tailCoreTailLoopDetermCols_;
    tilingDataDMA_->normalCoreTailLoopDetermCols = normalCoreTailLoopDetermCols_;
    tilingDataDMA_->updatesDeterminBufferSize = updatesDeterminBufferSize_;
    tilingDataDMA_->outBufferDeterminSize = outBufferDeterminSize_;
    tilingDataDMA_->normalCoreColDetermNum = normalCoreColDetermNum_;
    tilingDataDMA_->tailCoreColNumDeterm = tailCoreColNumDeterm_;

    tilingDataDMA_->withPos = hasPos_;
}

std::tuple<int64_t, int64_t> ScatterAddWithSortedSimdTiling::SimdDmaAutoTiling(int64_t usedCoreNum, int64_t colNumAlign,
                                                                               int64_t colLimitSize)
{
    /* 给定Shape[M, N] 和 block core num
    ** M切分成m块，N切分成n块，找到m*n <= usedCoreNum, 且m*n尽可能接近usedCoreNum的所有m和n的可能
    */
    std::set<int64_t> cutSet = FindFactor(usedCoreNum);
    std::vector<std::vector<int64_t>> allTiling;

    // 核先按照行方向切分，枚举m的取值
    for (int64_t m : cutSet) {
        // 行方向分核超过行方向数据量，则说明有空闲核
        if (m > static_cast<int64_t>(indicesNum_)) {
            continue;
        }

        int64_t n = usedCoreNum / m;
        n = n < 1 ? 1 : n;
        // 列方向分核超过列方向数据量，则说明有空闲核
        if (n > colNumAlign) {
            continue;
        }

        // ub重排模板，不会在列方向切分
        if (static_cast<int64_t>(varShape_[1] * updatesDtypeSize_) <= colLimitSize &&
            n > 1) { // 当尾轴长度小于设定最小值时，列方向不应切分
            continue;
        }

        int64_t rowNormalBlock = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), m); // 切块后每块行数
        // 列方向按照BASE_A_SIZE基本块切分
        int64_t colNormalBlock = Ops::Base::CeilDiv(colNumAlign, n); // 切块后每块列数

        // 正常切分块和尾块的差值计算
        int64_t delta = rowNormalBlock * colNormalBlock;
        if (m * n ==
            usedCoreNum) { // 选择把核尽量用满的组合。没用满时此条件不满足，delta为最大值，后面排序后不会选中此组合
            if (indicesNum_ % m == 0 &&
                colNumAlign % n == 0) { // 两个方向都能整切，每个块大小相同（理想情况），此时直接返回
                delta = 0;
            } else if (indicesNum_ % m == 0) { // 行方向整切，列不能整切，计算差值
                delta = static_cast<int64_t>(delta) - rowNormalBlock * (colNumAlign % colNormalBlock);
            } else if (colNumAlign % n == 0) { // 列方向整切，行不能整切，计算差值
                delta = static_cast<int64_t>(delta) - (indicesNum_ % rowNormalBlock) * colNormalBlock;
            } else { // 两个方向都不能整切，计算差值
                delta = static_cast<int64_t>(delta) - (indicesNum_ % rowNormalBlock) * (colNumAlign % colNormalBlock);
            }
        }

        allTiling.push_back({m, n, m * n, delta});
    }

    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int NIndex = 1;
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[NIndex], a[DeltaIndex]) < std::make_pair(b[NIndex], b[DeltaIndex]);
    });

    int64_t rowTileNum = allTiling[0][0];
    int64_t colTileNum = allTiling[0][1];
    return std::make_tuple(rowTileNum, colTileNum);
}

void ScatterAddWithSortedSimdTiling::DoBlockTilingDMA(uint64_t baseCol)
{
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    uint64_t colNumAlign = Ops::Base::CeilDiv(varShape_[1], baseCol); // 按设定的4096B切分，列能切分的份数
    needCoreNum_ = std::min(totalCoreNum_,
                            static_cast<uint64_t>(indicesNum_ * colNumAlign * BASE_A_SIZE /
                                                  BASE_BLOCK_SIZE)); // 按每个核最小分4096个数计算使用核数
    needCoreNum_ = needCoreNum_ == 0 ? 1 : needCoreNum_;
    std::tie(coreNumInRow_, coreNumInCol_) = SimdDmaAutoTiling(
        needCoreNum_, colNumAlign,
        COL_LIMIT_SIZE); // 分别计算两个方向切分份数，使得满足：1、尽量把核用满；2、尾核与整核数据量差值尽量小

    // 分别计算行和列的整核/尾核长度，整核列长512B对齐
    normalCoreRowNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), coreNumInRow_); // 每个核行数
    normalCoreColNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(varShape_[1]), coreNumInCol_);
    normalCoreColNum_ = Ops::Base::CeilAlign(
        normalCoreColNum_, static_cast<int64_t>(ubBlock / updatesDtypeSize_)); // 每个核列数(对齐block)
    coreNumInRow_ = Ops::Base::CeilDiv(static_cast<int64_t>(indicesNum_), normalCoreRowNum_);
    coreNumInCol_ = Ops::Base::CeilDiv(static_cast<int64_t>(varShape_[1]), normalCoreColNum_);
    needCoreNum_ = coreNumInRow_ * coreNumInCol_;
    tailCoreRowNum_ = indicesNum_ - (coreNumInRow_ - 1) * normalCoreRowNum_;
    tailCoreColNum_ = varShape_[1] - (coreNumInCol_ - 1) * normalCoreColNum_;
}

bool ScatterAddWithSortedSimdTiling::IsSimdDma()
{
    if (!isAscend96_) {
        return false;
    }
    if (isDeterminTemplate_ == 0) {
        return false;
    }
    if ((varSize_ * varTypeSize_ >= INT32_MAX && indicesDtype_ == ge::DataType::DT_INT32) ||
        (updatesSize_ * updatesDtypeSize_ >= INT32_MAX && posDtype_ == ge::DataType::DT_INT32 &&
         isUpdateScalar_ == 0)) {
        isCastIndices_ = true;
    }

    ubBlock_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));

    uint64_t baseCol = BASE_A_SIZE / updatesDtypeSize_; // 列按BASE_A_SIZE切分
    DoBlockTilingDMA(baseCol); // 计算每个核切分的数量，分别按行/列分整核/尾核

    // 核处理的尾轴小于6k 或者 大于6k需要索引重复度大于20 走此模板
    if (normalCoreColNum_ * updatesDtypeSize_ > DMA_CORE_INNER_SIZE && indicesNum_ / varShape_[0] <= DUP_NUM_DMA) {
        return false;
    }

    // 核处理的行数大于等于16行走此模板
    if (normalCoreRowNum_ < DMA_MIN_CORE_ROW) {
        return false;
    }

    // ub至少能放得下24行走DMA模板
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    FrontAndBackIndex = Ops::Base::CeilAlign(static_cast<int64_t>(indicesDtypeSize_), ubBlock_);

    int64_t minIndicesUbFactor = MIN_INDICES_NUM_DMA;
    int64_t minUpdatesUbSize = minIndicesUbFactor * normalCoreColNum_ * updatesDtypeSize_ *
                               BUFFER_NUM_SIMD_SORT; // updates
    int64_t minResUbSize = minIndicesUbFactor * normalCoreColNum_ * updatesDtypeSize_ *
                           BUFFER_NUM_SIMD_SORT; // add res updates
    int64_t minIndicesUbSize = (Ops::Base::CeilAlign(minIndicesUbFactor * indicesDtypeSize_, ubBlock) +
                                SORT_STAT_PADDING) *
                               BUFFER_NUM_SIMD_SORT; // sorted indices
    int64_t minPosUbSize = (isUpdateScalar_ == 0 && hasPos_) ?
                               Ops::Base::CeilAlign(minIndicesUbFactor * posDtypeSize_, ubBlock) *
                                   BUFFER_NUM_SIMD_SORT :
                               0; // pos
    int64_t minUniqueIdCntUbSize = Ops::Base::CeilAlign((minIndicesUbFactor + 1) * sizeof(int32_t),
                                                        ubBlock); // uniqueIdCount
    int64_t minUpdateSumIdxUbSize = Ops::Base::CeilAlign((minIndicesUbFactor + 1) * indicesDtypeSize_,
                                                         ubBlock); // updateSumIdx
    int64_t castIndicesSize = isCastIndices_ ? Ops::Base::CeilAlign(minIndicesUbFactor * sizeof(int64_t), ubBlock) :
                                               0; // indices cast
    int64_t castPosSize = (isCastIndices_ && isUpdateScalar_ == 0 && hasPos_) ?
                              Ops::Base::CeilAlign(minIndicesUbFactor * sizeof(int64_t), ubBlock) :
                              0; // pos cast
    uint64_t needUbSize = minUpdatesUbSize + minResUbSize + minIndicesUbSize + minPosUbSize + minUniqueIdCntUbSize +
                          minUpdateSumIdxUbSize + castIndicesSize + castPosSize + FrontAndBackIndex * NUM_TWO +
                          ubBlock +
                          normalCoreColNum_ *
                              updatesDtypeSize_; // 核前后id + workspace id 2个 + 一行存放循环结束最后一个id的累加结果
    bool isSimdDma = ubSize_ >= needUbSize;
    return isSimdDma;
}

/**
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
uint64_t ScatterAddWithSortedSimdTiling::CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd) const
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t indicesDtypeCastSize = isCastIndices_ ? sizeof(int64_t) : 0;
    int64_t posDtypeCastSize = (isCastIndices_ && isUpdateScalar_ == 0 && hasPos_) ? sizeof(int64_t) : 0;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / NUM_TWO;

        tmpTotalSize = Ops::Base::CeilAlign(baseXoMid * normalCoreNormalLoopCols_ * updatesDtypeSize_, ubBlock) *
                           BUFFER_NUM_SIMD_SORT + // updates
                       (Ops::Base::CeilAlign(baseXoMid * indicesDtypeSize_, ubBlock) + SORT_STAT_PADDING) *
                           BUFFER_NUM_SIMD_SORT +                                                        // indices
                       Ops::Base::CeilAlign(baseXoMid * posDtypeSize_, ubBlock) * BUFFER_NUM_SIMD_SORT + // pos
                       Ops::Base::CeilAlign(baseXoMid * indicesDtypeCastSize, ubBlock) +                 // indices cast
                       Ops::Base::CeilAlign(baseXoMid * posDtypeCastSize, ubBlock) +                     // pos cast
                       Ops::Base::CeilAlign(baseXoMid * normalCoreNormalLoopCols_ * updatesDtypeSize_, ubBlock) *
                           BUFFER_NUM_SIMD_SORT + // 存放相加后的临时变量
                       Ops::Base::CeilAlign((baseXoMid + 1) * sizeof(int32_t),
                                            ubBlock) + // 每个唯一索引重复个数 uniqueIdCountQue
                       Ops::Base::CeilAlign((baseXoMid + 1) * indicesDtypeSize_,
                                            ubBlock); // 每个唯一索引的索引值 updateSumIdxQue
        if (tmpTotalSize <= ubSize_) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

ge::graphStatus ScatterAddWithSortedSimdTiling::SimdDmaTiling()
{
    isModeDMA_ = true;
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    ubSize_ = ubSize_ - FrontAndBackIndex * NUM_TWO - ubBlock - normalCoreColNum_ * updatesDtypeSize_;

    normalCoreNormalLoopCols_ = normalCoreColNum_;

    // 预估ub能容纳行的上界(未考虑索引去重的辅助buffer)
    uint64_t maxBaseRow = (ubSize_ - NUM_TWO * BUFFER_NUM_SIMD_SORT * ubBlock) / BUFFER_NUM_SIMD_SORT /
                          (normalCoreNormalLoopCols_ * updatesDtypeSize_ * NUM_TWO + indicesDtypeSize_ + posDtypeSize_);

    // 通过二分法计算排序情况下的单循环行数
    normalCoreNormalLoopRows_ = CalBestBaseSize(1, maxBaseRow);
    // 在单ub循环行数超过block行数时，将单ub循环行数缩小至block行数
    if (normalCoreNormalLoopRows_ > normalCoreRowNum_) {
        normalCoreNormalLoopRows_ = normalCoreRowNum_;
    }

    // GatherDMA的行数不超过4095
    if (normalCoreNormalLoopRows_ > SCATTER_DMA_MAX_ROW) {
        normalCoreNormalLoopRows_ = SCATTER_DMA_MAX_ROW;
    }

    DeterminTemplateUbTiling();

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void ScatterAddWithSortedSimdTiling::SetTilingData()
{
    if (isModeDMA_) {
        SetTilingDataDMA();
        return;
    }
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

    if (IsSimdDma()) {
        return SimdDmaTiling();
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

    if (isModeDMA_) {
        uint64_t isCast = isCastIndices_ ? TPL_ADDR_B64 : TPL_ADDR_B32;
        return GET_TPL_TILING_KEY(TPL_MODE_SIMD_DMA, TPL_SCALAR_FALSE, TPL_DETERM_TRUE, isCast);
    }

    uint64_t isDeterm = isDeterminTemplate_ ? TPL_DETERM_TRUE : TPL_DETERM_FALSE;
    return GET_TPL_TILING_KEY(TPL_MODE_SIMD, isScalar, isDeterm, TPL_ADDR_B32);
}

ge::graphStatus ScatterAddWithSortedSimdTiling::GetWorkspaceSize()
{
    size_t useWorkspace = 0;
    if (isDeterminTemplate_) {
        useWorkspace += coreNumInRow_ * BUFFER_NUM * vecAlignSize_ + coreNumInRow_ * CACHE_ALIGN_SIZE;
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

void ScatterAddWithSortedSimdTiling::DumpTilingInfoDMA()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", UB Size: " << ubSize_;
    info << ", varShape[0]: " << tilingDataDMA_->varShape[0];
    info << ", varShape[1]: " << tilingDataDMA_->varShape[1];
    info << ", indicesNum: " << tilingDataDMA_->indicesNum;
    info << ", rowTileNum: " << tilingDataDMA_->rowTileNum;
    info << ", colTileNum: " << tilingDataDMA_->colTileNum;
    info << ", needCoreNum: " << tilingDataDMA_->needCoreNum;

    info << ", normBlockRow: " << tilingDataDMA_->normBlockRow;
    info << ", tailBlockRow: " << tilingDataDMA_->tailBlockRow;
    info << ", normBlockCol: " << tilingDataDMA_->normBlockCol;
    info << ", tailBlockCol: " << tilingDataDMA_->tailBlockCol;
    info << ", ubFactorRow: " << tilingDataDMA_->ubFactorRow;
    info << ", ubFactorCol: " << tilingDataDMA_->ubFactorCol;

    info << ", vecAlignSize: " << tilingDataDMA_->vecAlignSize;
    info << ", indicesWorkspaceBufferSize: " << tilingDataDMA_->indicesWorkspaceBufferSize;
    info << ", coreNumInColDeterm: " << tilingDataDMA_->coreNumInColDeterm;
    info << ", tailCoreColUbDetermLoop: " << tilingDataDMA_->tailCoreColUbDetermLoop;
    info << ", normalCoreColUbDetermLoop: " << tilingDataDMA_->normalCoreColUbDetermLoop;
    info << ", tailCoreNormalLoopDetermCols: " << tilingDataDMA_->tailCoreNormalLoopDetermCols;
    info << ", normalCoreNormalLoopDetermCols: " << tilingDataDMA_->normalCoreNormalLoopDetermCols;
    info << ", tailCoreTailLoopDetermCols: " << tilingDataDMA_->tailCoreTailLoopDetermCols;
    info << ", normalCoreTailLoopDetermCols: " << tilingDataDMA_->normalCoreTailLoopDetermCols;
    info << ", updatesDeterminBufferSize: " << tilingDataDMA_->updatesDeterminBufferSize;
    info << ", outBufferDeterminSize: " << tilingDataDMA_->outBufferDeterminSize;
    info << ", normalCoreColDetermNum: " << tilingDataDMA_->normalCoreColDetermNum;
    info << ", tailCoreColNumDeterm: " << tilingDataDMA_->tailCoreColNumDeterm;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void ScatterAddWithSortedSimdTiling::DumpTilingInfo()
{
    if (isModeDMA_) {
        DumpTilingInfoDMA();
        return;
    }
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
