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
 * \file segment_sum_simd_tiling.cpp
 * \brief segment_sum_simd_tiling
 */
 
#include "segment_sum_simd_tiling.h"

namespace optiling {

const static uint64_t SIMD_ATOMIC_SUPPORT_KEY = 2000;
const static uint64_t SIMD_DETERM_KEY = 2002;
static constexpr int64_t SIMD_INNER_THRES = 32;
static constexpr int64_t BLOCK_TILING_THRES = 512;
static constexpr int64_t INNER_ADD_NUM = 128;
static constexpr int64_t BASE_BLOCK_ALIGN = 512;
static constexpr int64_t SINGLE_CORE_THRESHOLD = 4 * 1024;
static constexpr int64_t NUM_TWO = 2;
static constexpr uint64_t NUM_FOUR = 4;
static constexpr uint64_t MIN_INNER_SIZE = 256;
static constexpr size_t WS_SYS_SIZE = static_cast<size_t>(16 * 1024 * 1024);

static const std::set<ge::DataType> setAtomicSupportSimd = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT16, ge::DT_INT32, ge::DT_INT8, ge::DT_BF16};

bool SegmentSumSimdTiling::IsAtomicSupport()
{
    if (setAtomicSupportSimd.find(dataType_) != setAtomicSupportSimd.end()) {
        return true;
    }
    return false;
}

bool SegmentSumSimdTiling::IsCapable() 
{
    bool isFloat = (dataType_ == ge::DT_FLOAT || dataType_ == ge::DT_FLOAT16 || dataType_ == ge::DT_BF16);
    isAtomicSupport_ = IsAtomicSupport();
    isDeterministic_ = context_->GetDeterministic() == 1 && isFloat;
    bool isInnerSimd = innerDim_ * valueTypeBytes_ > SIMD_INNER_THRES;
    return isInnerSimd && isAtomicSupport_;
}


std::set<int64_t> ListFactors(int64_t usedCoreNum)
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

void SegmentSumSimdTiling::AutoTilingRowCol(int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum, int64_t rowTotalNum, int64_t colTotalNum)
{
    int64_t tmpEleNum = BASE_BLOCK_ALIGN / valueTypeBytes_;
    int64_t colBlockTotalNum = (colTotalNum + tmpEleNum - 1) / tmpEleNum;
    usedCoreNum = std::min(usedCoreNum, std::max(int64_t(1), rowTotalNum * colBlockTotalNum * tmpEleNum / (SINGLE_CORE_THRESHOLD)));

    std::set<int64_t> cutSet = ListFactors(usedCoreNum);
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
        constexpr int MIndex = 0;
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[MIndex]) < std::make_pair(b[DeltaIndex], b[MIndex]);
    });

    while (allTiling.size() > 1 && outerDim_ / allTiling[0][0] < std::min(NUM_FOUR, outerDim_)) {
        allTiling.erase(allTiling.begin());
    }

    rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    colTileNum = static_cast<uint16_t>(allTiling[0][1]);
}

void SegmentSumSimdTiling::DoBlockTiling()
{
    // 如果inner小于512B，分核及切UB均不使用inner
    if (innerDim_ * valueTypeBytes_ <= BLOCK_TILING_THRES) {
        blockNumInCol_ = 1;
        normalCoreInnerNum_ = innerDim_;
        tailCoreInnerNum_ = innerDim_;
        blockNumInRow_ = std::min(totalCoreNum_, outerDim_);
        normalCoreOutterNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(outerDim_), blockNumInRow_);
        blockNumInRow_ = Ops::Base::CeilDiv(static_cast<int64_t>(outerDim_), normalCoreOutterNum_);
        tailCoreOutterNum_ = outerDim_ - (blockNumInRow_ - 1) * normalCoreOutterNum_;
    } else {
        // inner大于512B，动态分核，使各核负载均衡且分的inner尽可能小
        int64_t rowTileNum = 0;
        int64_t colTileNum = 0;
        AutoTilingRowCol(rowTileNum, colTileNum, totalCoreNum_, outerDim_, innerDim_);

        normalCoreOutterNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(outerDim_), rowTileNum);
        blockNumInRow_ = Ops::Base::CeilDiv(static_cast<int64_t>(outerDim_), normalCoreOutterNum_);
        tailCoreOutterNum_ = outerDim_ - (blockNumInRow_ - 1) * normalCoreOutterNum_;

        normalCoreInnerNum_ = Ops::Base::CeilDiv(static_cast<int64_t>(innerDim_), colTileNum);
        blockNumInCol_ = Ops::Base::CeilDiv(static_cast<int64_t>(innerDim_), normalCoreInnerNum_);
        tailCoreInnerNum_ = innerDim_ - (blockNumInCol_ - 1) * normalCoreInnerNum_;
    }
    needCoreNum_ = blockNumInRow_ * blockNumInCol_;

    // outGM清零
    int64_t totalOutputSize = segmentNum_ * innerDim_;
    normalCoreClearNum_ = Ops::Base::CeilDiv(totalOutputSize, needCoreNum_);
    usedCoreNumForClear_ = Ops::Base::CeilDiv(totalOutputSize, normalCoreClearNum_);
    tailCoreClearNum_ = totalOutputSize - (usedCoreNumForClear_ - 1) * normalCoreClearNum_;
}

void SegmentSumSimdTiling::DoSplitColUBTiling(int64_t availableUbsize)
{
    // inner大于512B，UB初始inner为512B，如果行比较小不够分，则inner每次增加128B
    isSplitCol_ = true;
    int64_t innerSizeAlign = Ops::Base::CeilAlign(normalCoreInnerNum_ * valueTypeBytes_, ubBlockSize_);
    int64_t tmpColSize = BLOCK_TILING_THRES + idTypeBytes_;
    int64_t tmpRowNum = availableUbsize / tmpColSize;
    while (tmpRowNum - 1 > normalCoreOutterNum_) {
        if (tmpColSize >= innerSizeAlign + idTypeBytes_) {
            break;
        }
        tmpColSize += INNER_ADD_NUM;
        tmpRowNum = availableUbsize / tmpColSize;
    }
    int64_t rowNumInUb = std::min(tmpRowNum - 1, normalCoreOutterNum_);
    int64_t colSizeInUb = std::min(static_cast<int64_t>(tmpColSize - idTypeBytes_), innerSizeAlign);

    xBufferSize_ = rowNumInUb * colSizeInUb;
    segmentIdBufferSize_ = Ops::Base::CeilAlign(rowNumInUb * idTypeBytes_, ubBlockSize_);
    yBufferSize_ = colSizeInUb;

    normalCoreRowUbLoop_ = Ops::Base::CeilDiv(normalCoreOutterNum_, rowNumInUb);
    normalCoreNormalLoopOutters_ = Ops::Base::CeilDiv(normalCoreOutterNum_, normalCoreRowUbLoop_);
    normalCoreTailLoopOutters_ = normalCoreOutterNum_ - (normalCoreRowUbLoop_ - 1) * normalCoreNormalLoopOutters_;
    tailCoreRowUbLoop_ = Ops::Base::CeilDiv(tailCoreOutterNum_, rowNumInUb);
    tailCoreNormalLoopOutters_ = Ops::Base::CeilDiv(tailCoreOutterNum_, tailCoreRowUbLoop_);
    tailCoreTailLoopOutters_ = tailCoreOutterNum_ - (tailCoreRowUbLoop_ - 1) * tailCoreNormalLoopOutters_;


    int64_t colNumInUb = colSizeInUb / valueTypeBytes_;
    normalCoreColUbLoop_ = Ops::Base::CeilDiv(normalCoreInnerNum_, colNumInUb);
    normalCoreNormalLoopInners_ = Ops::Base::CeilDiv(normalCoreInnerNum_, normalCoreColUbLoop_);
    normalCoreTailLoopInners_ = normalCoreInnerNum_ - (normalCoreColUbLoop_ - 1) * normalCoreNormalLoopInners_;

    tailCoreColUbLoop_ = Ops::Base::CeilDiv(tailCoreInnerNum_, colNumInUb);
    tailCoreNormalLoopInners_ = Ops::Base::CeilDiv(tailCoreInnerNum_, tailCoreColUbLoop_);
    tailCoreTailLoopInners_ = tailCoreInnerNum_ - (tailCoreColUbLoop_ - 1) * tailCoreNormalLoopInners_;
}

void SegmentSumSimdTiling::DoUBTiling()
{
    int64_t availableUbsize =  ubSize_;

    availableUbsize -= ubBlockSize_ - idTypeBytes_; // 预留给segmentId对齐
    // 如果inner小于512B，分核及切UB均不使用inner
    if (normalCoreInnerNum_ * valueTypeBytes_ <= BLOCK_TILING_THRES) {
        int64_t innerSizeAlign = Ops::Base::CeilAlign(normalCoreInnerNum_ * valueTypeBytes_, ubBlockSize_);
        int64_t tmpColSize = innerSizeAlign + idTypeBytes_;
        int64_t tmpRowNum = availableUbsize / tmpColSize;
        int64_t rowNumInUb = std::min(tmpRowNum - 1, normalCoreOutterNum_);

        xBufferSize_ = rowNumInUb * innerSizeAlign;
        segmentIdBufferSize_ = Ops::Base::CeilAlign(rowNumInUb * idTypeBytes_, ubBlockSize_);
        yBufferSize_ = innerSizeAlign;

        normalCoreRowUbLoop_ = Ops::Base::CeilDiv(normalCoreOutterNum_, rowNumInUb);
        normalCoreNormalLoopOutters_ = Ops::Base::CeilDiv(normalCoreOutterNum_, normalCoreRowUbLoop_);
        normalCoreTailLoopOutters_ = normalCoreOutterNum_ - (normalCoreRowUbLoop_ - 1) * normalCoreNormalLoopOutters_;
        tailCoreRowUbLoop_ = Ops::Base::CeilDiv(tailCoreOutterNum_, rowNumInUb);
        tailCoreNormalLoopOutters_ = Ops::Base::CeilDiv(tailCoreOutterNum_, tailCoreRowUbLoop_);
        tailCoreTailLoopOutters_ = tailCoreOutterNum_ - (tailCoreRowUbLoop_ - 1) * tailCoreNormalLoopOutters_;

        normalCoreColUbLoop_ = 1;
        normalCoreNormalLoopInners_ = normalCoreInnerNum_;
        normalCoreTailLoopInners_ = normalCoreInnerNum_;

        tailCoreColUbLoop_ = 1;
        tailCoreNormalLoopInners_ = tailCoreInnerNum_;
        tailCoreTailLoopInners_ = tailCoreInnerNum_;
    } else {
        DoSplitColUBTiling(availableUbsize);
    }
}

void SegmentSumSimdTiling::DoMultCoreAddTiling()
{
    if (innerDim_ * valueTypeBytes_ <= BLOCK_TILING_THRES) {
        usedCoreNumForMultAdd_ = 1;
        normalCoreMultAddInners_ = innerDim_;
        tailCoreMultAddInners_ = innerDim_;
    } else {
        uint64_t minInners = MIN_INNER_SIZE / valueTypeBytes_;
        normalCoreMultAddInners_ = Ops::Base::CeilAlign(std::max(minInners, Ops::Base::CeilDiv(innerDim_, static_cast<uint64_t>(needCoreNum_))), minInners); // 一个核至少处理256B的inner 且Align(256B)
        usedCoreNumForMultAdd_ = Ops::Base::CeilDiv(static_cast<int64_t>(innerDim_), normalCoreMultAddInners_);
        tailCoreMultAddInners_ = innerDim_ - (usedCoreNumForMultAdd_ - 1) * normalCoreMultAddInners_;
    }
    int64_t mulAddUbsize =  ubSize_;
    mulAddUbsize /= blockNumInRow_ * NUM_TWO + 1;
    int64_t availableInnerUb = Ops::Base::FloorAlign(mulAddUbsize, static_cast<int64_t>(ubBlockSize_));

    int64_t innerNumInUb = availableInnerUb / valueTypeBytes_;
    normalCoreMultAddInnerLoop_ = Ops::Base::CeilDiv(normalCoreMultAddInners_, innerNumInUb);
    normalCoreMultAddNormalLoopInners_ = Ops::Base::CeilDiv(normalCoreMultAddInners_, normalCoreMultAddInnerLoop_);
    normalCoreMultAddTailLoopInners_ = normalCoreMultAddInners_ - (normalCoreMultAddInnerLoop_ - 1) * normalCoreMultAddNormalLoopInners_;

    tailCoreMultAddInnerLoop_ = Ops::Base::CeilDiv(tailCoreMultAddInners_, innerNumInUb);
    tailCoreMultAddNormalLoopInners_ = Ops::Base::CeilDiv(tailCoreMultAddInners_, tailCoreMultAddInnerLoop_);
    tailCoreMultAddTailLoopInners_ = tailCoreMultAddInners_ - (tailCoreMultAddInnerLoop_ - 1) * tailCoreMultAddNormalLoopInners_;

    multAddXBufferSize_ = blockNumInRow_ * NUM_TWO * availableInnerUb;
    multAddYBufferSize_ = availableInnerUb;
}

ge::graphStatus SegmentSumSimdTiling::DoOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "[SegmentSum] GetDeterministic state: %u", context_->GetDeterministic());

    DoBlockTiling();
    DoUBTiling();

    if (isDeterministic_ || !isAtomicSupport_) {
        DoMultCoreAddTiling();
    }

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SegmentSumSimdTiling::SetTilingData()
{
    tilingData_ = context_->GetTilingData<SegmentSumSimdTilingData>();
    tilingData_->needCoreNum = needCoreNum_;
    tilingData_->outerDim = outerDim_;
    tilingData_->innerDim = innerDim_;

    tilingData_->xBufferSize = xBufferSize_;
    tilingData_->segmentIdBufferSize = segmentIdBufferSize_;
    tilingData_->yBufferSize = yBufferSize_;

    tilingData_->usedCoreNumForClear = usedCoreNumForClear_;
    tilingData_->normalCoreClearNum = normalCoreClearNum_;
    tilingData_->tailCoreClearNum = tailCoreClearNum_;

    tilingData_->blockNumInRow = blockNumInRow_;
    tilingData_->blockNumInCol = blockNumInCol_;

    tilingData_->normalCoreInnerNum = normalCoreInnerNum_;
    tilingData_->tailCoreInnerNum = tailCoreInnerNum_;
    tilingData_->normalCoreOutterNum = normalCoreOutterNum_;
    tilingData_->tailCoreOutterNum = tailCoreOutterNum_;

    tilingData_->normalCoreRowUbLoop = normalCoreRowUbLoop_;
    tilingData_->normalCoreNormalLoopOutters = normalCoreNormalLoopOutters_;
    tilingData_->normalCoreTailLoopOutters = normalCoreTailLoopOutters_;
    tilingData_->tailCoreRowUbLoop = tailCoreRowUbLoop_;
    tilingData_->tailCoreNormalLoopOutters = tailCoreNormalLoopOutters_;
    tilingData_->tailCoreTailLoopOutters = tailCoreTailLoopOutters_;

    tilingData_->normalCoreColUbLoop = normalCoreColUbLoop_;
    tilingData_->normalCoreNormalLoopInners = normalCoreNormalLoopInners_;
    tilingData_->normalCoreTailLoopInners = normalCoreTailLoopInners_;
    tilingData_->tailCoreColUbLoop = tailCoreColUbLoop_;
    tilingData_->tailCoreNormalLoopInners = tailCoreNormalLoopInners_;
    tilingData_->tailCoreTailLoopInners = tailCoreTailLoopInners_;

    tilingData_->usedCoreNumForMultAdd = usedCoreNumForMultAdd_;
    tilingData_->normalCoreMultAddInners = normalCoreMultAddInners_;
    tilingData_->tailCoreMultAddInners = tailCoreMultAddInners_;

    tilingData_->normalCoreMultAddInnerLoop = normalCoreMultAddInnerLoop_;
    tilingData_->normalCoreMultAddNormalLoopInners = normalCoreMultAddNormalLoopInners_;
    tilingData_->normalCoreMultAddTailLoopInners = normalCoreMultAddTailLoopInners_;
    tilingData_->tailCoreMultAddInnerLoop = tailCoreMultAddInnerLoop_;
    tilingData_->tailCoreMultAddNormalLoopInners = tailCoreMultAddNormalLoopInners_;
    tilingData_->tailCoreMultAddTailLoopInners = tailCoreMultAddTailLoopInners_;

    tilingData_->multAddXBufferSize = multAddXBufferSize_;
    tilingData_->multAddYBufferSize = multAddYBufferSize_;
    return;
}

uint64_t SegmentSumSimdTiling::GetTilingKey() const
{
    return SIMD_ATOMIC_SUPPORT_KEY;
}

ge::graphStatus SegmentSumSimdTiling::GetWorkspaceSize()
{
    auto useWorkspace = WS_SYS_SIZE; // 可以不用初值
    if (isDeterministic_ || !isAtomicSupport_) {
        useWorkspace += blockNumInRow_ * NUM_TWO * (innerDim_ * valueTypeBytes_ + idTypeBytes_) + idTypeBytes_; // 对齐idTypeBytes_  头尾id最好需要间隔 cache line
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = useWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SegmentSumSimdTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "SegmentSum simd PostTiling enter.");
    context_->SetBlockDim(needCoreNum_);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void SegmentSumSimdTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", UB Size: " << ubSize_;
    info << ", needCoreNum: " << tilingData_->needCoreNum;
    info << ", outerDim: " << tilingData_->outerDim;
    info << ", innerDim: " << tilingData_->innerDim;

    info << ", xBufferSize: " << tilingData_->xBufferSize;
    info << ", segmentIdBufferSize: " << tilingData_->segmentIdBufferSize;
    info << ", yBufferSize: " << tilingData_->yBufferSize;

    info << ", usedCoreNumForClear: " << tilingData_->usedCoreNumForClear;
    info << ", normalCoreClearNum: " << tilingData_->normalCoreClearNum;
    info << ", tailCoreClearNum: " << tilingData_->tailCoreClearNum;

    info << ", blockNumInRow: " << tilingData_->blockNumInRow;
    info << ", blockNumInCol: " << tilingData_->blockNumInCol;

    info << ", normalCoreInnerNum: " << tilingData_->normalCoreInnerNum;
    info << ", tailCoreInnerNum: " << tilingData_->tailCoreInnerNum;
    info << ", normalCoreOutterNum: " << tilingData_->normalCoreOutterNum;
    info << ", tailCoreOutterNum: " << tilingData_->tailCoreOutterNum;
    
    info << ", normalCoreRowUbLoop: " << tilingData_->normalCoreRowUbLoop;
    info << ", normalCoreNormalLoopOutters: " << tilingData_->normalCoreNormalLoopOutters;
    info << ", normalCoreTailLoopOutters: " << tilingData_->normalCoreTailLoopOutters;
    info << ", tailCoreRowUbLoop: " << tilingData_->tailCoreRowUbLoop;
    info << ", tailCoreNormalLoopOutters: " << tilingData_->tailCoreNormalLoopOutters;
    info << ", tailCoreTailLoopOutters: " << tilingData_->tailCoreTailLoopOutters;

    info << ", normalCoreColUbLoop: " << tilingData_->normalCoreColUbLoop;
    info << ", normalCoreNormalLoopInners: " << tilingData_->normalCoreNormalLoopInners;
    info << ", normalCoreTailLoopInners: " << tilingData_->normalCoreTailLoopInners;
    info << ", tailCoreColUbLoop: " << tilingData_->tailCoreColUbLoop;
    info << ", tailCoreNormalLoopInners: " << tilingData_->tailCoreNormalLoopInners;
    info << ", tailCoreTailLoopInners: " << tilingData_->tailCoreTailLoopInners;

    info << ", usedCoreNumForMultAdd: " << tilingData_->usedCoreNumForMultAdd;
    info << ", normalCoreMultAddInners: " << tilingData_->normalCoreMultAddInners;
    info << ", tailCoreMultAddInners: " << tilingData_->tailCoreMultAddInners;

    info << ", normalCoreMultAddInnerLoop: " << tilingData_->normalCoreMultAddInnerLoop;
    info << ", normalCoreMultAddNormalLoopInners: " << tilingData_->normalCoreMultAddNormalLoopInners;
    info << ", normalCoreMultAddTailLoopInners: " << tilingData_->normalCoreMultAddTailLoopInners;
    info << ", tailCoreMultAddInnerLoop: " << tilingData_->tailCoreMultAddInnerLoop;
    info << ", tailCoreMultAddNormalLoopInners: " << tilingData_->tailCoreMultAddNormalLoopInners;
    info << ", tailCoreMultAddTailLoopInners: " << tilingData_->tailCoreMultAddTailLoopInners;

    info << ", multAddXBufferSize: " << tilingData_->multAddXBufferSize;
    info << ", multAddYBufferSize: " << tilingData_->multAddYBufferSize;
    
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("SegmentSum", SegmentSumSimdTiling, 0);
}