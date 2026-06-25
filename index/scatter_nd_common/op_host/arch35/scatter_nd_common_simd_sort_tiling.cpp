/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_common_simd_sort_tiling.cpp
 * \brief scatter_nd_common_simd_sort_tiling
 */

#include "scatter_nd_common_simd_sort_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

using namespace ScatterNdCommon;

static constexpr int64_t MIN_SIZE_SIMD_NONDETERMINSTIC = 128;
static constexpr int64_t TEMPLATE_MODE = 2;
static constexpr int64_t SIMD_NONDETERMINSTIC_MIN_HANDLE_SIZE = 4096;
static constexpr int64_t SPLIT_THRESHOLD = 64;
static constexpr int64_t ALIGN_SIZE = 32;
static constexpr int64_t MIN_HANDLE_SIZE = 128;
static constexpr int64_t FP32_BYTES = 4;
static constexpr int64_t INT32_BYTES = 4;
static constexpr uint64_t RESERVE_SIZE = 256;
static constexpr int64_t CUT_THRESHOLD = 128;
static constexpr int64_t INDICES_MIN_BLOCK_SIZE = 1024;
static constexpr uint64_t DB_BUFFER = 2;

static const std::set<ge::DataType> setAtomicNotSupport = {ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64};

bool ScatterNdCommonSimdSortTiling::IsCapable()
{
    if (afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_NONDETERMINSTIC &&
        setAtomicNotSupport.find(updateDtype_) == setAtomicNotSupport.end()) {
        return true;
    }
    if (updateDtype_ == ge::DT_INT8 || updateDtype_ == ge::DT_INT16) {
        return true;
    }
    return false;
}

uint64_t ScatterNdCommonSimdSortTiling::GetTilingKey() const
{
    OP_LOGD("ScatterNdCommonSimdSortTiling::GetTilingKey begin");
    uint64_t addrMode = 1;
    if (varInAxis_ < INT32_MAX) { // rank维索引合一可能超过int32最大值
        addrMode = 0;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, indiceCastMode_, addrMode);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

void ScatterNdCommonSimdSortTiling::SetTilingData()
{
    ScatterNdCommon::ScatterNdCommonSimdSortTilingData *tilingData = 
        context_->GetTilingData<ScatterNdCommon::ScatterNdCommonSimdSortTilingData>();

    for (int32_t i = 0; i < OPTILING_MAX_RANK_COUNT; i++) {
        tilingData->strideList[i] = strideList_[i];
    }
    for (int32_t i = 0; i < OPTILING_MAX_SHAPE_RANK; i++) {
        tilingData->outPutShape[i] = outPutShape_[i];
    }
    tilingData->eachCoreAfterAxisCount = eachCoreAfterAxisCount_;
    tilingData->indexRankSize = rankSize_;
    tilingData->eachCoreIndexCount = eachCoreIndexCount_;
    tilingData->tailCoreIndexCount = tailCoreIndexCount_;
    tilingData->indicesFactor = indicesFactor_;
    tilingData->indiceTailNum = indiceTailNum_;
    tilingData->indicesLoopSize = indicesLoopSize_;
    tilingData->afterAxis = afterAxis_;
    tilingData->afterAxisFactor = afterAxisFactor_;
    tilingData->usedCoreNumBefore = usedCoreNumBefore_;
    tilingData->updateLoopSize = updateLoopSize_;
    tilingData->tailUpdateLoopSize = tailUpdateLoopSize_;
    tilingData->tailUpdateTailNum = tailUpdateTailNum_;
    tilingData->updateTailNum = updateTailNum_;
    tilingData->isSplitAfterAxis = isSplitAfterAxis_;
    tilingData->singleCol = singleCol_;
}

void ScatterNdCommonSimdSortTiling::DoOpTilingSplitIndicesSingleCol()
{
    int64_t indicesDtypeCastSize = indiceCastMode_ ? indiceCastDtypeSize_ : 0;
    int64_t indicesRealTypeSize = indiceCastMode_ ? indiceCastDtypeSize_ : outOfSetTypeSize_;
    ge::DataType sortTmpType = indiceCastMode_ ?  indiceCastDtype_ : outOfSetDtype_;

    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE - MIN_HANDLE_SIZE * FP32_BYTES) / DB_BUFFER);
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;

    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t indicesAlignSize = Ops::Base::CeilAlign(oneIndexSize, ubBlock) +             // indices
                                Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) +        // outOfset
                                Ops::Base::CeilAlign(indicesDtypeCastSize, ubBlock) +    // cast_indices
                                Ops::Base::CeilAlign(indicesRealTypeSize + TWO * ALIGN_SIZE, ubBlock) + 
                                Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + 
                                Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +
                                Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock);

    int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) + 
                                Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) + 
                                GetSortTmpSize(sortTmpType, SPLIT_THRESHOLD, false);
    //每次最少搬入indices数
    int64_t minIndicesFactorSize = indicesAlignSize * SPLIT_THRESHOLD;
    // 搬入多行indices，搬入一行updates
    if (minIndicesFactorSize + updateAlignSize > halfUbSize) { // 优先保证indices至少处理64行，实际 afterAxisFactor_ 按剩余ub给一行updates
        indicesFactor_ = SPLIT_THRESHOLD;
        int64_t sortTmpSize = GetSortTmpSize(sortTmpType, SPLIT_THRESHOLD, false);
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesAlignSize - sortTmpSize) / (varTypeSize_ + varTypeSize_); // (update + updateSum)
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else { // 完整处理一行 afterAxis_,逐步调整 indicesFactor_ 以满足UB约束
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        indicesFactor_ = (halfUbSize - updateAlignSize) / indicesAlignSize;
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            int64_t occupy = Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                            Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) +
                            Ops::Base::CeilAlign(indicesFactor_ * indicesDtypeCastSize, ubBlock) + 
                            Ops::Base::CeilAlign(indicesFactor_ * (indicesRealTypeSize + TWO * ALIGN_SIZE), ubBlock) + 
                            Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) + 
                            Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES), ubBlock) +
                            Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) + 
                            Ops::Base::CeilAlign(varTypeSize_ * afterAxisFactor_, ubBlock) +
                            Ops::Base::CeilAlign(varTypeSize_ * afterAxisFactor_, ubBlock) + 
                            GetSortTmpSize(sortTmpType, indicesFactor_, false);
            restSize = halfUbSize - occupy;
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
            --indicesFactor_;
        }
    }
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
    singleCol_ = 1;
}

void ScatterNdCommonSimdSortTiling::DoOpTilingSplitAfter()
{
    int64_t ubSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE));
    ubSize = ubSize - MIN_HANDLE_SIZE * FP32_BYTES;//maxScore
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t afterAxisSize = afterAxis_ * varTypeSize_;
    /* split afterAxis */
    if (afterAxisSize < CUT_THRESHOLD) { // 尾轴不大，索引也不大，开1个核
        eachCoreAfterAxisCount_ = afterAxis_;
        usedCoreNumBefore_ = ONE;
        tailCoreAfterAxisCount_ = afterAxis_;
    } else {
        eachCoreAfterAxisCount_ = Ops::Base::CeilDiv(afterAxis_, totalCoreNum_); // 正常核处理的尾轴个数
        usedCoreNumBefore_ = Ops::Base::CeilDiv(afterAxis_, eachCoreAfterAxisCount_); // 使用核数
        tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1); // 尾核处理的尾轴个数
    }

    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
    * indicesFactor_: outOfsetBuf + indiecesQue + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
    *                 (uniqueIdCntQue_ + 1) + updateSumIdxQue_,
    * indicesFactor_ * eachCoreAfterAxisCount_: updatesQue_ + updateSumQue_
    */
    int64_t indicesSize = 0;
    if (!indiceCastMode_) {
        indicesSize = Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +       // indices
                            Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) +             // outofset
                            Ops::Base::CeilAlign(outOfSetTypeSize_ + TWO * ALIGN_SIZE, ubBlock) + // sortIndice
                            Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +                   // updateOriginIndices         
                            Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +                    // uniqueIdCount
                            Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) +              // updateSumIdx
                            GetSortTmpSize(outOfSetDtype_, 1, false);
    } else {
        indicesSize = Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +    // indices
                            Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +          // outofset
                            Ops::Base::CeilAlign(indiceCastDtypeSize_, ubBlock) +      // cast_outofset
                            Ops::Base::CeilAlign(indiceCastDtypeSize_ + TWO * ALIGN_SIZE, ubBlock) + // sortIndice
                            Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +               // updateOriginIndices
                            Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +         // uniqueIdCount
                            Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +          // updateSumIdx
                            GetSortTmpSize(indiceCastDtype_, 1, false);
    }
    int64_t oneBlockSize = indicesSize + (varTypeSize_ + varTypeSize_) * eachCoreAfterAxisCount_; // (update + updateSum)
    indicesFactor_ = ubSize / oneBlockSize;

    int64_t occupy = 0;
    if (!indiceCastMode_) {
        occupy = Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(outOfSetTypeSize_ + TWO * ALIGN_SIZE, ubBlock) + 
                        Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + 
                        Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +
                        Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) + 
                        Ops::Base::CeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) + 
                        Ops::Base::CeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) + 
                        GetSortTmpSize(outOfSetDtype_, 1, false);
    } else {
        occupy = Ops::Base::CeilAlign(rankSize_ * indicesTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(indiceCastDtypeSize_, ubBlock) + 
                        Ops::Base::CeilAlign(indiceCastDtypeSize_ + TWO * ALIGN_SIZE, ubBlock) + 
                        Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + 
                        Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +
                        Ops::Base::CeilAlign(indicesTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) + 
                        Ops::Base::CeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) + 
                        GetSortTmpSize(indiceCastDtype_, 1, false);
    }

    if (occupy > ubSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesSize);
        /* indicesBuf_ + outOfstBuf_ */
        indicesFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesSize;
        int64_t updatesSize = Ops::Base::CeilAlign((varTypeSize_) * indicesFactor_, ubBlock) + Ops::Base::CeilAlign((varTypeSize_) * indicesFactor_, ubBlock);
        afterAxisFactor_ = (ubSize - indicesFactor_ * indicesSize) / updatesSize;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum) / updateDtype_;
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
        indicesFactor_ = ubSize / (afterAxisFactor_ * (varTypeSize_ + varTypeSize_) + indicesSize);

        int64_t restSize = static_cast<int64_t>(-1);
        int64_t indicesDtypeCastSize = indiceCastMode_ ? indiceCastDtypeSize_ : 0;
        int64_t indicesRealTypeSize = indiceCastMode_ ? indiceCastDtypeSize_ : outOfSetTypeSize_;
        ge::DataType sortTmpType = indiceCastMode_ ?  indiceCastDtype_ : outOfSetDtype_;

        while (restSize <= 0) {
            restSize = ubSize - (Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                                    Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) +
                                    Ops::Base::CeilAlign(indicesFactor_ * indicesDtypeCastSize, ubBlock) +
                                    Ops::Base::CeilAlign(indicesFactor_ * (indicesRealTypeSize + TWO * ALIGN_SIZE), ubBlock) + 
                                    Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) + 
                                    Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES), ubBlock) +
                                    Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) + 
                                    indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_) * eachCoreAfterAxisCount_, ubBlock) +
                                    indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_) * eachCoreAfterAxisCount_, ubBlock) + 
                                    GetSortTmpSize(sortTmpType, indicesFactor_, false));
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
            --indicesFactor_;
        }
        // indicesFactor_ = Ops::Base::CeilAlign(indicesFactor_ * indicesTypeSize_, ALIGN_SIZE) / indicesTypeSize_;
    }

    /* 每个核分的indices相同 */
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, indicesFactor_);
    indiceTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * indicesFactor_;

    /* 主核循环次数 */
    updateLoopSize_ = Ops::Base::CeilDiv(eachCoreAfterAxisCount_, afterAxisFactor_);
    /* 主核尾loop处理afterAxis大小 */
    updateTailNum_ = eachCoreAfterAxisCount_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 尾核循环次数 */
    tailUpdateLoopSize_ = Ops::Base::CeilDiv(tailCoreAfterAxisCount_, afterAxisFactor_);
    /* 尾核尾loop处理afterAxis大小 */
    tailUpdateTailNum_ = tailCoreAfterAxisCount_ - (tailUpdateLoopSize_ - 1) * afterAxisFactor_;
    isSplitAfterAxis_ = 1;
}

void ScatterNdCommonSimdSortTiling::DoOpTilingSimdSplitIndices() // TODO: 没开double buffer 为啥 half ub ?
{
    int64_t indicesDtypeCastSize = indiceCastMode_ ? indiceCastDtypeSize_ : 0;
    int64_t indicesRealTypeSize = indiceCastMode_ ? indiceCastDtypeSize_ : outOfSetTypeSize_;
    ge::DataType sortTmpType = indiceCastMode_ ?  indiceCastDtype_ : outOfSetDtype_;

    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    // int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / DB_BUFFER);
    int64_t halfUbSize = static_cast<int64_t>(ubSize_ - RESERVE_SIZE);
    halfUbSize = halfUbSize - MIN_HANDLE_SIZE * FP32_BYTES;//maxScore

    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneIndexSize = static_cast<int64_t>(rankSize_) * indicesTypeSize_;
    
    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates, strideBuf放在RESERVE_SIZE中:
      * indicesFactor_: indiecesQue + outOfsetBuf + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
      *                 (uniqueIdCntQue_ + 1) + updateSumIdxQue_,
      * indicesFactor_ * eachCoreAfterAxisCount_: updatesQue_ + updateSumQue_
    */
    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t indicesAlignSize = Ops::Base::CeilAlign(oneIndexSize, ubBlock) +
                            Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock) +
                            Ops::Base::CeilAlign(indicesDtypeCastSize, ubBlock) +
                            Ops::Base::CeilAlign(indicesRealTypeSize + TWO * ALIGN_SIZE, ubBlock) + 
                            Ops::Base::CeilAlign(INT32_BYTES, ubBlock) + 
                            // Ops::Base::CeilAlign(INT32_BYTES * TWO, ubBlock) +
                            Ops::Base::CeilAlign(INT32_BYTES, ubBlock) +
                            Ops::Base::CeilAlign(outOfSetTypeSize_, ubBlock);

    int64_t updateAlignSize = Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) + 
                            Ops::Base::CeilAlign(varTypeSize_ * afterAxis_, ubBlock) + 
                            GetSortTmpSize(sortTmpType, 1, false);
    if (indicesAlignSize + updateAlignSize > halfUbSize) { // 优先保证indices处理的行数，实际 afterAxisFactor_ 可能远小于afterAxis_
        int64_t indicesSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesAlignSize);
        /* indicesBuf_ + outOfstBuf_ */
        indicesFactor_ = Ops::Base::CeilAlign(indicesSize, ALIGN_SIZE) / indicesAlignSize;
        afterAxisFactor_ = (halfUbSize - indicesFactor_ * indicesAlignSize) / indicesFactor_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else { // 全载 afterAxis_,逐步调整 indicesFactor_ 以满足UB约束
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        indicesFactor_ = halfUbSize / (updateAlignSize + indicesAlignSize);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            int64_t occupy = Ops::Base::CeilAlign(indicesFactor_ * rankSize_ * indicesTypeSize_, ubBlock) +
                            Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) +
                            Ops::Base::CeilAlign(indicesFactor_ * indicesDtypeCastSize, ubBlock) + 
                            Ops::Base::CeilAlign(indicesFactor_ * (indicesRealTypeSize + TWO * ALIGN_SIZE), ubBlock) + 
                            Ops::Base::CeilAlign(indicesFactor_ * INT32_BYTES, ubBlock) + 
                            Ops::Base::CeilAlign(indicesFactor_ * (INT32_BYTES), ubBlock) +
                            Ops::Base::CeilAlign(indicesFactor_ * outOfSetTypeSize_, ubBlock) + 
                            indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_) * afterAxisFactor_, ubBlock) +  // update
                            indicesFactor_ * Ops::Base::CeilAlign((varTypeSize_) * afterAxisFactor_, ubBlock) +  // updateSum
                            GetSortTmpSize(sortTmpType, indicesFactor_, false);
            restSize = halfUbSize - occupy;
            if (indicesFactor_ > indicesAxis_) {
                indicesFactor_ = indicesAxis_;
                break;
            }
            --indicesFactor_;
        }
    }
    /* 每个核分的update相同 */
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;
}

void ScatterNdCommonSimdSortTiling::DoOpTilingForSimdSort()
{
    GetCastType();
    // 列大于4k单行搬入搬出
    if (afterAxis_ * varTypeSize_ > SIMD_NONDETERMINSTIC_MIN_HANDLE_SIZE &&
       (indicesAxis_ > (totalCoreNum_ * SPLIT_THRESHOLD))) {
        DoOpTilingSplitIndicesSingleCol();
        return;
    }
    /* 优先分after */
    int64_t splitThresh = totalCoreNum_ * MIN_HANDLE_SIZE / varTypeSize_; 
    if ((afterAxis_ > splitThresh) || (indicesAxis_ < (totalCoreNum_ / TWO))) {
        DoOpTilingSplitAfter(); // 如果尾轴大于 核数*128B 或者 索引(g)小于一半核数 则切尾轴
        return;
    }
    DoOpTilingSimdSplitIndices(); // 否则切索引
    return;
}


ge::graphStatus ScatterNdCommonSimdSortTiling::DoOpTiling()
{
    DoOpTilingForSimdSort();
    SetStride();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdCommonSimdSortTiling::PostTiling()
{
    OP_LOGD("ScatterNdCommonSimdSortTiling::PostTiling begin");
    context_->SetBlockDim(usedCoreNumBefore_);

    return ge::GRAPH_SUCCESS;
}


std::string ScatterNdCommonSimdSortTiling::TilingDataToString()
{
    ScatterNdCommon::ScatterNdCommonSimdSortTilingData* tilingData =
        context_->GetTilingData<ScatterNdCommon::ScatterNdCommonSimdSortTilingData>();
    std::string str = " eachCoreAfterAxisCount:" + std::to_string(tilingData->eachCoreAfterAxisCount);
    str += " indexRankSize:" + std::to_string(tilingData->indexRankSize);
    str += " eachCoreIndexCount:" + std::to_string(tilingData->eachCoreIndexCount);
    str += " tailCoreIndexCount:" + std::to_string(tilingData->tailCoreIndexCount);
    str += " indicesFactor:" + std::to_string(tilingData->indicesFactor);
    str += " indiceTailNum:" + std::to_string(tilingData->indiceTailNum);
    str += " indicesLoopSize:" + std::to_string(tilingData->indicesLoopSize);
    str += " afterAxis:" + std::to_string(tilingData->afterAxis);
    str += " afterAxisFactor:" + std::to_string(tilingData->afterAxisFactor);
    str += " usedCoreNumBefore:" + std::to_string(tilingData->usedCoreNumBefore);
    str += " updateLoopSize:" + std::to_string(tilingData->updateLoopSize);
    str += " tailUpdateLoopSize:" + std::to_string(tilingData->tailUpdateLoopSize);
    str += " tailUpdateTailNum:" + std::to_string(tilingData->tailUpdateTailNum);
    str += " updateTailNum:" + std::to_string(tilingData->updateTailNum);
    str += " isSplitAfterAxis:" + std::to_string(tilingData->isSplitAfterAxis);
    str += " singleCol:" + std::to_string(tilingData->singleCol);
    for (int32_t i = 0; i < OPTILING_MAX_RANK_COUNT; i++) {
        str += " strideList[i]:" + std::to_string(tilingData->strideList[i]);
    }
    return str;
}

void ScatterNdCommonSimdSortTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "Tiling info is: %s", TilingDataToString().c_str());
}

} // namespace optiling