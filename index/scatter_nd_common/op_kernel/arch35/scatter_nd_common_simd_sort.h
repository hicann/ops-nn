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
 * \file scatter_nd_common_simd_sort.h
 * \brief scatter_nd_common_simd_sort
 */

#ifndef SCATTER_ND_COMMON_SIMD_SORT_H
#define SCATTER_ND_COMMON_SIMD_SORT_H

#include "scatter_nd_common_base.h"
#include "../inc/kernel_utils.h"

namespace ScatterNdCommon {
using namespace AscendC;

template <typename T, typename U, typename CAST_T, typename OFFSET_T, uint32_t castType, uint8_t Mode>
class ScatterNdCommonSimdSort : public ScatterNdCommonBase<T, U, CAST_T, OFFSET_T, castType, Mode> {
public:
    __aicore__ inline ScatterNdCommonSimdSort(const ScatterNdCommonSimdSortTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y);
    __aicore__ inline void ProcessSplitAfter();
    __aicore__ inline void ProcessSplitIndices();
    __aicore__ inline void ProcessSplitIndicesSingleCol();
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;

    TPipe& pipe_;
    const ScatterNdCommonSimdSortTilingData& tilingData_;

    int64_t curCoreIndexCount_{0};
    uint64_t strideList[MAX_RANK_COUNT];
};

template <typename T, typename U, typename CAST_T, typename OFFSET_T, uint32_t castType, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimdSort<T, U, CAST_T, OFFSET_T, castType, Mode>::Init(GM_ADDR var,
                                                                                             GM_ADDR indices,
                                                                                             GM_ADDR updates, GM_ADDR y)
{
    this->eachCoreAfterAxisCount_ = tilingData_.eachCoreAfterAxisCount;
    this->indexRankSize_ = tilingData_.indexRankSize;
    this->eachCoreIndexCount_ = tilingData_.eachCoreIndexCount;
    this->indicesFactor_ = tilingData_.indicesFactor;
    this->afterAxis_ = tilingData_.afterAxis;
    this->afterAxisFactor_ = tilingData_.afterAxisFactor;
    this->InitBaseBuffer(pipe_, tilingData_.indicesFactor, indices, updates, y, tilingData_.singleCol);

    curCoreIndexCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCoreIndexCount :
                                                                                 tilingData_.tailCoreIndexCount);
}

template <typename T, typename U, typename CAST_T, typename OFFSET_T, uint32_t castType, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimdSort<T, U, CAST_T, OFFSET_T, castType, Mode>::ProcessSplitAfter()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = tilingData_.indiceTailNum;
    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize :
                                                                                tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateTailNum :
                                                                                    tilingData_.updateTailNum;
    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        this->CopyIndiceInSplitAfter(rowIdx, rowDataLen);
        LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
        if (this->maxScore_ > SORT_HIST_THRESHOLD) {
            this->SortIndices(outOfstLocal, rowDataLen);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitAfter(rowIdx, colIdx, rowDataLen, colDataLen);
                event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                this->ComputeUpdateSum(rowDataLen, colDataLen);
                this->ComputeOutSplitAfter(colIdx, rowDataLen, colDataLen);
            }
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            LocalTensor<OFFSET_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<OFFSET_T>();
            this->updatesOriginIdexQue_.template FreeTensor(updatesOriginIdexLocal);
            this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
            this->uniqueIdCountQue_.template FreeTensor(uniqueIdCountLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitAfter(rowIdx, colIdx, rowDataLen, colDataLen);
                LocalTensor<T> updatesLocal = this->dataQueue_.template DeQue<T>();
                event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                this->CopyOutSplitAfter(outOfstLocal, updatesLocal, rowDataLen, colDataLen, colIdx);
                this->dataQueue_.template FreeTensor(updatesLocal);
            }
        }
    }
}

template <typename T, typename U, typename CAST_T, typename OFFSET_T, uint32_t castType, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimdSort<T, U, CAST_T, OFFSET_T, castType, Mode>::ProcessSplitIndices()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;
    int64_t rowLoopNum = ops::CeilDiv(curCoreIndexCount_, tilingData_.indicesFactor);
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.indicesFactor * (rowLoopNum - 1);

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        this->CopyIndiceInSplitIndices(rowIdx, rowDataLen);
        LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
        if (this->maxScore_ > SORT_HIST_THRESHOLD) {
            this->SortIndices(outOfstLocal, rowDataLen);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
                // 累加相同索引的updates
                event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                this->ComputeUpdateSum(rowDataLen, colDataLen);
                this->ComputeOutSplitIndices(colIdx, rowDataLen, colDataLen);
            }

            //本次行数处理完成,释放资源
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            LocalTensor<OFFSET_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<OFFSET_T>();
            this->uniqueIdCountQue_.template FreeTensor(uniqueIdCountLocal);
            this->updatesOriginIdexQue_.template FreeTensor(updatesOriginIdexLocal);
            this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
                LocalTensor<T> updatesLocal = this->dataQueue_.template DeQue<T>();
                event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                this->CopyOutSplitIndices(outOfstLocal, updatesLocal, rowDataLen, colDataLen, colIdx);
                this->dataQueue_.template FreeTensor(updatesLocal);
            }
        }
    }
}

template <typename T, typename U, typename CAST_T, typename OFFSET_T, uint32_t castType, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimdSort<T, U, CAST_T, OFFSET_T, castType, Mode>::ProcessSplitIndicesSingleCol()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;
    int64_t rowLoopNum = ops::CeilDiv(curCoreIndexCount_, tilingData_.indicesFactor);
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.indicesFactor * (rowLoopNum - 1);

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        this->CopyIndiceInSplitIndices(rowIdx, rowDataLen);
        LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
        if (this->maxScore_ > SORT_HIST_THRESHOLD) {
            this->SortIndices(outOfstLocal, rowDataLen);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->ComputeSortInOutSingleCol(rowIdx, colIdx, rowDataLen, colLen);
            }
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            LocalTensor<OFFSET_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<OFFSET_T>();
            this->uniqueIdCountQue_.template FreeTensor(uniqueIdCountLocal);
            this->updatesOriginIdexQue_.template FreeTensor(updatesOriginIdexLocal);
            this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyInAndOutSingleCol(outOfstLocal, rowIdx, colIdx, rowDataLen, colDataLen);
            }
        }
    }
}

template <typename T, typename U, typename CAST_T, typename OFFSET_T, uint32_t castType, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimdSort<T, U, CAST_T, OFFSET_T, castType, Mode>::Process()
{
    LocalTensor<OFFSET_T> strideLocal = this->strideBuf_.template Get<OFFSET_T>();
    LocalTensor<U> outputShapeLocal = this->outputShapeBuf_.template Get<U>();
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideLocal(i) = tilingData_.strideList[i];
    }
    for (int32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        outputShapeLocal(i) = tilingData_.outPutShape[i];
    }

    if (tilingData_.isSplitAfterAxis == 1) {
        ProcessSplitAfter();
    } else {
        if (tilingData_.singleCol) {
            ProcessSplitIndicesSingleCol();
        } else {
            ProcessSplitIndices();
        }
    }
}
} // namespace ScatterNdCommon
#endif // SCATTER_ND_COMMON_SIMD_SORT_H