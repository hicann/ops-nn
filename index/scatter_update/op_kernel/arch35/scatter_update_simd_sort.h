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
 * \file scatter_update_simt.h
 * \brief scatter_update
 */
#ifndef SCATTER_UPDATE_SIMD_SORT_H
#define SCATTER_UPDATE_SIMD_SORT_H

#include "scatter_update_common.h"
#include "scatter_update_simd_common.h"
#include "scatter_update_struct.h"
#include "../inc/platform.h"

namespace ScatterUpdate {
using namespace AscendC;

constexpr uint64_t INDICES_ONE_SORT = 1;

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType>
class ScatterUpdateSimdSort : public ScatterUpdateSimdCommon<T, U, updatesIsScalar>  {
public:
    __aicore__ inline ScatterUpdateSimdSort(const ScatterUpdateTilingData& tilingData, TPipe& pipe) :
        ScatterUpdateSimdCommon<T, U, updatesIsScalar> (tilingData, pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessNotSplitColSort();
    __aicore__ inline void ProcessSplitColSort();
    __aicore__ inline void ProcessNotSplitUpdatesWithSort(uint32_t rowLen, uint32_t outLen);
    __aicore__ inline void ProcessSplitUpdatesWithSort(uint32_t rowLen);

    __aicore__ inline void SyncVtoS() {
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdSToV);
        WaitFlag<HardEvent::V_S>(eventIdSToV);
    }

private:
    TBuf<QuePosition::VECCALC> sortIndicesBuf_;
    TBuf<QuePosition::VECCALC> updatesOriginIdexBuf_;
    TBuf<QuePosition::VECCALC> uniqueIdCountBuf_;
    TBuf<QuePosition::VECCALC> castIndicesQue_;
};

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimdSort<T, U, CAST_T, updatesIsScalar, castType>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    this->InitBase(var, indices, updates);
    this->pipe_.InitBuffer(updatesOriginIdexBuf_, this->tilingData_.indicesUbFactor * sizeof(uint32_t));
    this->pipe_.InitBuffer(uniqueIdCountBuf_, ops::CeilAlign((this->tilingData_.indicesUbFactor + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
    if constexpr (castType == CAST_NOT_CAST) {
        this->pipe_.InitBuffer(sortIndicesBuf_, (this->tilingData_.indicesUbFactor * sizeof(U) + SORT_PAD_NUM * UB_AGLIN_VALUE));
 	} else {
        this->pipe_.InitBuffer(sortIndicesBuf_, (this->tilingData_.indicesUbFactor * sizeof(CAST_T) + SORT_PAD_NUM * UB_AGLIN_VALUE));
        this->pipe_.InitBuffer(castIndicesQue_, (this->tilingData_.indicesUbFactor * sizeof(CAST_T)));
 	}
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimdSort<T, U, CAST_T, updatesIsScalar, castType>::Process()
{
    if (GetBlockIdx() >= this->tilingData_.usedCoreNum) {
        return;
    }
    if (!this->needSplitCol_) {
        ProcessNotSplitColSort();
    } else {
        ProcessSplitColSort();
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimdSort<T, U, CAST_T, updatesIsScalar, castType>::ProcessNotSplitColSort()
{
    int64_t indicesOffset = 0;
    int64_t updatesOffset = 0;
    int64_t indicesStride = this->tilingData_.processRowPerUb;
    int64_t updatesStride = this->tilingData_.processRowPerUb * this->tilingData_.colTotal;
    uint32_t rowByUb = 0;
    uint32_t uniqueIdNum = 0;
    for (int64_t i = 0; i < this->tilingData_.rowLoopByUb; i++) {
        rowByUb = i == this->tilingData_.rowLoopByUb - 1 ? this->tilingData_.processRowTail : this->tilingData_.processRowPerUb;
        this->CopyInIndices(indicesOffset, rowByUb);
        this->CopyInUpdates(updatesOffset, rowByUb, this->tilingData_.processColNum);
        LocalTensor<U> indicesLocal = this->indicesInQueue_.template DeQue<U>();
        LocalTensor<CAST_T> sortIndicesLocal = sortIndicesBuf_.Get<CAST_T>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexBuf_.Get<uint32_t>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountBuf_.Get<int32_t>();
        if constexpr (castType == CAST_NOT_CAST) {
            uniqueIdNum = SortAndComputeUniqueIdx<U>(rowByUb, indicesLocal, sortIndicesLocal, uniqueIdCountLocal, updatesOriginIdexLocal);
        } else {
 	        LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
 	        IndicesSortCast<U, CAST_T, castType>(indicesLocal, indicesCastLocal, uniqueIdCountLocal, rowByUb);
 	        uniqueIdNum = SortAndComputeUniqueIdx<CAST_T>(
 	            rowByUb, indicesCastLocal, sortIndicesLocal, uniqueIdCountLocal, updatesOriginIdexLocal);
 	    }
        ProcessNotSplitUpdatesWithSort(uniqueIdNum, this->tilingData_.processColNum);
        indicesOffset = indicesOffset + indicesStride;
        updatesOffset = updatesOffset + updatesStride;
        this->indicesInQueue_.FreeTensor(indicesLocal);
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimdSort<T, U, CAST_T, updatesIsScalar, castType>::ProcessSplitColSort()
{
    int64_t indicesOffset = 0;
    int64_t updatesOffset = 0;
    int64_t indicesStride = this->tilingData_.processRowPerUb;
    uint32_t colByUb = 0;
    uint32_t rowByUb = 0;
    uint32_t uniqueIdNum = 0;
    for (int64_t i = 0; i < this->tilingData_.rowLoopByUb; i++) {
        rowByUb = i == this->tilingData_.rowLoopByUb - 1 ? this->tilingData_.processRowTail : this->tilingData_.processRowPerUb;
        this->CopyInIndices(indicesOffset, rowByUb);
        LocalTensor<U> indicesLocal = this->indicesInQueue_.template DeQue<U>();
        LocalTensor<CAST_T> sortIndicesLocal = sortIndicesBuf_.Get<CAST_T>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexBuf_.Get<uint32_t>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountBuf_.Get<int32_t>();
        if constexpr (castType == CAST_NOT_CAST) {
            uniqueIdNum = SortAndComputeUniqueIdx<U>(rowByUb, indicesLocal, sortIndicesLocal, uniqueIdCountLocal, updatesOriginIdexLocal);
        } else {
 	        LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
 	        IndicesSortCast<U, CAST_T, castType>(indicesLocal, indicesCastLocal, uniqueIdCountLocal, rowByUb);
 	        uniqueIdNum = SortAndComputeUniqueIdx<CAST_T>(
 	            rowByUb, indicesCastLocal, sortIndicesLocal, uniqueIdCountLocal, updatesOriginIdexLocal);
 	    }
        ProcessSplitUpdatesWithSort(uniqueIdNum);
        this->indicesInQueue_.FreeTensor(indicesLocal);
        indicesOffset = indicesOffset + indicesStride;
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimdSort<T, U, CAST_T, updatesIsScalar, castType>::ProcessNotSplitUpdatesWithSort(uint32_t rowLen, uint32_t outLen)
{
    LocalTensor<T> updatesLocal = this->updateInQueue_.template DeQue<T>();
    SyncVtoS();
    LocalTensor<U> sortIndicesLocal = sortIndicesBuf_.Get<U>();
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(U);
    LocalTensor<U> shiftSortLocal = sortIndicesLocal[shiftOffset];
    LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexBuf_.Get<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountBuf_.Get<int32_t>();

    int64_t unRepeatIndex = 0;
    uint32_t updatesIndexOffset = (outLen + this->BLOCK_NUM - 1) / this->BLOCK_NUM * this->BLOCK_NUM;
    for (int64_t i = 0; i < rowLen; i++) {
        unRepeatIndex = uniqueIdCountLocal(i);
        U varIndex = shiftSortLocal(unRepeatIndex);
        if (static_cast<int64_t>(varIndex) < 0 || static_cast<int64_t>(varIndex) >= this->tilingData_.varShape[0]) {
            continue;
        }
        int64_t varOffset = static_cast<int64_t>(varIndex * this->tilingData_.varStride + this->tilingData_.colBase * this->colOffset_);
        int64_t updatesIndex = updatesOriginIdexLocal(unRepeatIndex) * updatesIndexOffset;
        this->CopyOutUpdates(varOffset, outLen, updatesLocal[updatesIndex]);
    }

    this->updateInQueue_.FreeTensor(updatesLocal);
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimdSort<T, U, CAST_T, updatesIsScalar, castType>::ProcessSplitUpdatesWithSort(uint32_t rowLen)
{
    SyncVtoS();
    LocalTensor<U> sortIndicesLocal = sortIndicesBuf_.Get<U>();
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(U);
    LocalTensor<U> shiftSortLocal = sortIndicesLocal[shiftOffset];
    LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexBuf_.Get<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountBuf_.Get<int32_t>();
    int64_t unRepeatIndex = 0;
    uint32_t colByUb = 0;
    for (int64_t i = 0; i < rowLen; i++) {
        unRepeatIndex = uniqueIdCountLocal(i);
        U varIndex = shiftSortLocal(unRepeatIndex);
        if (static_cast<int64_t>(varIndex) < 0 || static_cast<int64_t>(varIndex) >= this->tilingData_.varShape[0]) {
            continue;
        }
        int64_t updatesOffset = updatesOriginIdexLocal(unRepeatIndex) * this->tilingData_.colTotal;
        // FOR循环搬运updates
        for (int64_t j = 0; j < this->tilingData_.colLoopByUb; j++) {
            colByUb = j == this->tilingData_.colLoopByUb - 1 ? this->tilingData_.processColTail : this->tilingData_.processColPerUb;
            this->CopyInUpdates(updatesOffset, INDICES_ONE_SORT, colByUb);
            LocalTensor<T> updatesLocal = this->updateInQueue_.template DeQue<T>();
            int64_t varOffset = static_cast<int64_t>(varIndex * this->tilingData_.varStride + this->tilingData_.colBase * this->colOffset_ + j * this->tilingData_.processColPerUb);
            this->CopyOutUpdates(varOffset, colByUb, updatesLocal);
            this->updateInQueue_.FreeTensor(updatesLocal);
            updatesOffset = updatesOffset + colByUb;
        }
    }
}
}
#endif  // SCATTER_UPDATE_SIMD_SORT_H