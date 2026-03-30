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
#ifndef SCATTER_UPDATE_SIMD_IMPL_H
#define SCATTER_UPDATE_SIMD_IMPL_H

#include "scatter_update_common.h"
#include "scatter_update_simd_common.h"
#include "scatter_update_struct.h"
#include "../inc/platform.h"

namespace ScatterUpdate {
using namespace AscendC;

constexpr uint64_t INDICES_ONE = 1;

template<typename T, typename U, bool updatesIsScalar>
class ScatterUpdateSIMDImpl : public ScatterUpdateSimdCommon<T, U, updatesIsScalar> {
public:
    __aicore__ inline ScatterUpdateSIMDImpl(const ScatterUpdateTilingData& tilingData, TPipe& pipe) :
        ScatterUpdateSimdCommon<T, U, updatesIsScalar> (tilingData, pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessNotSplitCol();
    __aicore__ inline void ProcessSplitCol();
    __aicore__ inline void ProcessUpdates(LocalTensor<U> indicesLocal, uint32_t rowLen, uint32_t outLen, uint64_t oneRowOffSet);
    __aicore__ inline void ProcessUpdatesSplit(uint32_t outLen, uint64_t oneRowOffSet, U indicesValue);

    __aicore__ inline void SyncMte2toS() {
        auto sWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(sWaitMTEEventID);
        WaitFlag<HardEvent::MTE2_S>(sWaitMTEEventID);
    }
};

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSIMDImpl<T, U, updatesIsScalar>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    this->InitBase(var, indices, updates);
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSIMDImpl<T, U, updatesIsScalar>::Process()
{
    if (GetBlockIdx() >= this->tilingData_.usedCoreNum) {
        return;
    }
    if (!this->needSplitCol_) {
        ProcessNotSplitCol();
    } else {
        ProcessSplitCol();
    }
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSIMDImpl<T, U, updatesIsScalar>::ProcessNotSplitCol()
{
    int64_t indicesOffset = 0;
    int64_t updatesOffset = 0;
    int64_t indicesStride = this->tilingData_.processRowPerUb;
    int64_t updatesStride = this->tilingData_.processRowPerUb * this->tilingData_.colTotal;
    uint32_t rowByUb = 0;
    for (int64_t i = 0; i < this->tilingData_.rowLoopByUb; i++) {
        rowByUb = i == this->tilingData_.rowLoopByUb - 1 ? this->tilingData_.processRowTail : this->tilingData_.processRowPerUb;
        this->CopyInIndices(indicesOffset, rowByUb);
        LocalTensor<U> indicesLocal = this->indicesInQueue_.template DeQue<U>();
        this->CopyInUpdates(updatesOffset, rowByUb, this->tilingData_.processColNum);
        ProcessUpdates(indicesLocal, rowByUb, this->tilingData_.processColNum, 0);
        indicesOffset = indicesOffset + indicesStride;
        updatesOffset = updatesOffset + updatesStride;
        this->indicesInQueue_.FreeTensor(indicesLocal);
    }
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSIMDImpl<T, U, updatesIsScalar>::ProcessUpdates(LocalTensor<U> indicesLocal, uint32_t rowLen, uint32_t outLen, uint64_t oneRowOffSet)
{
    LocalTensor<T> updatesLocal = this->updateInQueue_.template DeQue<T>();
    uint32_t updatesOffset = (outLen + this->BLOCK_NUM - 1) / this->BLOCK_NUM * this->BLOCK_NUM;
    SyncMte2toS();
    for (int64_t i = 0; i < rowLen; ++i) {
        U indicesValue = indicesLocal.GetValue(i);
        if (static_cast<int64_t>(indicesValue) < 0 || static_cast<int64_t>(indicesValue) >= this->tilingData_.varShape[0]) {
            continue;
        }
        int64_t indicesOffset = static_cast<int64_t>(indicesValue * this->tilingData_.varStride + this->tilingData_.colBase * this->colOffset_ + oneRowOffSet);
        this->CopyOutUpdates(indicesOffset, outLen, updatesLocal[i * updatesOffset]);
    }
    this->updateInQueue_.FreeTensor(updatesLocal);
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSIMDImpl<T, U, updatesIsScalar>::ProcessSplitCol()
{
    int64_t indicesOffset = 0;
    int64_t updatesOffset = 0;
    int64_t indicesStride = this->tilingData_.processRowPerUb;
    uint32_t colByUb = 0;
    uint32_t rowByUb = 0;
    for (int64_t i = 0; i < this->tilingData_.rowLoopByUb; i++) {
        rowByUb = i == this->tilingData_.rowLoopByUb - 1 ? this->tilingData_.processRowTail : this->tilingData_.processRowPerUb;
        this->CopyInIndices(indicesOffset, rowByUb);
        LocalTensor<U> indicesLocal = this->indicesInQueue_.template DeQue<U>();
        SyncMte2toS();
        for (int64_t k = 0; k < rowByUb; k++) {
            U indicesValue = indicesLocal.GetValue(k);
            if (static_cast<int64_t>(indicesValue) < 0 || static_cast<int64_t>(indicesValue) >= this->tilingData_.varShape[0]) {
                continue;
            }
            for (int64_t j = 0; j < this->tilingData_.colLoopByUb; j++) {
                colByUb = j == this->tilingData_.colLoopByUb - 1 ? this->tilingData_.processColTail : this->tilingData_.processColPerUb;
                this->CopyInUpdates(updatesOffset, INDICES_ONE, colByUb);
                ProcessUpdatesSplit(colByUb, j * this->tilingData_.processColPerUb, indicesValue);
                updatesOffset = updatesOffset + colByUb;
            }
            updatesOffset = INDICES_ONE * this->tilingData_.colTotal * (k + 1);
        }
        this->indicesInQueue_.FreeTensor(indicesLocal);
        indicesOffset = indicesOffset + indicesStride;
    }
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSIMDImpl<T, U, updatesIsScalar>::ProcessUpdatesSplit(uint32_t outLen, uint64_t oneRowOffSet, U indicesValue)
{
    LocalTensor<T> updatesLocal = this->updateInQueue_.template DeQue<T>();
    SyncMte2toS();

    int64_t indicesOffset = static_cast<int64_t>(indicesValue * this->tilingData_.varStride + this->tilingData_.colBase * this->colOffset_ + oneRowOffSet);
    this->CopyOutUpdates(indicesOffset, outLen, updatesLocal);
    this->updateInQueue_.FreeTensor(updatesLocal);
}
}
#endif  // SCATTER_UPDATE_SIMD_IMPL_H