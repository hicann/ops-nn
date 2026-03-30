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
 * \file scatter_update_common.h
 * \brief scatter_update
 */
#ifndef ASCENDC_SCATTER_UPDATE_SIMD_COMMON_H_
#define ASCENDC_SCATTER_UPDATE_SIMD_COMMON_H_

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace ScatterUpdate {
using namespace AscendC;

constexpr uint64_t UB_AGLIN_VALUE_32 = 32;
constexpr int64_t DB_BUFFER = 2;

template<typename T, typename U, bool updatesIsScalar>
class ScatterUpdateSimdCommon {
public:
    __aicore__ inline ScatterUpdateSimdCommon(const ScatterUpdateTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void InitBase(GM_ADDR var, GM_ADDR indices, GM_ADDR updates);
    __aicore__ inline void CopyInIndices(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyInUpdates(int64_t offset, uint32_t rowLen, uint32_t colLen);
    __aicore__ inline void CopyOutUpdates(int64_t offset, int64_t dataLen, LocalTensor<T> updatesLocal);

    __aicore__ inline void SyncStoV() {
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
    }

protected:
    int32_t blockIdx_ {0};
    int64_t updateStart_ {0};
    // mode
    static constexpr int32_t BLOCK_NUM = UB_AGLIN_VALUE_32 / sizeof(T);
    bool needSplitCol_ {true};
    int64_t colOffset_ = 0;

    AscendC::GlobalTensor<T> varRefGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> updateInQueue_;
    TPipe& pipe_;
    const ScatterUpdateTilingData& tilingData_;
    TQue<QuePosition::VECIN, DB_BUFFER> indicesInQueue_;
};

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSimdCommon<T, U, updatesIsScalar>::InitBase(GM_ADDR var, GM_ADDR indices, GM_ADDR updates) {
    blockIdx_ = GetBlockIdx();
    bool isNorm = (blockIdx_ + 1) % tilingData_.colTileNum != 0;
    needSplitCol_ = isNorm ? tilingData_.normNeedSplitRow : tilingData_.tailNeedSplitRow;
    colOffset_ = blockIdx_ % tilingData_.colTileNum;
    int64_t rowOffset = blockIdx_ / tilingData_.colTileNum;
    updateStart_ = tilingData_.rowBase * rowOffset * tilingData_.colTotal + tilingData_.colBase * colOffset_;

    pipe_.InitBuffer(indicesInQueue_, DB_BUFFER, tilingData_.indicesUbFactor * sizeof(U));
    pipe_.InitBuffer(updateInQueue_, DB_BUFFER,  tilingData_.updateUbSize);

    varRefGm_.SetGlobalBuffer((__gm__ T *)(var));
    indicesGm_.SetGlobalBuffer((__gm__ U *)(indices) + rowOffset * tilingData_.rowBase);
    if constexpr (updatesIsScalar) {
        updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    } else {
        updatesGm_.SetGlobalBuffer((__gm__ T *)(updates) + updateStart_);
    }
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSimdCommon<T, U, updatesIsScalar>::CopyInIndices(int64_t offset, int64_t dataLen)
{
    LocalTensor<U> indicesLocal = indicesInQueue_.AllocTensor<U>();
    DataCopyExtParams inParams = { 1, static_cast<uint32_t>(dataLen * sizeof(U)), 0, 0, 0 };
    DataCopyPadExtParams<U> padParams = { false, 0, 0, 0 };
    DataCopyPad(indicesLocal, indicesGm_[offset], inParams, padParams);
    indicesInQueue_.EnQue(indicesLocal);
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSimdCommon<T, U, updatesIsScalar>::CopyInUpdates(int64_t offset, uint32_t rowLen, uint32_t colLen)
{
    if constexpr (updatesIsScalar) {
        T updatesValue = updatesGm_.GetValue(0);
        // todo:下面这个同步是updatesValue这个标量和vector之间的，因为vector使用的update
        SyncStoV();
        LocalTensor<T> updatesLocal = updateInQueue_.AllocTensor<T>();
        Duplicate(updatesLocal, updatesValue, tilingData_.updateUbSize/sizeof(T));
        updateInQueue_.EnQue(updatesLocal);
    } else {
        LocalTensor<T> updatesInLocal = updateInQueue_.AllocTensor<T>();
        DataCopyExtParams inParams{
        static_cast<uint16_t>(rowLen),
        static_cast<uint32_t>(colLen * sizeof(T)),
        static_cast<uint32_t>((tilingData_.colTotal - colLen) * sizeof(T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
        };
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(updatesInLocal, updatesGm_[offset], inParams, padParams);
        updateInQueue_.EnQue(updatesInLocal);
    }
}

template<typename T, typename U, bool updatesIsScalar>
__aicore__ inline void ScatterUpdateSimdCommon<T, U, updatesIsScalar>::CopyOutUpdates(int64_t offset, int64_t dataLen, LocalTensor<T> updatesLocal)
{
    DataCopyExtParams outParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    DataCopyPad(varRefGm_[offset], updatesLocal, outParams);
}
}  // namespace ScatterUpdate
#endif