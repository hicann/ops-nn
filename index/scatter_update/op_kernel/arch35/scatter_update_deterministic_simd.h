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
 * \file scatter_update_deterministic_simd.h
 * \brief scatter_update
 */
#ifndef SCATTER_UPDATE_DETERMINISTIC_SIMD_H
#define SCATTER_UPDATE_DETERMINISTIC_SIMD_H

#include "scatter_update_common.h"
#include "scatter_update_struct.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"

namespace ScatterUpdate {
using namespace AscendC;

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
class ScatterUpdateDeterministicSimd : public ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType> {
public:
    __aicore__ inline ScatterUpdateDeterministicSimd(const ScatterUpdateTilingData& tilingData, TPipe& pipe) : 
        ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType> (tilingData, pipe) {};

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSplitCol();
    __aicore__ inline void ProcessSplitRow();
    __aicore__ inline void CopyInUpdates(uint64_t updatesGmOffset, uint16_t blockCount, uint32_t blockLen);
    __aicore__ inline void CopyOutUpdates(uint64_t varGmOffset, uint32_t updateOffset, uint32_t updatesCount);

private:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> updatesQueue_;

    uint64_t updatesBlockLoop_{0};
    uint64_t updatesTailLoopSize_{0};
};

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimd<T, U, MASK_T, splitCol, CAST_T, castType>::Init(
            GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    this->InitBase(var, indices, updates);

    if constexpr (splitCol) {
        updatesBlockLoop_= this->tilingData_.updatesNormBlockLoop;
        updatesTailLoopSize_ = this->tilingData_.updatesNormBlockTailLoopSize;
        if (this->blockIdx_ == this->tilingData_.usedCoreNum - 1) {
            updatesBlockLoop_= this->tilingData_.updatesTailBlockLoop;
            updatesTailLoopSize_ = this->tilingData_.updatesTailBlockTailLoopSize;
        }
        this->pipe_.InitBuffer(updatesQueue_, 1, this->tilingData_.updateColUbFactor * sizeof(T));
    } else {
        this->pipe_.InitBuffer(updatesQueue_, 1,
            ops::CeilAlign(this->tilingData_.indicesUbFactor * this->tilingData_.varShape[1] * sizeof(T), UB_AGLIN_VALUE));
        this->InitSetBuffer(var, indices, updates, workspace);
    }
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimd<T, U, MASK_T, splitCol, CAST_T, castType>::CopyInUpdates(
                     uint64_t updatesGmOffset, uint16_t blockCount, uint32_t blockLen)
{
    LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();

    DataCopyExtParams updatesCopyParams { blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0 };
    DataCopyPadExtParams<T> updatesPadParams { false, 0, 0, 0 };
    DataCopyPad(updatesLocal, this->updatesGm_[updatesGmOffset], updatesCopyParams, updatesPadParams);
    updatesQueue_.EnQue<T>(updatesLocal);
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimd<T, U, MASK_T, splitCol, CAST_T, castType>::CopyOutUpdates(
                    uint64_t varGmOffset, uint32_t updateOffset, uint32_t updatesCount)
{
    DataCopyExtParams outParams = { 1, static_cast<uint32_t>(updatesCount * sizeof(T)), 0, 0, 0 };
    LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
    DataCopyPad(this->varGm_[varGmOffset], updatesLocal[updateOffset], outParams);
    updatesQueue_.FreeTensor(updatesLocal);
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimd<T, U, MASK_T, splitCol, CAST_T, castType>::ProcessSplitCol()
{
    uint32_t indicesCount = this->tilingData_.indicesUbFactor;
    for (uint64_t idx = 0; idx < this->tilingData_.indicesLoopSize; idx++) {
        if (idx == this->tilingData_.indicesLoopSize - 1) {
            indicesCount = this->tilingData_.indicesTailLoopNum;
        }
        uint64_t indicesGmOffset = idx * this->tilingData_.indicesUbFactor;
        this->CopyInIndices(indicesGmOffset, indicesCount);

        auto eventIDMTE2_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2_S);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2_S);
        LocalTensor<U> indicesLocal = this->indicesQue_.template DeQue<U>();
        for (uint32_t i = 0; i < indicesCount; i++) {
            U indicesValue = indicesLocal.GetValue(i);
            if (indicesValue < 0 || indicesValue >= this->tilingData_.varShape[0]) {
                continue;
            }
            
            uint64_t updatesGmOffset = (indicesGmOffset + i) * this->tilingData_.varShape[1] + 
                                       this->blockIdx_ * this->tilingData_.normBlockColNum;
            uint64_t varGmOffset = indicesValue * this->tilingData_.varStride + 
                                   this->blockIdx_ * this->tilingData_.normBlockColNum;
            uint32_t updatesCount = this->tilingData_.updateColUbFactor;
            for (uint64_t j = 0; j < updatesBlockLoop_; j++) {
                if (j == updatesBlockLoop_ - 1) {
                    updatesCount = updatesTailLoopSize_;
                }
                CopyInUpdates(updatesGmOffset, 1, updatesCount);
                updatesGmOffset += this->tilingData_.updateColUbFactor;
                CopyOutUpdates(varGmOffset, 0, updatesCount);
                varGmOffset += this->tilingData_.updateColUbFactor;
            }
        }
        this->indicesQue_.FreeTensor(indicesLocal);
    }
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimd<T, U, MASK_T, splitCol, CAST_T, castType>::ProcessSplitRow()
{
    uint32_t indicesCount = this->tilingData_.indicesUbFactor;
    uint32_t updatesAlign = ops::CeilAlign(this->tilingData_.varShape[1], UB_AGLIN_VALUE / sizeof(T));
    for (uint64_t idx = 0; idx < this->indicesBlockLoop_; idx++) {
        if (idx == this->indicesBlockLoop_ - 1) {
            indicesCount = this->indicesTailLoopSize_;
        }
        uint64_t indicesGmOffset = this->blockIdx_ * this->tilingData_.normBlockIndices + 
                                   idx * this->tilingData_.indicesUbFactor;
        uint64_t updatesGmOffset = indicesGmOffset * this->tilingData_.varShape[1];
        uint32_t updatesCount = indicesCount * this->tilingData_.varShape[1];
        this->CopyInIndices(indicesGmOffset, indicesCount);
        CopyInUpdates(updatesGmOffset, indicesCount, this->tilingData_.varShape[1]);

        auto eventIDMTE2_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2_S);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2_S);
        LocalTensor<U> indicesLocal = this->indicesQue_.template DeQue<U>();
        for (uint32_t i = 0; i < indicesCount; i++) {
            U indicesValue = indicesLocal.GetValue(i);
            if (indicesValue < 0 || indicesValue >= this->tilingData_.varShape[0]) {
                continue;
            }
            if (static_cast<MASK_T>(indicesGmOffset + i) != this->workspaceMask_.GetValue(indicesValue)) {
                continue;
            }
            uint64_t varGmOffset = indicesValue * this->tilingData_.varStride;
            uint32_t updateOffset = i * updatesAlign;
            CopyOutUpdates(varGmOffset, updateOffset, this->tilingData_.varShape[1]);
        }

        this->indicesQue_.FreeTensor(indicesLocal);
    }
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimd<T, U, MASK_T, splitCol, CAST_T, castType>::Process()
{
    if (GetBlockIdx() >= this->tilingData_.usedCoreNum) {
        return;
    }

    if constexpr (splitCol) {
        ProcessSplitCol();
    } else {
        this->CalcMask();
        SyncAll();
        ProcessSplitRow();
    }
}

}
#endif  // SCATTER_UPDATE_DETERMINISTIC_SIMD_H