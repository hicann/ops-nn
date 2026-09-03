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
 * \file scatter_nd_update_deterministic_simd.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_DETER_SIMD_H
#define SCATTER_ND_UPDATE_DETER_SIMD_H

#include "kernel_operator.h"
#include "scatter_nd_update_common.h"
#include "op_kernel/math_util.h"
namespace ScatterNdUpdate {
using namespace AscendC;

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T = INDICES_T>
class ScatterNdUpdateDeterministicSimd
    : public ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T> {
public:
    __aicore__ inline ScatterNdUpdateDeterministicSimd(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe)
        : ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSplitCol();

private:
    __aicore__ inline void CopyInUpdate(LocalTensor<PARAMS_T>& updateLocal);
    __aicore__ inline void CopyOutUpdate(LocalTensor<PARAMS_T>& updateLocal, uint64_t varGmOffSet);
    __aicore__ inline void InitSplitCol(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);

private:
    TYPE_T updateOffSet = 0;
    TYPE_T indiceOffSet = 0;
    TYPE_T idxLoopSize = 0;
    TYPE_T updateLoopSize = 0;
    int32_t updateCount = 0;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> inQueue_;

    int64_t colBlockLoop_ = 0;
    int64_t colBlockTail_ = 0;
    int64_t colOffset_ = 0;
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
{
    if (this->tiling_.isPcieThrough == 1) {
        InitSplitCol(x, indices, updates, y, workspace);
    } else {
        this->InitBase(x, indices, updates, y, workspace);
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::InitSplitCol(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
{
    this->blockIdx = GetBlockIdx();
    this->indicesUbFactor = this->tiling_.indicesUbFactor;
    this->rankSize_ = this->tiling_.rankSize;

    this->idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    this->updateGm.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    this->outputGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);

    if (this->blockIdx >= this->tiling_.usedCoreNumForCol) {
        return;
    }

    int64_t colCount = (this->blockIdx == this->tiling_.usedCoreNumForCol - 1) ? this->tiling_.tailBlockColNum :
                                                                                 this->tiling_.normBlockColNum;
    int64_t colUbFactor = this->tiling_.updateColUbFactor;
    colBlockLoop_ = Ops::Base::CeilDiv(colCount, colUbFactor);
    colBlockTail_ = colCount - colUbFactor * (colBlockLoop_ - 1);
    colOffset_ = this->blockIdx * this->tiling_.normBlockColNum;

    this->pipe_.InitBuffer(
        this->indicesQue_, 1,
        Ops::Base::CeilAlign(this->indicesUbFactor * this->rankSize_ * sizeof(INDICES_T), UB_AGLIN_VALUE));
    this->pipe_.InitBuffer(this->strideBuf_, MAX_SHAPE_RANK * sizeof(INDICES_T));
    this->pipe_.InitBuffer(this->outOfstBuf_,
                           Ops::Base::CeilAlign(this->indicesUbFactor * sizeof(OFFSET_T), UB_AGLIN_VALUE));
    this->pipe_.InitBuffer(inQueue_, 2, Ops::Base::CeilAlign(colUbFactor * sizeof(PARAMS_T), UB_AGLIN_VALUE));

    LocalTensor<INDICES_T> strideLocal = this->strideBuf_.template Get<INDICES_T>();
    for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        strideLocal(i) = static_cast<INDICES_T>(this->tiling_.strideList[i]);
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Process()
{
    // if input is empty, return directly
    if (this->tiling_.sliceSize == 0) {
        return;
    }
    if (this->tiling_.isPcieThrough == 1) {
        ProcessSplitCol();
        return;
    }
    SyncAll();
    this->CalcMask();
    SyncAll();

    if (this->blockIdx >= this->tiling_.usedCoreNumBefore) {
        return;
    }
    this->pipe_.Reset();
    this->pipe_.InitBuffer(inQueue_, 2,
                           Ops::Base::CeilAlign(this->tiling_.afterAxisFactor * sizeof(PARAMS_T), UB_AGLIN_VALUE));

    if (this->blockIdx == this->tiling_.usedCoreNumBefore - 1) {
        this->currBlockHandleIdx = this->tiling_.tailCoreIndexCount;
    } else {
        this->currBlockHandleIdx = this->tiling_.eachCoreIndexCount;
    }
    this->idxLoopSize = Ops::Base::CeilDiv(this->currBlockHandleIdx, static_cast<TYPE_T>(this->tiling_.indicesFactor));
    this->updateLoopSize = this->tiling_.updateLoopSize;
    this->indiceBlockOffSet = this->blockIdx * this->tiling_.eachCoreIndexCount;
    for (TYPE_T i = 0; i < this->idxLoopSize; i++) {
        // simd每次只处理一行
        this->indiceOffSet = this->indiceBlockOffSet + i;

        int64_t globalValRowIdx = this->varIdxGm(this->indiceOffSet);
        // 越界校验
        if (globalValRowIdx < 0 || globalValRowIdx >= this->tiling_.outputStorageShapeSize) {
            continue;
        }

        // 获取行对应varIdx
        if (this->maskGm(globalValRowIdx / this->tiling_.sliceSize) != this->indiceOffSet) {
            continue;
        }

        for (TYPE_T j = 0; j < this->updateLoopSize; j++) {
            LocalTensor<PARAMS_T> updateLocal = inQueue_.template AllocTensor<PARAMS_T>();
            this->updateOffSet = this->indiceOffSet * this->tiling_.sliceSize + j * this->tiling_.afterAxisFactor;
            uint64_t varGmOffSet = globalValRowIdx + j * this->tiling_.afterAxisFactor;
            if (j == this->updateLoopSize - 1) {
                this->updateCount = this->tiling_.updateTailNum;
            } else {
                this->updateCount = this->tiling_.afterAxisFactor;
            }
            CopyInUpdate(updateLocal);
            inQueue_.EnQue(updateLocal);
            updateLocal = inQueue_.DeQue<PARAMS_T>();
            CopyOutUpdate(updateLocal, varGmOffSet);
            inQueue_.template FreeTensor(updateLocal);
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::ProcessSplitCol()
{
    if (this->blockIdx >= this->tiling_.usedCoreNumForCol) {
        return;
    }

    int64_t colUbFactor = this->tiling_.updateColUbFactor;
    int64_t varFullDimSize = this->tiling_.outputStorageShapeSize;
    uint32_t indicesCount = static_cast<uint32_t>(this->indicesUbFactor);

    // 每个核均处理所有的索引，不同之处在于每个核处理的update列不同.
    for (uint64_t idx = 0; idx < this->tiling_.colIndicesLoopSize; idx++) {
        if (idx == this->tiling_.colIndicesLoopSize - 1) {
            indicesCount = static_cast<uint32_t>(this->tiling_.colIndicesTailNum);
        }
        uint64_t indicesGmOffset = idx * this->indicesUbFactor;
        this->CopyInIndices(indicesGmOffset * this->rankSize_, indicesCount * this->rankSize_);

        LocalTensor<INDICES_T> indicesLocal = this->indicesQue_.template DeQue<INDICES_T>();
        LocalTensor<OFFSET_T> flatOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        this->ComputeOutOfset(indicesLocal, flatOfstLocal, static_cast<int32_t>(indicesCount),
                              static_cast<int32_t>(this->rankSize_));
        this->indicesQue_.FreeTensor(indicesLocal);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        for (uint32_t i = 0; i < indicesCount; i++) {
            OFFSET_T indicesValue = flatOfstLocal.GetValue(i);
            if (indicesValue < 0 || indicesValue >= varFullDimSize) {
                continue;
            }

            uint64_t updatesGmOffset = (indicesGmOffset + i) * static_cast<uint64_t>(this->tiling_.sliceSize) +
                                       colOffset_;
            uint64_t varGmOffset = static_cast<uint64_t>(indicesValue) + colOffset_;
            for (int64_t j = 0; j < colBlockLoop_; j++) {
                this->updateCount = (j == colBlockLoop_ - 1) ? static_cast<int32_t>(colBlockTail_) :
                                                               static_cast<int32_t>(colUbFactor);
                this->updateOffSet = updatesGmOffset + j * colUbFactor;
                uint64_t varOutOffset = varGmOffset + j * colUbFactor;

                LocalTensor<PARAMS_T> updateLocal = inQueue_.template AllocTensor<PARAMS_T>();
                CopyInUpdate(updateLocal);
                inQueue_.EnQue(updateLocal);
                updateLocal = inQueue_.DeQue<PARAMS_T>();
                CopyOutUpdate(updateLocal, varOutOffset);
                inQueue_.template FreeTensor(updateLocal);
            }
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyInUpdate(
    LocalTensor<PARAMS_T>& updateLocal)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->updateCount * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(updateLocal, this->updateGm[this->updateOffSet], xCopyParams, xPadParams);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyOutUpdate(
    LocalTensor<PARAMS_T>& updateLocal, uint64_t varGmOffSet)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->updateCount * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPad(this->outputGm[varGmOffSet], updateLocal[0], xCopyParams);
}
} // namespace ScatterNdUpdate

#endif // SCATTER_ND_UPDATE_DETER_SIMD_H
