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
 * \file scatter_update_mask_simd.h
 * \brief scatter_update
 */
#ifndef SCATTER_UPDATE_MASK_SIMD_IMPL_H
#define SCATTER_UPDATE_MASK_SIMD_IMPL_H

#include "scatter_update_simd_common.h"

namespace ScatterUpdate {
using namespace AscendC;

constexpr int64_t INDICES_MAX_BATCH_COPY_THRESHOLD = 4 * 1024;

template <typename T, typename U>
class ScatterUpdateMaskSIMDImpl : public ScatterUpdateSimdCommon<T, U, false> {
public:
    __aicore__ inline ScatterUpdateMaskSIMDImpl(const ScatterUpdateTilingData& tilingData, TPipe& pipe) :
        ScatterUpdateSimdCommon<T, U, false> (tilingData, pipe) {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSplitCol();
    __aicore__ inline void ProcessSplitColForBatchIndices(int64_t start,
                                                          int64_t end,
                                                          LocalTensor<U>& indicesLocal,
                                                          LocalTensor<uint8_t>& maskTensor);
    __aicore__ inline void ProcessNotSplitCol();

    __aicore__ inline void ProcessNotSplitUpdates(uint32_t rowLen, uint32_t alignedLength, uint32_t originalLength);
    __aicore__ inline void ProcessSplitUpdates(U indicesValue, uint32_t outLen, uint64_t oneRowOffSet);

private:
    TBuf<TPosition::VECCALC> maskBuf_;
    uint64_t blockOffset_ = 0;
    uint64_t curBlockTailRowNum_ = 0;
    uint64_t indicesRowCopyed_ = 0;
};

template <typename T, typename U>
__aicore__ inline void ScatterUpdateMaskSIMDImpl<T, U>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    this->blockIdx_ = GetBlockIdx();
    this->varRefGm_.SetGlobalBuffer((__gm__ T *)(var));
    this->indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
    this->updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));

    blockOffset_ = this->blockIdx_ * this->tilingData_.normBlockRowNum;
    this->pipe_.InitBuffer(maskBuf_, this->tilingData_.varShape[0] * sizeof(uint8_t));

    if (this->tilingData_.normNeedSplitRow == 1) {
        this->pipe_.InitBuffer(this->indicesInQueue_,
            1,
            ops::CeilAlign(
                this->tilingData_.indicesBatchCopySizeAlign, static_cast<uint64_t>(platform::GetUbBlockSize())));
    } else {
        this->pipe_.InitBuffer(this->indicesInQueue_,
            DB_BUFFER,
            ops::CeilAlign(
                this->tilingData_.processRowPerUb * sizeof(U), static_cast<uint64_t>(platform::GetUbBlockSize())));
    }

    this->pipe_.InitBuffer(this->updateInQueue_, DB_BUFFER, this->tilingData_.updateUbSize);

    LocalTensor<uint8_t> maskTensor = maskBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(maskTensor, static_cast<uint8_t>(0), this->tilingData_.varShape[0]);
}

template <typename T, typename U>
__aicore__ inline void ScatterUpdateMaskSIMDImpl<T, U>::Process()
{
    if (this->blockIdx_ >= this->tilingData_.usedCoreNum) {
        return;
    }

    if (this->tilingData_.normNeedSplitRow == 1) {
        ProcessSplitCol();
    } else {
        ProcessNotSplitCol();
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterUpdateMaskSIMDImpl<T, U>::ProcessNotSplitCol()
{
    if (this->blockIdx_ == AscendC::GetBlockNum() - 1) {
        curBlockTailRowNum_ = this->tilingData_.tailBlockRowNum -
                              (this->tilingData_.tailBlockLoop - 1) * (this->tilingData_.processRowPerUb);
    } else {
        curBlockTailRowNum_ = this->tilingData_.normBlockRowNum -
                              (this->tilingData_.normBlockLoop - 1) * (this->tilingData_.processRowPerUb);
    }

    int64_t rowLoop = (this->blockIdx_ == AscendC::GetBlockNum() - 1) ? this->tilingData_.tailBlockLoop
                                                                      : this->tilingData_.normBlockLoop;

    for (int64_t i = 0; i < rowLoop; ++i) {
        int64_t updatesOffset = (blockOffset_ + i * this->tilingData_.processRowPerUb) * this->tilingData_.colTotal;
        int64_t indicesOffset = blockOffset_ + i * this->tilingData_.processRowPerUb;

        uint32_t rowByUb = i == rowLoop - 1 ? curBlockTailRowNum_ : this->tilingData_.processRowPerUb;
        this->CopyInIndices(indicesOffset, rowByUb);
        this->CopyInUpdates(updatesOffset, rowByUb, this->tilingData_.colTotal);
        this->ProcessNotSplitUpdates(rowByUb, this->tilingData_.processColPerUb, this->tilingData_.colTotal);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterUpdateMaskSIMDImpl<T, U>::ProcessNotSplitUpdates(
    uint32_t rowLen, uint32_t alignedLength, uint32_t originalLength)
{
    LocalTensor<U> indicesLocal = this->indicesInQueue_.template DeQue<U>();
    LocalTensor<T> updatesLocal = this->updateInQueue_.template DeQue<T>();
    LocalTensor<uint8_t> maskTensor = maskBuf_.Get<uint8_t>();

    for (int64_t i = 0; i < rowLen; ++i) {
        U indicesValue = indicesLocal.GetValue(i);
        if (static_cast<int64_t>(indicesValue) < 0 ||
            static_cast<int64_t>(indicesValue) >= this->tilingData_.varShape[0] ||
            maskTensor.GetValue(static_cast<int64_t>(indicesValue))) {
            continue;
        }
        int64_t indicesOffset = static_cast<int64_t>(indicesValue) * this->tilingData_.varStride;
        this->CopyOutUpdates(indicesOffset, originalLength, updatesLocal[i * alignedLength]);
        maskTensor.SetValue(static_cast<int64_t>(indicesValue), 1);
    }

    this->updateInQueue_.FreeTensor(updatesLocal);
    this->indicesInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterUpdateMaskSIMDImpl<T, U>::ProcessSplitColForBatchIndices(int64_t start, int64_t end, LocalTensor<U>& indicesLocal, LocalTensor<uint8_t>& maskTensor)
{
    DataCopyExtParams inParams = { 1, static_cast<uint32_t>((end - start) * sizeof(U)), 0, 0, 0 };
    DataCopyPadExtParams<U> padParams = { false, 0, 0, 0 };
    DataCopyPad(indicesLocal, this->indicesGm_[blockOffset_ + start], inParams, padParams);
    this->indicesInQueue_.template EnQue(indicesLocal);
    indicesLocal = this->indicesInQueue_.template DeQue<U>();

    for (int64_t i = start; i < end; i++) {
        int64_t updatesOffset = (blockOffset_ + i) * this->tilingData_.colTotal;
        int64_t indicesOffset = blockOffset_ + i;

        U indicesValue = indicesLocal.GetValue(i);

        if (static_cast<int64_t>(indicesValue) < 0 ||
            static_cast<int64_t>(indicesValue) >= this->tilingData_.varShape[0] ||
            maskTensor.GetValue(static_cast<int64_t>(indicesValue))) {
            continue;
        }

        for (int64_t j = 0; j < this->tilingData_.colLoopByUb - 1; j++) {
            this->CopyInUpdates(
                updatesOffset + j * this->tilingData_.processColPerUb, INDICES_ONE, this->tilingData_.processColPerUb);
            ProcessSplitUpdates(indicesValue, this->tilingData_.processColPerUb, j * this->tilingData_.processColPerUb);
        }
        this->CopyInUpdates(updatesOffset + (this->tilingData_.colLoopByUb - 1) * this->tilingData_.processColPerUb,
            INDICES_ONE,
            this->tilingData_.processColTail);
        ProcessSplitUpdates(indicesValue,
            this->tilingData_.processColTail,
            (this->tilingData_.colLoopByUb - 1) * this->tilingData_.processColPerUb);

        maskTensor.SetValue(static_cast<int64_t>(indicesValue), 1);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterUpdateMaskSIMDImpl<T, U>::ProcessSplitCol()
{
    int64_t rowLoop = (this->blockIdx_ == AscendC::GetBlockNum() - 1) ? this->tilingData_.tailBlockLoop
                                                                      : this->tilingData_.normBlockLoop;
    int64_t batchIndicesNum = INDICES_MAX_BATCH_COPY_THRESHOLD / sizeof(U);
    int64_t batchRowLoop = 1;
    if (rowLoop > batchIndicesNum) {
        batchRowLoop = ops::CeilAlign(rowLoop / batchIndicesNum, 1L);
    }
    LocalTensor<uint8_t> maskTensor = maskBuf_.Get<uint8_t>();
    LocalTensor<U> indicesLocal = this->indicesInQueue_.template AllocTensor<U>();

    for (int64_t i = 0; i < batchRowLoop - 1; i++) {
        ProcessSplitColForBatchIndices(i * batchIndicesNum, (i + 1) * batchIndicesNum, indicesLocal, maskTensor);
    }
    ProcessSplitColForBatchIndices((batchRowLoop - 1) * batchIndicesNum, rowLoop, indicesLocal, maskTensor);
    this->indicesInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterUpdateMaskSIMDImpl<T, U>::ProcessSplitUpdates(
    U indicesValue, uint32_t outLen, uint64_t oneRowOffSet)
{
    LocalTensor<T> updatesLocal = this->updateInQueue_.template DeQue<T>();
    int64_t indicesOffset = static_cast<int64_t>(indicesValue) * this->tilingData_.varStride + oneRowOffSet;
    this->CopyOutUpdates(indicesOffset, outLen, updatesLocal);
    this->updateInQueue_.FreeTensor(updatesLocal);
}
}  // namespace ScatterUpdate
#endif  // SCATTER_UPDATE_MASK_SIMD_IMPL