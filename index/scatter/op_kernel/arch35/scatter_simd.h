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
 * \file scatter_simd.h
 * \brief
 */

#ifndef ASCENDC_SCATTER_SIMD_H_
#define ASCENDC_SCATTER_SIMD_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SCATTER {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIM_2 = 2;

template <typename IDX_T, const uint16_t SIMD_MODE>
class ScatterSimd
{
public:
    __aicore__ inline ScatterSimd(const ScatterTilingData* tiling, TPipe* pipe) : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y);
    __aicore__ inline void Process();
    template <typename T>
    __aicore__ inline void CopyIn(
        LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst, uint32_t copyLen);
    __aicore__ inline void CopyOut(LocalTensor<int8_t> yLocal, int64_t offset, uint32_t nBurst, uint32_t copyLen);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void GetIndex(int64_t idx, int64_t endIdx, int64_t& batchDst, int64_t& scatterDst);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void NoSplitColProcessForMode0(int64_t colsAlign);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void NoSplitColProcessForMode1(int64_t colsAlign);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void SplitColProcess(int64_t colsAlign);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void SplitColProcessForMode0(int64_t colsAlign);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void SplitColProcessForMode1(int64_t colsAlign);
    __aicore__ inline void CopyInAndOut(int64_t srcOffset, int64_t dstOffset, int64_t cols);

private:
    GlobalTensor<IDX_T> indicesGm_;
    GlobalTensor<int8_t> updatesGm_;
    GlobalTensor<int8_t> yGm_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> inQueue_;

    TPipe* pipe_ = nullptr;
    const ScatterTilingData* tilingData_;

    int64_t blockIdx_{0};
    int64_t curCoreData_{0};
    int64_t maxIndex_{0};
    int64_t indicesOffsetBase_{0};
    int64_t curIndexSize_{0};
};

template <typename IDX_T, const uint16_t SIMD_MODE>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y)
{
    yGm_.SetGlobalBuffer((__gm__ int8_t*)(y));
    updatesGm_.SetGlobalBuffer((__gm__ int8_t*)(updates));
    indicesGm_.SetGlobalBuffer((__gm__ IDX_T*)(indices));

    pipe_->InitBuffer(inQueue_, DB_BUFFER, tilingData_->loopLength * tilingData_->dtypeSize);
    pipe_->InitBuffer(indexBuf_, tilingData_->indicesUbSize);

    blockIdx_ = GetBlockIdx();
    curCoreData_ = blockIdx_ != (tilingData_->aivCoreNum - 1) ? tilingData_->blockFactor : tilingData_->tailBlockData;
    maxIndex_ = tilingData_->indicesUbSize / sizeof(IDX_T);
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <typename T>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::CopyIn(
    LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename IDX_T, const uint16_t SIMD_MODE>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::CopyOut(
    LocalTensor<int8_t> yLocal, int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(yGm_[offset], yLocal, dataCoptExtParams);
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::GetIndex(
    int64_t idx, int64_t endIdx, int64_t& batchDst, int64_t& scatterDst)
{
    if (idx >= indicesOffsetBase_ + curIndexSize_) {
        int64_t copyLen = 0;
        if (maxIndex_ >= (endIdx - idx + 1) * IDX_DIM) {
            copyLen = (endIdx - idx + 1) * IDX_DIM;
        } else {
            copyLen = maxIndex_;
        }
        indicesOffsetBase_ = idx;
        curIndexSize_ = copyLen / IDX_DIM;
        LocalTensor<IDX_T> tmpLocal = indexBuf_.Get<IDX_T>();
        CopyIn(tmpLocal, indicesGm_, indicesOffsetBase_ * IDX_DIM, 1, copyLen);

        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
    LocalTensor<IDX_T> indexLocal = indexBuf_.Get<IDX_T>();
    if constexpr (IDX_DIM == DIM_2) {
        batchDst = static_cast<int64_t>(indexLocal.GetValue((idx - indicesOffsetBase_) * DIM_2));
        scatterDst += static_cast<int64_t>(indexLocal.GetValue((idx - indicesOffsetBase_) * DIM_2 + 1));
    } else {
        scatterDst += static_cast<int64_t>(indexLocal.GetValue(idx - indicesOffsetBase_));
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::NoSplitColProcess(int64_t colsAlign)
{
    if constexpr (SIMD_MODE == 0) {
        NoSplitColProcessForMode0<IDX_DIM>(colsAlign);
    } else {
        NoSplitColProcessForMode1<IDX_DIM>(colsAlign);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::NoSplitColProcessForMode0(int64_t colsAlign)
{
    int64_t onceMaxRows = tilingData_->loopLength * tilingData_->dtypeSize / colsAlign;
    int64_t loopNum = Ops::Base::CeilDiv(curCoreData_, onceMaxRows);
    // indice -> b, 2 or b
    // update -> b1, o, s1, i
    // output -> b, o, s, i
    int64_t srcStart = blockIdx_ * tilingData_->blockFactor;
    int64_t srcEnd = srcStart + curCoreData_ - 1;
    int64_t endBatch = srcEnd / tilingData_->updatesDim1;
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (curCoreData_ - i * onceMaxRows) : onceMaxRows;
        int64_t cols = tilingData_->updatesDim2 * tilingData_->updatesDim3;

        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        int64_t srcOffset = (srcStart + i * onceMaxRows) * cols * tilingData_->dtypeSize;
        CopyIn(xLocal, updatesGm_, srcOffset, rows, cols * tilingData_->dtypeSize);
        inQueue_.EnQue<int8_t>(xLocal);

        LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
        for (int64_t j = 0; j < rows; j++) {
            int64_t srcIdx = srcStart + i * onceMaxRows + j;
            int64_t batchI = srcIdx / tilingData_->updatesDim1;
            int64_t outerI = srcIdx - batchI * tilingData_->updatesDim1;
            int64_t batchDst = batchI;
            int64_t scatterDst = 0;
            GetIndex<IDX_DIM>(batchI, endBatch, batchDst, scatterDst);
            int64_t dstOffset = ((batchDst * tilingData_->inputDim1 + outerI) * tilingData_->inputDim2 + scatterDst) *
                                tilingData_->inputDim3 * tilingData_->dtypeSize;
            CopyOut(yLocal[j * colsAlign], dstOffset, 1, cols * tilingData_->dtypeSize);
        }
        inQueue_.FreeTensor(yLocal);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::NoSplitColProcessForMode1(int64_t colsAlign)
{
    int64_t onceMaxRows = tilingData_->loopLength * tilingData_->dtypeSize / colsAlign;
    int64_t loopNum = Ops::Base::CeilDiv(curCoreData_, onceMaxRows);
    // indice -> b, 2 or b
    // update -> b1, o, s1, i
    // output -> b, o, s, i
    int64_t srcStart = blockIdx_ * tilingData_->blockFactor;
    int64_t srcEnd = srcStart + curCoreData_ - 1;
    int64_t endBatch = srcEnd / (tilingData_->updatesDim1 * tilingData_->updatesDim2);
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (curCoreData_ - i * onceMaxRows) : onceMaxRows;
        int64_t cols = tilingData_->updatesDim3;

        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        int64_t srcOffset = (srcStart + i * onceMaxRows) * tilingData_->updatesDim3 * tilingData_->dtypeSize;
        CopyIn(xLocal, updatesGm_, srcOffset, rows, cols * tilingData_->dtypeSize);
        inQueue_.EnQue<int8_t>(xLocal);

        LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
        for (int64_t j = 0; j < rows; j++) {
            int64_t srcIdx = srcStart + i * onceMaxRows + j;
            int64_t batchI = srcIdx / (tilingData_->updatesDim1 * tilingData_->updatesDim2);
            int64_t batchRes = srcIdx - batchI * (tilingData_->updatesDim1 * tilingData_->updatesDim2);
            int64_t outerI = batchRes / tilingData_->updatesDim2;
            int64_t scatterI = batchRes - outerI * tilingData_->updatesDim2;
            int64_t batchDst = batchI;
            int64_t scatterDst = scatterI;
            GetIndex<IDX_DIM>(batchI, endBatch, batchDst, scatterDst);
            int64_t dstOffset = ((batchDst * tilingData_->inputDim1 + outerI) * tilingData_->inputDim2 + scatterDst) *
                                tilingData_->inputDim3 * tilingData_->dtypeSize;
            CopyOut(yLocal[j * colsAlign], dstOffset, 1, cols * tilingData_->dtypeSize);
        }
        inQueue_.FreeTensor(yLocal);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::SplitColProcess(int64_t colsAlign)
{
    if constexpr (SIMD_MODE == 0) {
        SplitColProcessForMode0<IDX_DIM>(colsAlign);
    } else {
        SplitColProcessForMode1<IDX_DIM>(colsAlign);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::CopyInAndOut(int64_t srcOffset, int64_t dstOffset, int64_t cols)
{
    LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
    CopyIn(xLocal, updatesGm_, srcOffset, 1, cols * tilingData_->dtypeSize);
    inQueue_.EnQue<int8_t>(xLocal);

    LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
    CopyOut(yLocal, dstOffset, 1, cols * tilingData_->dtypeSize);
    inQueue_.FreeTensor(yLocal);
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::SplitColProcessForMode0(int64_t colsAlign)
{
    // indice -> b, 2 or b
    // update -> b1, o, s1, i
    // output -> b, o, s, i
    int64_t srcStart = blockIdx_ * tilingData_->blockFactor;
    int64_t srcEnd = srcStart + curCoreData_ - 1;
    int64_t endBatch = srcEnd / tilingData_->updatesDim1;
    int64_t colsTotal = tilingData_->updatesDim2 * tilingData_->updatesDim3;
    int64_t loopNum = Ops::Base::CeilDiv(colsTotal, tilingData_->loopLength);
    for (int64_t i = 0; i < curCoreData_; i++) {
        int64_t srcIdx = srcStart + i;
        int64_t batchI = srcIdx / tilingData_->updatesDim1;
        int64_t outerI = srcIdx - batchI * tilingData_->updatesDim1;

        int64_t batchDst = batchI;
        int64_t scatterDst = 0;
        GetIndex<IDX_DIM>(batchI, endBatch, batchDst, scatterDst);
        int64_t dstIndex = ((batchDst * tilingData_->inputDim1 + outerI) * tilingData_->inputDim2 + scatterDst) *
                           tilingData_->inputDim3;

        int64_t srcIndex = (srcStart + i) * colsTotal;
        for (int64_t j = 0; j < loopNum; j++) {
            int64_t cols = (j == loopNum - 1) ? (colsTotal - j * tilingData_->loopLength) : tilingData_->loopLength;
            int64_t srcOffset = (srcIndex + j * tilingData_->loopLength) * tilingData_->dtypeSize;
            int64_t dstOffset = (dstIndex + j * tilingData_->loopLength) * tilingData_->dtypeSize;
            CopyInAndOut(srcOffset, dstOffset, cols);
        }
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::SplitColProcessForMode1(int64_t colsAlign)
{
    // indice -> b, 2 or b
    // update -> b1, o, s1, i
    // output -> b, o, s, i
    int64_t srcStart = blockIdx_ * tilingData_->blockFactor;
    int64_t srcEnd = srcStart + curCoreData_ - 1;
    int64_t endBatch = srcEnd / (tilingData_->updatesDim1 * tilingData_->updatesDim2);
    int64_t loopNum = Ops::Base::CeilDiv(tilingData_->updatesDim3, tilingData_->loopLength);
    for (int64_t i = 0; i < curCoreData_; i++) {
        int64_t srcIdx = srcStart + i;
        int64_t batchI = srcIdx / (tilingData_->updatesDim1 * tilingData_->updatesDim2);
        int64_t batchRes = srcIdx - batchI * (tilingData_->updatesDim1 * tilingData_->updatesDim2);
        int64_t outerI = batchRes / tilingData_->updatesDim2;
        int64_t scatterI = batchRes - outerI * tilingData_->updatesDim2;

        int64_t batchDst = batchI;
        int64_t scatterDst = scatterI;
        GetIndex<IDX_DIM>(batchI, endBatch, batchDst, scatterDst);
        int64_t dstIndex = ((batchDst * tilingData_->inputDim1 + outerI) * tilingData_->inputDim2 + scatterDst) *
                           tilingData_->inputDim3;

        int64_t srcIndex = (srcStart + i) * tilingData_->updatesDim3;
        for (int64_t j = 0; j < loopNum; j++) {
            int64_t cols =
                (j == loopNum - 1) ? (tilingData_->updatesDim3 - j * tilingData_->loopLength) : tilingData_->loopLength;
            int64_t srcOffset = (srcIndex + j * tilingData_->loopLength) * tilingData_->dtypeSize;
            int64_t dstOffset = (dstIndex + j * tilingData_->loopLength) * tilingData_->dtypeSize;
            CopyInAndOut(srcOffset, dstOffset, cols);
        }
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
__aicore__ inline void ScatterSimd<IDX_T, SIMD_MODE>::Process()
{
    if (blockIdx_ >= tilingData_->aivCoreNum) {
        return;
    }
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
    int64_t colsAlign = 0;
    if constexpr (SIMD_MODE == 0) {
        // split b*o
        colsAlign = (tilingData_->updatesDim2 * tilingData_->updatesDim3 * tilingData_->dtypeSize + ubBlockSize - 1) /
                    ubBlockSize * ubBlockSize;
    } else {
        // split b*o*s
        colsAlign = (tilingData_->updatesDim3 * tilingData_->dtypeSize + ubBlockSize - 1) / ubBlockSize * ubBlockSize;
    }
    if (colsAlign <= tilingData_->loopLength * tilingData_->dtypeSize) {
        if (tilingData_->indicesDim == DIM_2) {
            NoSplitColProcess<DIM_2>(colsAlign);
        } else {
            NoSplitColProcess<1>(colsAlign);
        }
    } else {
        if (tilingData_->indicesDim == DIM_2) {
            SplitColProcess<DIM_2>(colsAlign);
        } else {
            SplitColProcess<1>(colsAlign);
        }
    }
}
} // namespace SCATTER

#endif // ASCENDC_SCATTER_SIMD_H_
