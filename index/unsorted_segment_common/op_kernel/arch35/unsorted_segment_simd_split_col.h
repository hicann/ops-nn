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
 * \file unsorted_segment_simd_split_col.h
 * \brief unsorted_segment_simd_split_col
 */
#ifndef UNSORTED_SEGMENT_SIMD_SPLIT_COL_H_
#define UNSORTED_SEGMENT_SIMD_SPLIT_COL_H_

#include "unsorted_segment_base.h"

namespace UnsortedSegment {
using namespace AscendC;
constexpr uint32_t SPLIT_COL_DB_BUF = 2;

template <typename X_T, typename IDS_T, uint8_t Mode>
class KernelSimdSplitCol
{
public:
    __aicore__ inline KernelSimdSplitCol(const UnsortedSegmentSimdSplitColTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<IDS_T> idsGm_;
    TQue<QuePosition::VECIN, SPLIT_COL_DB_BUF> xQue_;
    TQue<QuePosition::VECIN, SPLIT_COL_DB_BUF> idsQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSimdSplitColTilingData* td_;
};

template <typename X_T, typename IDS_T, uint8_t Mode>
__aicore__ inline void KernelSimdSplitCol<X_T, IDS_T, Mode>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));

    pipe_->InitBuffer(xQue_, SPLIT_COL_DB_BUF, td_->baseS * td_->baseA * sizeof(X_T));
    pipe_->InitBuffer(
        idsQue_, SPLIT_COL_DB_BUF, Aligned(static_cast<uint64_t>(td_->baseS * sizeof(IDS_T)), ONE_BLOCK_SIZE));
    pipe_->InitBuffer(yQue_, 1, td_->outputOuterDim * td_->baseA * sizeof(X_T));
}

template <typename X_T, typename IDS_T, uint8_t Mode>
__aicore__ inline void KernelSimdSplitCol<X_T, IDS_T, Mode>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    uint64_t curCoreCols = GetBlockIdx() != (GetBlockNum() - 1) ? td_->normBlockData : td_->tailBlockData;
    uint64_t blockOffset = GetBlockIdx() * td_->normBlockData;
    uint64_t aLoopNum = Ops::Base::CeilDiv(curCoreCols, td_->baseA);
    uint64_t sLoopNum = Ops::Base::CeilDiv(td_->inputOuterDim, td_->baseS);

    for (uint64_t aLoop = 0; aLoop < aLoopNum; aLoop++) {
        uint64_t cols = (aLoop == aLoopNum - 1) ? (curCoreCols - aLoop * td_->baseA) : td_->baseA;
        uint64_t colsAlign = Aligned(static_cast<uint64_t>(cols * sizeof(X_T)), ONE_BLOCK_SIZE) / sizeof(X_T);

        LocalTensor<X_T> yLocal = yQue_.AllocTensor<X_T>();
        if constexpr (Mode == 0) {
            Duplicate(yLocal, GetDtypeMax<X_T>(), td_->outputOuterDim * colsAlign);
        }
        yQue_.EnQue<X_T>(yLocal);
        yLocal = yQue_.DeQue<X_T>();
        for (uint64_t sLoop = 0; sLoop < sLoopNum; sLoop++) {
            uint64_t rows = (sLoop == sLoopNum - 1) ? (td_->inputOuterDim - sLoop * td_->baseS) : td_->baseS;

            LocalTensor<IDS_T> idsLocal = idsQue_.AllocTensor<IDS_T>();
            CopyIn(idsLocal, idsGm_, sLoop * td_->baseS, 1, rows, 0);
            idsQue_.EnQue<IDS_T>(idsLocal);
            event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

            LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();
            uint64_t offset = blockOffset + sLoop * td_->baseS * td_->innerDim + aLoop * td_->baseA;
            CopyIn(xLocal, xGm_, offset, rows, cols, td_->innerDim - cols);
            xQue_.EnQue<X_T>(xLocal);

            idsLocal = idsQue_.DeQue<IDS_T>();
            xLocal = xQue_.DeQue<X_T>();
            for (uint64_t i = 0; i < rows; i++) {
                uint64_t dstIdx = idsLocal.GetValue(i);
                if (dstIdx < 0 || dstIdx >= td_->outputOuterDim) {
                    continue;
                }
                uint64_t dstOffset = dstIdx * colsAlign;
                if constexpr (Mode ==0) {
                    AscendC::Min(yLocal[dstOffset], yLocal[dstOffset], xLocal[i * colsAlign], cols);
                }
            }
            xQue_.FreeTensor(xLocal);
            idsQue_.FreeTensor(idsLocal);
        }
        yQue_.EnQue<X_T>(yLocal);
        yLocal = yQue_.DeQue<X_T>();
        uint64_t offset = blockOffset + aLoop * td_->baseA;
        CopyOut(yGm_, yLocal, offset, td_->outputOuterDim, cols, 0, td_->innerDim - cols);
        yQue_.FreeTensor(yLocal);
    }
}
} // namespace UnsortedSegment

#endif