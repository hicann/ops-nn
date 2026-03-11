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
 * \file unsorted_segment_simd_dyn_sort.h
 * \brief unsorted_segment_simd_dyn_sort
 */
#ifndef UNSORTED_SEGMENT_SIMD_DYN_SORT_H_
#define UNSORTED_SEGMENT_SIMD_DYN_SORT_H_

#include "unsorted_segment_base.h"

namespace UnsortedSegment {
using namespace AscendC;
constexpr uint32_t DYN_SORT_DB_BUF = 1;
constexpr uint32_t SORT_PADDING = 64;
constexpr uint32_t HELP_FRE = 2;
static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

template <typename X_T, typename IDS_T, uint8_t Mode, uint32_t CAST_MODE>
class KernelSimdDynSort
{
public:
    __aicore__ inline KernelSimdDynSort(const UnsortedSegmentSimdDynSortTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void ProcessIndices(uint64_t blockOffsetIdx, uint64_t sLoop, uint32_t rows, int64_t& arNum);
    __aicore__ inline void ComputeXSum(uint32_t cols, uint32_t colsAlign, int64_t arNum);
    __aicore__ inline void CopyResToGm(uint32_t cols, uint32_t colsAlign, uint64_t ubOffset, int64_t& arNum);
    __aicore__ inline void Compute();
    __aicore__ inline void Process();
    using CAST_T = typename CastType<IDS_T, CAST_MODE>::type;
private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<IDS_T> idsGm_;
    TQue<QuePosition::VECIN, DYN_SORT_DB_BUF> xQue_;
    TQue<QuePosition::VECIN, DYN_SORT_DB_BUF> idsQue_;
    TQue<QuePosition::VECIN, DYN_SORT_DB_BUF> idsCastQue_;
    TQue<QuePosition::VECOUT, 1> outQueueRes_;
    TBuf<QuePosition::VECCALC> noDupBuf_;
    TBuf<QuePosition::VECCALC> sortedIdxBuf_;
    TBuf<QuePosition::VECCALC> sortedKeyBuf_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf_;
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSimdDynSortTilingData* td_;
    static constexpr uint32_t vfLengthX_ = VF_SIZE / sizeof(X_T);
    static constexpr uint32_t shiftOffset_ = ONE_BLOCK_SIZE / sizeof(CAST_T);
};

template <typename X_T, typename IDS_T, uint8_t Mode, uint32_t CAST_MODE>
__aicore__ inline void KernelSimdDynSort<X_T, IDS_T, Mode, CAST_MODE>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    InitGm<X_T, Mode>(output, td_->outputOuterDim * td_->innerDim);

    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));

    pipe_->InitBuffer(xQue_, DYN_SORT_DB_BUF, td_->sortBaseS * td_->sortBaseA * sizeof(X_T));
    pipe_->InitBuffer(outQueueRes_, 1, td_->sortBaseS * td_->sortBaseA * sizeof(X_T));

    uint64_t idsAlignB32 = Aligned(static_cast<uint64_t>(td_->sortBaseS * sizeof(uint32_t)), ONE_BLOCK_SIZE);
    uint64_t idsAlign = Aligned(static_cast<uint64_t>(td_->sortBaseS * sizeof(CAST_T)), ONE_BLOCK_SIZE);

    pipe_->InitBuffer(idsQue_, DYN_SORT_DB_BUF, idsAlign);
    pipe_->InitBuffer(noDupBuf_, idsAlignB32 + SORT_PADDING);
    if constexpr (CAST_MODE == CAST_0) {
        pipe_->InitBuffer(sortedKeyBuf_, idsAlign + SORT_PADDING);
    } else {
        pipe_->InitBuffer(idsCastQue_, DYN_SORT_DB_BUF, idsAlign);
        pipe_->InitBuffer(sortedKeyBuf_, idsAlign + SORT_PADDING);
    }
    pipe_->InitBuffer(sortedIdxBuf_, idsAlignB32);
    pipe_->InitBuffer(sharedTmpBuf_, Aligned(static_cast<uint64_t>(td_->sortSharedBufSize), ONE_BLOCK_SIZE));
}

template <typename X_T, typename IDS_T, uint8_t Mode, uint32_t CAST_MODE>
__aicore__ inline void KernelSimdDynSort<X_T, IDS_T, Mode, CAST_MODE>::ProcessIndices(
    uint64_t blockOffsetIdx, uint64_t sLoop, uint32_t rows, int64_t& arNum)
{
    LocalTensor<IDS_T> idsLocal = idsQue_.AllocTensor<IDS_T>();
    CopyIn(idsLocal, idsGm_, blockOffsetIdx + sLoop * td_->sortBaseS, 1, rows, 0);
    idsQue_.EnQue<IDS_T>(idsLocal);

    idsLocal = idsQue_.DeQue<IDS_T>();
    LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.Get<uint32_t>();
    LocalTensor<CAST_T> sortedKeyLocal = sortedKeyBuf_.Get<CAST_T>();
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();
    LocalTensor<int32_t> noDupRes = noDupBuf_.Get<int32_t>();

    Duplicate(sortedKeyLocal, static_cast<CAST_T>(-1), static_cast<uint32_t>(shiftOffset_ * HELP_FRE + rows));
    LocalTensor<CAST_T> sortedDstLocal = sortedKeyLocal[shiftOffset_];

    if constexpr (CAST_MODE == CAST_0) {
        AscendC::Sort<IDS_T, false, sortConfig>(sortedDstLocal, sortedIdxLocal, idsLocal, sharedTmpBuffer, rows);
    } else {
        LocalTensor<CAST_T> idsCastLocal = idsCastQue_.AllocTensor<CAST_T>();
        IndicesSortCast<IDS_T, CAST_T, CAST_MODE>(idsLocal, idsCastLocal, noDupRes, rows);
        AscendC::Sort<CAST_T, false, sortConfig>(sortedDstLocal, sortedIdxLocal, idsCastLocal, sharedTmpBuffer, rows);
        idsCastQue_.FreeTensor(idsCastLocal);
    }

    Duplicate(noDupRes, 0, rows);
    arNum = UniqueGetElm<CAST_T>(sortedKeyLocal, noDupRes, rows);
    UniqueStat(noDupRes, arNum);
    idsQue_.FreeTensor(idsLocal);
}

template <typename X_T, typename IDS_T, uint8_t Mode, uint32_t CAST_MODE>
__aicore__ inline void KernelSimdDynSort<X_T, IDS_T, Mode, CAST_MODE>::ComputeXSum(uint32_t cols, uint32_t colsAlign, int64_t arNum)
{
    LocalTensor<uint32_t> sortedIdxLocal = sortedIdxBuf_.template Get<uint32_t>();
    LocalTensor<int32_t> noDupRes = noDupBuf_.template Get<int32_t>();
    LocalTensor<X_T> xLocal = xQue_.DeQue<X_T>();
    LocalTensor<X_T> resLocal = outQueueRes_.AllocTensor<X_T>();

    if constexpr (Mode == 0) {
        Duplicate(resLocal, GetDtypeMax<X_T>(), arNum * colsAlign);
    }
    outQueueRes_.EnQue<X_T>(resLocal);
    resLocal = outQueueRes_.DeQue<X_T>();

    uint32_t idxOffset = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(arNum); i++) {
        uint32_t segCount = static_cast<uint32_t>(noDupRes(i));
        for (uint32_t j = 0; j < segCount; j++) {
            uint32_t srcIdx = sortedIdxLocal.GetValue(idxOffset + j);
            if constexpr (Mode == 0) {
                AscendC::Min(resLocal[i * colsAlign], resLocal[i * colsAlign], xLocal[srcIdx * colsAlign], cols);
            }
        }
        idxOffset += segCount;
    }

    xQue_.FreeTensor<X_T>(xLocal);
    outQueueRes_.EnQue<X_T>(resLocal);
}

template <typename X_T, typename IDS_T, uint8_t Mode, uint32_t CAST_MODE>
__aicore__ inline void KernelSimdDynSort<X_T, IDS_T, Mode, CAST_MODE>::CopyResToGm(
    uint32_t cols, uint32_t colsAlign, uint64_t ubOffset, int64_t& arNum)
{
    LocalTensor<X_T> resLocal = outQueueRes_.DeQue<X_T>();
    LocalTensor<CAST_T> sortedKeyLocal = sortedKeyBuf_.Get<CAST_T>();
    LocalTensor<int32_t> noDupRes = noDupBuf_.Get<int32_t>();

    int32_t tmpIndex = shiftOffset_;

    if constexpr (Mode == 0) {
        SetAtomicMin<X_T>();
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(arNum); i++) {
        uint64_t dstIdx = sortedKeyLocal(tmpIndex);
        uint64_t offset = dstIdx * td_->innerDim + ubOffset;
        tmpIndex = tmpIndex + noDupRes(i);

        if (dstIdx < 0 || dstIdx >= td_->outputOuterDim) {
            continue;
        }

        CopyOut(yGm_, resLocal[i * colsAlign], offset, 1, cols);
    }
    SetAtomicNone();
    outQueueRes_.FreeTensor(resLocal);
}

template <typename X_T, typename IDS_T, uint8_t Mode, uint32_t CAST_MODE>
__aicore__ inline void KernelSimdDynSort<X_T, IDS_T, Mode, CAST_MODE>::Compute()
{
    uint64_t sIdx = GetBlockIdx() / td_->aTileNum;
    uint64_t aIdx = GetBlockIdx() % td_->aTileNum;
    uint64_t curCoreRows = sIdx != (td_->sTileNum - 1) ? td_->normBlockS : td_->tailBlockS;
    uint64_t curCoreCols = aIdx != (td_->aTileNum - 1) ? td_->normBlockA : td_->tailBlockA;

    uint64_t blockOffsetIdx = sIdx * td_->normBlockS;
    uint64_t blockOffsetX = sIdx * td_->normBlockS * td_->innerDim + aIdx * td_->normBlockA;

    uint64_t aLoopNum = Ops::Base::CeilDiv(curCoreCols, td_->sortBaseA);
    uint64_t sLoopNum = Ops::Base::CeilDiv(curCoreRows, td_->sortBaseS);

    for (uint64_t sLoop = 0; sLoop < sLoopNum; sLoop++) {
        uint32_t rows = (sLoop == sLoopNum - 1) ? (curCoreRows - sLoop * td_->sortBaseS) : td_->sortBaseS;
        int64_t arNum = 0;
        ProcessIndices(blockOffsetIdx, sLoop, rows, arNum);

        for (uint64_t aLoop = 0; aLoop < aLoopNum; aLoop++) {
            uint32_t cols = (aLoop == aLoopNum - 1) ? (curCoreCols - aLoop * td_->sortBaseA) : td_->sortBaseA;
            uint32_t colsAlign = Aligned(static_cast<uint64_t>(cols * sizeof(X_T)), ONE_BLOCK_SIZE) / sizeof(X_T);

            LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();
            uint64_t offset = blockOffsetX + sLoop * td_->sortBaseS * td_->innerDim + aLoop * td_->sortBaseA;
            CopyIn(xLocal, xGm_, offset, rows, cols, td_->innerDim - cols);
            xQue_.EnQue<X_T>(xLocal);

            ComputeXSum(cols, colsAlign, arNum);
            uint64_t ubOffset = aIdx * td_->normBlockA + aLoop * td_->sortBaseA;
            CopyResToGm(cols, colsAlign, ubOffset, arNum);
        }
    }
}

template <typename X_T, typename IDS_T, uint8_t Mode, uint32_t CAST_MODE>
__aicore__ inline void KernelSimdDynSort<X_T, IDS_T, Mode, CAST_MODE>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    Compute();
}
} // namespace UnsortedSegment

#endif