/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_PROD_SEGMENT_SORT_H
#define UNSORTED_SEGMENT_PROD_SEGMENT_SORT_H

#include "../unsorted_segment_common/arch35/unsorted_segment_base.h"

namespace UnsortedSegmentProd {
using namespace AscendC;

constexpr uint32_t SEG_SORT_TILE = 8192;
constexpr int32_t SEG_SORT_SENTINEL = 2147483647;
constexpr int32_t SEG_PREFETCH_DEPTH = 8;
constexpr uint32_t SEG_SORT_PAD = 2 * 32;
constexpr uint32_t SEG_TILE_ALIGN = 32;
constexpr uint32_t SEG_TILE_PAD = (SEG_SORT_TILE + SEG_TILE_ALIGN - 1) / SEG_TILE_ALIGN * SEG_TILE_ALIGN;
constexpr uint32_t SEG_SORT_SHIFT_OFF = 32 / sizeof(int32_t);
constexpr uint32_t SEG_WS_ALIGN = 8;
constexpr uint32_t SEG_WS_GUARD = 128;
constexpr uint32_t SEG_WS_AREA_CNT = 4;
constexpr uint32_t SEG_COL_UB_BUDGET = 176u * 1024u;
constexpr uint32_t SEG_COL_ALIGN = 32;
constexpr uint32_t SEG_COL_STRIDE_ALIGN = 8;
constexpr uint32_t SEG_FLAG_STRIDE = 16;
constexpr SortConfig segSortConfig{SortType::RADIX_SORT, false};

template <typename X_T, typename IDS_T, typename ADDR_T>
class KernelProdSegmentSort {
public:
    __aicore__ inline KernelProdSegmentSort(const UnsortedSegment::UnsortedSegmentProdSegmentSortTilingData& tiling,
                                            TPipe& pipe)
        : td_(tiling), pipe_(pipe)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SortSegmentIds(ADDR_T rowNum);
    __aicore__ inline void SortSingleCore(ADDR_T rowNum);
    __aicore__ inline void SortLocalRuns(ADDR_T rowNum, ADDR_T runCnt, ADDR_T runLen, int32_t curSel);
    __aicore__ inline void SortMergeTree(ADDR_T runCnt, ADDR_T runLen, ADDR_T roundNum, int32_t curSel);
    __aicore__ inline void ReduceSegments(ADDR_T rowNum);
    __aicore__ inline void FoldOwnedSegments(ADDR_T posBegin, ADDR_T posEnd, ADDR_T rowNum, ADDR_T colOffset,
                                             ADDR_T colWidth, ADDR_T partialStride, LocalTensor<X_T>& accUb,
                                             TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue, event_t evMte2ToV,
                                             event_t evVToMte3, event_t evMte3ToMte2, const DataCopyExtParams& copyCol,
                                             const DataCopyPadExtParams<X_T>& padCol, int32_t& splitSeg,
                                             ADDR_T& splitSegEnd);
    __aicore__ inline void CombineSplitSegment(ADDR_T rowNum, ADDR_T colOffset, ADDR_T colWidth, ADDR_T partialStride,
                                               int32_t splitSeg, ADDR_T splitSegEnd, LocalTensor<X_T>& accUb,
                                               LocalTensor<X_T>& tmpUb, event_t evMte2ToV, event_t evVToMte3,
                                               event_t evMte3ToMte2, event_t evVToMte2,
                                               const DataCopyExtParams& copyCol,
                                               const DataCopyPadExtParams<X_T>& padCol);
    __aicore__ inline void PrimeRowPrefetch(TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue, ADDR_T posStart,
                                            ADDR_T primeCnt, ADDR_T colOffset, const DataCopyExtParams& copyCol,
                                            const DataCopyPadExtParams<X_T>& padCol);
    __aicore__ inline void PrefetchNextRow(TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue, ADDR_T posStart,
                                           ADDR_T cur, ADDR_T cnt, ADDR_T ahead, ADDR_T colOffset,
                                           const DataCopyExtParams& copyCol, const DataCopyPadExtParams<X_T>& padCol);
    __aicore__ inline void MulRowsIntoAcc(LocalTensor<X_T>& accUb, TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue,
                                          ADDR_T posStart, ADDR_T posEnd, ADDR_T colOffset, ADDR_T colWidth,
                                          const DataCopyExtParams& copyCol, const DataCopyPadExtParams<X_T>& padCol);
    __aicore__ inline void LoadCols(LocalTensor<X_T>& dstUb, const GlobalTensor<X_T>& srcGm, ADDR_T offset,
                                    event_t evMte2ToV, const DataCopyExtParams& copyCol,
                                    const DataCopyPadExtParams<X_T>& padCol);
    __aicore__ inline void StoreCols(LocalTensor<X_T>& srcUb, const GlobalTensor<X_T>& dstGm, ADDR_T offset,
                                     event_t evVToMte3, const DataCopyExtParams& copyCol);
    __aicore__ inline bool AllIdsInvalid();

    const UnsortedSegment::UnsortedSegmentProdSegmentSortTilingData& td_;
    TPipe& pipe_;
    GlobalTensor<IDS_T> idsGm_;
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> outputGm_;
    GlobalTensor<int32_t> sortedIdsGm_;
    GlobalTensor<uint32_t> originPosGm_;
    GlobalTensor<int32_t> scratchIdsGm_;
    GlobalTensor<uint32_t> scratchPosGm_;
    GlobalTensor<int32_t> partialGm_;
    uint32_t blockIdx_{0};
    ADDR_T rowNum_{0};

    static constexpr uint32_t SEG_COL_BYTES = sizeof(X_T) + (SEG_PREFETCH_DEPTH + 1u) * sizeof(X_T);
    static constexpr ADDR_T SEG_COL_MAX = static_cast<ADDR_T>((SEG_COL_UB_BUDGET / SEG_COL_BYTES) / SEG_COL_ALIGN *
                                                              SEG_COL_ALIGN);
};

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output,
                                                                       GM_ADDR workspace)
{
    if (td_.innerDim == 0 || td_.blockNum == 0) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)segmentIds);
    xGm_.SetGlobalBuffer((__gm__ X_T*)x);
    outputGm_.SetGlobalBuffer((__gm__ X_T*)output);

    rowNum_ = (static_cast<ADDR_T>(td_.blockTilingSize) * (static_cast<ADDR_T>(td_.blockNum) - 1) +
               static_cast<ADDR_T>(td_.tailBlockTilingSize)) /
              static_cast<ADDR_T>(td_.innerDim);

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        blockIdx_ = static_cast<uint32_t>(td_.blockNum);
        return;
    }
    ADDR_T padM = rowNum_;
    if (rowNum_ > static_cast<ADDR_T>(SEG_SORT_TILE)) {
        ADDR_T runCnt = 1;
        while (runCnt * 2 <= static_cast<ADDR_T>(td_.blockNum)) {
            runCnt *= 2;
        }
        while ((rowNum_ + runCnt - 1) / runCnt > static_cast<ADDR_T>(SEG_SORT_TILE)) {
            runCnt *= 2;
        }
        padM = runCnt * ((rowNum_ + runCnt - 1) / runCnt);
    }
    const ADDR_T mAlign = (padM + SEG_WS_ALIGN - 1) / SEG_WS_ALIGN * SEG_WS_ALIGN + SEG_WS_GUARD;
    sortedIdsGm_.SetGlobalBuffer((__gm__ int32_t*)userWs);
    originPosGm_.SetGlobalBuffer((__gm__ uint32_t*)(userWs + mAlign * sizeof(int32_t)));
    scratchIdsGm_.SetGlobalBuffer((__gm__ int32_t*)(userWs + 2 * mAlign * sizeof(int32_t)));
    scratchPosGm_.SetGlobalBuffer(
        (__gm__ uint32_t*)(userWs + static_cast<ADDR_T>(SEG_WS_AREA_CNT - 1) * mAlign * sizeof(int32_t)));
    partialGm_.SetGlobalBuffer(
        (__gm__ int32_t*)(userWs + static_cast<ADDR_T>(SEG_WS_AREA_CNT) * mAlign * sizeof(int32_t)));
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::SortSingleCore(ADDR_T rowNum)
{
    constexpr int32_t shiftOff = static_cast<int32_t>(SEG_SORT_SHIFT_OFF);
    constexpr uint32_t tilePad = SEG_TILE_PAD;
    if (blockIdx_ == 0) {
        TBuf<TPosition::VECCALC> rawBuf, keyBuf, sortedBuf, posBuf;
        pipe_.InitBuffer(rawBuf, tilePad * sizeof(IDS_T) + SEG_SORT_PAD);
        pipe_.InitBuffer(keyBuf, tilePad * sizeof(int32_t) + SEG_SORT_PAD);
        pipe_.InitBuffer(sortedBuf, (tilePad + shiftOff) * sizeof(int32_t) + SEG_SORT_PAD);
        pipe_.InitBuffer(posBuf, tilePad * sizeof(uint32_t) + SEG_SORT_PAD);
        LocalTensor<IDS_T> rawUb = rawBuf.Get<IDS_T>();
        LocalTensor<int32_t> keyUb = keyBuf.Get<int32_t>();
        LocalTensor<uint32_t> posUb = posBuf.Get<uint32_t>();
        LocalTensor<int32_t> sortedUb = sortedBuf.Get<int32_t>()[shiftOff];
        DataCopyPadExtParams<IDS_T> padIds{false, 0, 0, 0};
        const event_t evMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t evVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        DataCopyExtParams copyIds{1, static_cast<uint32_t>(rowNum * sizeof(IDS_T)), 0, 0, 0};
        if constexpr (sizeof(IDS_T) == 8) {
            DataCopyPad(rawUb, idsGm_, copyIds, padIds);
            SetFlag<HardEvent::MTE2_V>(evMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(evMte2ToV);
            Cast(keyUb, rawUb, RoundMode::CAST_NONE, static_cast<int32_t>(rowNum));
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(keyUb, idsGm_, copyIds, padIds);
            SetFlag<HardEvent::MTE2_V>(evMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(evMte2ToV);
        }
        AscendC::Sort<int32_t, true, segSortConfig>(sortedUb, posUb, keyUb, static_cast<uint32_t>(rowNum));
        DataCopyExtParams copyOut{1, static_cast<uint32_t>(rowNum * sizeof(int32_t)), 0, 0, 0};
        SetFlag<HardEvent::V_MTE3>(evVToMte3);
        WaitFlag<HardEvent::V_MTE3>(evVToMte3);
        DataCopyPad(sortedIdsGm_, sortedUb, copyOut);
        DataCopyPad(originPosGm_, posUb, copyOut);
    }
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        sortedIdsGm_);
    AscendC::SyncAll();
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::SortLocalRuns(ADDR_T rowNum, ADDR_T runCnt,
                                                                                ADDR_T runLen, int32_t curSel)
{
    constexpr int32_t shiftOff = static_cast<int32_t>(SEG_SORT_SHIFT_OFF);
    constexpr uint32_t tilePad = SEG_TILE_PAD;
    const ADDR_T blockNum = static_cast<ADDR_T>(td_.blockNum);
    {
        TBuf<TPosition::VECCALC> rawBuf, keyBuf, sortedBuf, posBuf;
        pipe_.InitBuffer(rawBuf, tilePad * sizeof(IDS_T) + SEG_SORT_PAD);
        pipe_.InitBuffer(keyBuf, tilePad * sizeof(int32_t) + SEG_SORT_PAD);
        pipe_.InitBuffer(sortedBuf, (tilePad + shiftOff) * sizeof(int32_t) + SEG_SORT_PAD);
        pipe_.InitBuffer(posBuf, tilePad * sizeof(uint32_t) + SEG_SORT_PAD);
        LocalTensor<IDS_T> rawUb = rawBuf.Get<IDS_T>();
        LocalTensor<int32_t> keyUb = keyBuf.Get<int32_t>();
        LocalTensor<uint32_t> posUb = posBuf.Get<uint32_t>();
        LocalTensor<int32_t> sortedUb = sortedBuf.Get<int32_t>()[shiftOff];
        DataCopyPadExtParams<IDS_T> padIds{false, 0, 0, 0};
        const event_t evMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t evVToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        const event_t evSToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        const event_t evMte3ToMte2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
        for (ADDR_T r = static_cast<ADDR_T>(blockIdx_); r < runCnt; r += blockNum) {
            const ADDR_T runStart = r * runLen;
            const ADDR_T realLen = (runStart >= rowNum) ? 0 :
                                                          ((rowNum - runStart < runLen) ? (rowNum - runStart) : runLen);
            if (realLen > 0) {
                DataCopyExtParams copyIds{1, static_cast<uint32_t>(realLen * sizeof(IDS_T)), 0, 0, 0};
                if constexpr (sizeof(IDS_T) == 8) {
                    DataCopyPad(rawUb, idsGm_[runStart], copyIds, padIds);
                    SetFlag<HardEvent::MTE2_V>(evMte2ToV);
                    WaitFlag<HardEvent::MTE2_V>(evMte2ToV);
                    Cast(keyUb, rawUb, RoundMode::CAST_NONE, static_cast<int32_t>(realLen));
                    PipeBarrier<PIPE_V>();
                } else {
                    DataCopyPad(keyUb, idsGm_[runStart], copyIds, padIds);
                    SetFlag<HardEvent::MTE2_V>(evMte2ToV);
                    WaitFlag<HardEvent::MTE2_V>(evMte2ToV);
                }
                AscendC::Sort<int32_t, true, segSortConfig>(sortedUb, posUb, keyUb, static_cast<uint32_t>(realLen));
                Adds(posUb.template ReinterpretCast<int32_t>(), posUb.template ReinterpretCast<int32_t>(),
                     static_cast<int32_t>(runStart), static_cast<int32_t>(realLen));
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_S>(evVToS);
            WaitFlag<HardEvent::V_S>(evVToS);
            for (ADDR_T p = realLen; p < runLen; p++) {
                sortedUb.SetValue(p, SEG_SORT_SENTINEL);
                posUb.SetValue(p, 0u);
            }
            DataCopyExtParams copyOut{1, static_cast<uint32_t>(runLen * sizeof(int32_t)), 0, 0, 0};
            SetFlag<HardEvent::S_MTE3>(evSToMte3);
            WaitFlag<HardEvent::S_MTE3>(evSToMte3);
            if (curSel == 1) {
                DataCopyPad(sortedIdsGm_[runStart], sortedUb, copyOut);
                DataCopyPad(originPosGm_[runStart], posUb, copyOut);
            } else {
                DataCopyPad(scratchIdsGm_[runStart], sortedUb, copyOut);
                DataCopyPad(scratchPosGm_[runStart], posUb, copyOut);
            }
            SetFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
        }
    }
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        sortedIdsGm_);
    AscendC::SyncAll();
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::SortMergeTree(ADDR_T runCnt, ADDR_T runLen,
                                                                                ADDR_T roundNum, int32_t curSel)
{
    constexpr uint32_t tilePad = SEG_TILE_PAD;
    const ADDR_T blockNum = static_cast<ADDR_T>(td_.blockNum);
    TBuf<TPosition::VECCALC> outIdsBuf, outPosBuf;
    pipe_.InitBuffer(outIdsBuf, tilePad * sizeof(int32_t) + SEG_SORT_PAD);
    pipe_.InitBuffer(outPosBuf, tilePad * sizeof(uint32_t) + SEG_SORT_PAD);
    LocalTensor<int32_t> outIdsUb = outIdsBuf.Get<int32_t>();
    LocalTensor<uint32_t> outPosUb = outPosBuf.Get<uint32_t>();
    const event_t evSToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
    const event_t evMte3ToS = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
    for (ADDR_T round = 0; round < roundNum; round++) {
        const ADDR_T curRunLen = runLen << round;
        const ADDR_T pairLen = curRunLen * 2;
        GlobalTensor<int32_t> curIds = (curSel == 1) ? sortedIdsGm_ : scratchIdsGm_;
        GlobalTensor<uint32_t> curPos = (curSel == 1) ? originPosGm_ : scratchPosGm_;
        for (ADDR_T s = static_cast<ADDR_T>(blockIdx_); s < runCnt; s += blockNum) {
            const ADDR_T outBase = s * runLen;
            const ADDR_T pairBase = (outBase / pairLen) * pairLen;
            const ADDR_T leftBase = pairBase;
            const ADDR_T rightBase = pairBase + curRunLen;
            const ADDR_T localLo = outBase - pairBase;
            ADDR_T lo = (localLo > curRunLen) ? (localLo - curRunLen) : 0;
            ADDR_T hi = (localLo < curRunLen) ? localLo : curRunLen;
            while (lo < hi) {
                const ADDR_T mid = (lo + hi + 1) / 2;
                const int32_t leftVal = curIds.GetValue(leftBase + mid - 1);
                const ADDR_T rj = localLo - mid;
                const int32_t rightVal = (rj >= curRunLen) ? SEG_SORT_SENTINEL : curIds.GetValue(rightBase + rj);
                if (leftVal <= rightVal) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            ADDR_T li = leftBase + lo;
            ADDR_T ri = rightBase + (localLo - lo);
            const ADDR_T lEnd = leftBase + curRunLen;
            const ADDR_T rEnd = rightBase + curRunLen;
            for (ADDR_T o = 0; o < runLen; o++) {
                bool takeLeft;
                if (li >= lEnd) {
                    takeLeft = false;
                } else if (ri >= rEnd) {
                    takeLeft = true;
                } else {
                    takeLeft = (curIds.GetValue(li) <= curIds.GetValue(ri));
                }
                if (takeLeft) {
                    outIdsUb.SetValue(o, curIds.GetValue(li));
                    outPosUb.SetValue(o, curPos.GetValue(li));
                    li++;
                } else {
                    outIdsUb.SetValue(o, curIds.GetValue(ri));
                    outPosUb.SetValue(o, curPos.GetValue(ri));
                    ri++;
                }
            }
            DataCopyExtParams copySlice{1, static_cast<uint32_t>(runLen * sizeof(int32_t)), 0, 0, 0};
            SetFlag<HardEvent::S_MTE3>(evSToMte3);
            WaitFlag<HardEvent::S_MTE3>(evSToMte3);
            if (curSel == 1) {
                DataCopyPad(scratchIdsGm_[outBase], outIdsUb, copySlice);
                DataCopyPad(scratchPosGm_[outBase], outPosUb, copySlice);
            } else {
                DataCopyPad(sortedIdsGm_[outBase], outIdsUb, copySlice);
                DataCopyPad(originPosGm_[outBase], outPosUb, copySlice);
            }
            SetFlag<HardEvent::MTE3_S>(evMte3ToS);
            WaitFlag<HardEvent::MTE3_S>(evMte3ToS);
        }
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(sortedIdsGm_);
        curSel ^= 1;
        AscendC::SyncAll();
    }
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::SortSegmentIds(ADDR_T rowNum)
{
    if (rowNum <= static_cast<ADDR_T>(SEG_SORT_TILE)) {
        SortSingleCore(rowNum);
        return;
    }
    const ADDR_T blockNum = static_cast<ADDR_T>(td_.blockNum);
    ADDR_T runCnt = 1;
    while (runCnt * 2 <= blockNum) {
        runCnt *= 2;
    }
    while ((rowNum + runCnt - 1) / runCnt > static_cast<ADDR_T>(SEG_SORT_TILE)) {
        runCnt *= 2;
    }
    const ADDR_T runLen = (rowNum + runCnt - 1) / runCnt;
    ADDR_T roundNum = 0;
    {
        ADDR_T q = runCnt;
        while (q > 1) {
            q /= 2;
            roundNum++;
        }
    }
    const int32_t startSel = ((roundNum & 1) == 0) ? 1 : 0;
    SortLocalRuns(rowNum, runCnt, runLen, startSel);
    SortMergeTree(runCnt, runLen, roundNum, startSel);
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::PrimeRowPrefetch(
    TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue, ADDR_T posStart, ADDR_T primeCnt, ADDR_T colOffset,
    const DataCopyExtParams& copyCol, const DataCopyPadExtParams<X_T>& padCol)
{
    for (ADDR_T p = 0; p < primeCnt; p++) {
        uint32_t row = originPosGm_.GetValue(posStart + p);
        LocalTensor<X_T> buf = rowQue.template AllocTensor<X_T>();
        DataCopyPad(buf, xGm_[static_cast<ADDR_T>(row) * static_cast<ADDR_T>(td_.innerDim) + colOffset], copyCol,
                    padCol);
        rowQue.EnQue(buf);
    }
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::PrefetchNextRow(
    TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue, ADDR_T posStart, ADDR_T cur, ADDR_T cnt, ADDR_T ahead,
    ADDR_T colOffset, const DataCopyExtParams& copyCol, const DataCopyPadExtParams<X_T>& padCol)
{
    if (cur + ahead < cnt) {
        uint32_t row = originPosGm_.GetValue(posStart + cur + ahead);
        LocalTensor<X_T> buf = rowQue.template AllocTensor<X_T>();
        DataCopyPad(buf, xGm_[static_cast<ADDR_T>(row) * static_cast<ADDR_T>(td_.innerDim) + colOffset], copyCol,
                    padCol);
        rowQue.EnQue(buf);
    }
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::MulRowsIntoAcc(
    LocalTensor<X_T>& accUb, TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue, ADDR_T posStart, ADDR_T posEnd,
    ADDR_T colOffset, ADDR_T colWidth, const DataCopyExtParams& copyCol, const DataCopyPadExtParams<X_T>& padCol)
{
    if (posStart >= posEnd) {
        return;
    }
    constexpr ADDR_T AHEAD = SEG_PREFETCH_DEPTH - 1;
    const ADDR_T cnt = posEnd - posStart;
    const ADDR_T primeCnt = (cnt < AHEAD) ? cnt : AHEAD;
    PrimeRowPrefetch(rowQue, posStart, primeCnt, colOffset, copyCol, padCol);
    for (ADDR_T j = 0; j < cnt; j++) {
        LocalTensor<X_T> rowUb = rowQue.template DeQue<X_T>();
        PrefetchNextRow(rowQue, posStart, j, cnt, AHEAD, colOffset, copyCol, padCol);
        Mul(accUb, accUb, rowUb, static_cast<int32_t>(colWidth));
        rowQue.FreeTensor(rowUb);
    }
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::LoadCols(LocalTensor<X_T>& dstUb,
                                                                           const GlobalTensor<X_T>& srcGm,
                                                                           ADDR_T offset, event_t evMte2ToV,
                                                                           const DataCopyExtParams& copyCol,
                                                                           const DataCopyPadExtParams<X_T>& padCol)
{
    DataCopyPad(dstUb, srcGm[offset], copyCol, padCol);
    SetFlag<HardEvent::MTE2_V>(evMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(evMte2ToV);
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::StoreCols(LocalTensor<X_T>& srcUb,
                                                                            const GlobalTensor<X_T>& dstGm,
                                                                            ADDR_T offset, event_t evVToMte3,
                                                                            const DataCopyExtParams& copyCol)
{
    SetFlag<HardEvent::V_MTE3>(evVToMte3);
    WaitFlag<HardEvent::V_MTE3>(evVToMte3);
    DataCopyPad(dstGm[offset], srcUb, copyCol);
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::FoldOwnedSegments(
    ADDR_T posBegin, ADDR_T posEnd, ADDR_T rowNum, ADDR_T colOffset, ADDR_T colWidth, ADDR_T partialStride,
    LocalTensor<X_T>& accUb, TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH>& rowQue, event_t evMte2ToV, event_t evVToMte3,
    event_t evMte3ToMte2, const DataCopyExtParams& copyCol, const DataCopyPadExtParams<X_T>& padCol, int32_t& splitSeg,
    ADDR_T& splitSegEnd)
{
    const ADDR_T innerDim = static_cast<ADDR_T>(td_.innerDim);
    const ADDR_T outputOuterDim = static_cast<ADDR_T>(td_.outputOuterDim);
    ADDR_T k = posBegin;
    if (k > 0 && sortedIdsGm_.GetValue(k) == sortedIdsGm_.GetValue(k - 1)) {
        int32_t seg = sortedIdsGm_.GetValue(k);
        ADDR_T segStart = k;
        while (k < posEnd && sortedIdsGm_.GetValue(k) == seg) {
            k++;
        }
        if (seg >= 0 && static_cast<ADDR_T>(seg) < outputOuterDim) {
            uint32_t firstRow = originPosGm_.GetValue(segStart);
            LoadCols(accUb, xGm_, static_cast<ADDR_T>(firstRow) * innerDim + colOffset, evMte2ToV, copyCol, padCol);
            MulRowsIntoAcc(accUb, rowQue, segStart + 1, k, colOffset, colWidth, copyCol, padCol);
            SetFlag<HardEvent::V_MTE3>(evVToMte3);
            WaitFlag<HardEvent::V_MTE3>(evVToMte3);
            DataCopyPad(partialGm_[static_cast<ADDR_T>(blockIdx_) * partialStride].template ReinterpretCast<X_T>(),
                        accUb, copyCol);
            SetFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
        }
    }
    while (k < posEnd && k < rowNum) {
        int32_t seg = sortedIdsGm_.GetValue(k);
        ADDR_T segStart = k;
        while (k < rowNum && sortedIdsGm_.GetValue(k) == seg) {
            k++;
        }
        ADDR_T segEnd = k;
        if (seg < 0 || static_cast<ADDR_T>(seg) >= outputOuterDim) {
            if (segEnd > posEnd) {
                break;
            }
            continue;
        }
        const ADDR_T outOffset = static_cast<ADDR_T>(seg) * innerDim + colOffset;
        if (segEnd <= posEnd) {
            LoadCols(accUb, outputGm_, outOffset, evMte2ToV, copyCol, padCol);
            MulRowsIntoAcc(accUb, rowQue, segStart, segEnd, colOffset, colWidth, copyCol, padCol);
            StoreCols(accUb, outputGm_, outOffset, evVToMte3, copyCol);
            SetFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
        } else {
            uint32_t firstRow = originPosGm_.GetValue(segStart);
            LoadCols(accUb, xGm_, static_cast<ADDR_T>(firstRow) * innerDim + colOffset, evMte2ToV, copyCol, padCol);
            MulRowsIntoAcc(accUb, rowQue, segStart + 1, posEnd, colOffset, colWidth, copyCol, padCol);
            splitSeg = seg;
            splitSegEnd = segEnd;
            break;
        }
    }
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::CombineSplitSegment(
    ADDR_T rowNum, ADDR_T colOffset, ADDR_T colWidth, ADDR_T partialStride, int32_t splitSeg, ADDR_T splitSegEnd,
    LocalTensor<X_T>& accUb, LocalTensor<X_T>& tmpUb, event_t evMte2ToV, event_t evVToMte3, event_t evMte3ToMte2,
    event_t evVToMte2, const DataCopyExtParams& copyCol, const DataCopyPadExtParams<X_T>& padCol)
{
    if (splitSeg < 0) {
        return;
    }
    const ADDR_T innerDim = static_cast<ADDR_T>(td_.innerDim);
    const ADDR_T outOffset = static_cast<ADDR_T>(splitSeg) * innerDim + colOffset;
    const ADDR_T blockNum = static_cast<ADDR_T>(td_.blockNum);
    for (ADDR_T c = static_cast<ADDR_T>(blockIdx_) + 1; c < blockNum && c * rowNum / blockNum < splitSegEnd; c++) {
        LoadCols(tmpUb, partialGm_.template ReinterpretCast<X_T>(), c * partialStride, evMte2ToV, copyCol, padCol);
        Mul(accUb, accUb, tmpUb, static_cast<int32_t>(colWidth));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(evVToMte2);
        WaitFlag<HardEvent::V_MTE2>(evVToMte2);
    }
    LoadCols(tmpUb, outputGm_, outOffset, evMte2ToV, copyCol, padCol);
    Mul(accUb, accUb, tmpUb, static_cast<int32_t>(colWidth));
    PipeBarrier<PIPE_V>();
    StoreCols(accUb, outputGm_, outOffset, evVToMte3, copyCol);
    SetFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::ReduceSegments(ADDR_T rowNum)
{
    const ADDR_T blockNum = static_cast<ADDR_T>(td_.blockNum);
    const ADDR_T innerDim = static_cast<ADDR_T>(td_.innerDim);
    const ADDR_T posBegin = static_cast<ADDR_T>(blockIdx_) * rowNum / blockNum;
    const ADDR_T posEnd = (static_cast<ADDR_T>(blockIdx_) + 1) * rowNum / blockNum;
    const bool ownsRange = posBegin < posEnd;

    const ADDR_T colChunk = (innerDim < SEG_COL_MAX) ? innerDim : SEG_COL_MAX;
    const uint32_t colPad = (static_cast<uint32_t>(colChunk) + SEG_COL_ALIGN - 1) / SEG_COL_ALIGN * SEG_COL_ALIGN;
    const ADDR_T partialStride = (colChunk + SEG_COL_STRIDE_ALIGN - 1) / SEG_COL_STRIDE_ALIGN * SEG_COL_STRIDE_ALIGN;

    TBuf<TPosition::VECCALC> accBuf, tmpBuf;
    TQue<QuePosition::VECIN, SEG_PREFETCH_DEPTH> rowQue;
    pipe_.InitBuffer(accBuf, colPad * sizeof(X_T));
    pipe_.InitBuffer(tmpBuf, colPad * sizeof(X_T));
    pipe_.InitBuffer(rowQue, SEG_PREFETCH_DEPTH, colPad * sizeof(X_T));
    LocalTensor<X_T> accUb = accBuf.Get<X_T>();
    LocalTensor<X_T> tmpUb = tmpBuf.Get<X_T>();
    const event_t evMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
    const event_t evVToMte3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    const event_t evMte3ToMte2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
    const event_t evVToMte2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
    DataCopyPadExtParams<X_T> padCol{false, 0, 0, static_cast<X_T>(0)};

    for (ADDR_T c0 = 0; c0 < innerDim; c0 += colChunk) {
        const ADDR_T colWidth = (innerDim - c0 < colChunk) ? (innerDim - c0) : colChunk;
        DataCopyExtParams copyCol{1, static_cast<uint32_t>(colWidth * sizeof(X_T)), 0, 0, 0};
        int32_t splitSeg = -1;
        ADDR_T splitSegEnd = 0;
        if (ownsRange) {
            FoldOwnedSegments(posBegin, posEnd, rowNum, c0, colWidth, partialStride, accUb, rowQue, evMte2ToV,
                              evVToMte3, evMte3ToMte2, copyCol, padCol, splitSeg, splitSegEnd);
        }
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(partialGm_);
        AscendC::SyncAll();
        CombineSplitSegment(rowNum, c0, colWidth, partialStride, splitSeg, splitSegEnd, accUb, tmpUb, evMte2ToV,
                            evVToMte3, evMte3ToMte2, evVToMte2, copyCol, padCol);
        if (c0 + colChunk < innerDim) {
            AscendC::SyncAll();
        }
    }
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline bool KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::AllIdsInvalid()
{
    const int64_t outputOuterDim = static_cast<int64_t>(td_.outputOuterDim);
    const ADDR_T blockNum = static_cast<ADDR_T>(td_.blockNum);
    const ADDR_T shard = (rowNum_ + blockNum - 1) / blockNum;
    const ADDR_T lo = static_cast<ADDR_T>(blockIdx_) * shard;
    ADDR_T hi = lo + shard;
    if (hi > rowNum_) {
        hi = rowNum_;
    }
    bool allInvalid = true;
    if (lo < hi) {
        constexpr uint32_t tile = SEG_TILE_PAD;
        TBuf<TPosition::VECCALC> idsBuf, reduceBuf;
        pipe_.InitBuffer(idsBuf, tile * sizeof(IDS_T) + SEG_SORT_PAD);
        pipe_.InitBuffer(reduceBuf, UnsortedSegment::AllIdsInvalidReduceBufBytes<IDS_T>(tile));
        LocalTensor<IDS_T> idsUb = idsBuf.Get<IDS_T>();
        LocalTensor<IDS_T> reduceDst = reduceBuf.Get<IDS_T>();
        const event_t evMte2ToV = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        DataCopyPadExtParams<IDS_T> padIds{false, 0, 0, 0};
        for (ADDR_T p = lo; p < hi; p += tile) {
            const ADDR_T len = (hi - p < tile) ? (hi - p) : tile;
            DataCopyExtParams copyIds{1, static_cast<uint32_t>(len * sizeof(IDS_T)), 0, 0, 0};
            DataCopyPad(idsUb, idsGm_[p], copyIds, padIds);
            SetFlag<HardEvent::MTE2_V>(evMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(evMte2ToV);
            if (!UnsortedSegment::AllIdsInvalidVectorized(idsUb, static_cast<uint32_t>(len), outputOuterDim,
                                                          reduceDst)) {
                allInvalid = false;
                break;
            }
        }
    }
    partialGm_.SetValue(static_cast<uint32_t>(blockIdx_) * SEG_FLAG_STRIDE, allInvalid ? 1 : 0);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        partialGm_);
    AscendC::SyncAll();
    bool globalAllInvalid = true;
    for (ADDR_T b = 0; b < blockNum; b++) {
        if (partialGm_.GetValue(static_cast<uint32_t>(b) * SEG_FLAG_STRIDE) == 0) {
            globalAllInvalid = false;
            break;
        }
    }
    return globalAllInvalid;
}

template <typename X_T, typename IDS_T, typename ADDR_T>
__aicore__ inline void KernelProdSegmentSort<X_T, IDS_T, ADDR_T>::Process()
{
    if constexpr (sizeof(X_T) == 4) {
        if (td_.innerDim == 0 || td_.outputOuterDim == 0 || rowNum_ == 0 ||
            blockIdx_ >= static_cast<uint32_t>(td_.blockNum)) {
            return;
        }
        if (AllIdsInvalid()) {
            return;
        }
        pipe_.Reset();
        SortSegmentIds(rowNum_);
        pipe_.Reset();
        ReduceSegments(rowNum_);
    }
}

} // namespace UnsortedSegmentProd
#endif
