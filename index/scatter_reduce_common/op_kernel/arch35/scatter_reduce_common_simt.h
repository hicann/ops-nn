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
 * \file scatter_reduce_common_simt.h
 * \brief Dispatcher for the non-nd scatter-reduce ops (Mul/Div/Max/Min). For each index entry m:
 *        var[indices[m]] (slice of sliceSize elems) = reduce(var[...], updates[m]). This file holds the
 *        ScatterReduceSimt class (Init + Process dispatch). All compute paths live in sibling files,
 *        dispatched from Process() — there is no SIMT-atomic path (a single scalar/sort cover every case):
 *          - scatter_reduce_common_sort.h   : sort + count-balanced multi-core split; sliceSize>=2 column
 *                                             tiling, sliceSize==1 large-M scalar row-fold (all modes)
 *          - scatter_reduce_common_scalar.h : single-core scalar reduction for small sliceSize==1
 */
#ifndef SCATTER_REDUCE_COMMON_SIMT_H
#define SCATTER_REDUCE_COMMON_SIMT_H

#include "scatter_reduce_common_struct.h"
#include "kernel_operator.h"

namespace ScatterReduceCommon {
using namespace AscendC;

// sliceSize==1 scalar-path cap: M + varFirstDim must fit the single-core UB staging (idx+upd+var+acc). Above
// this, the sort + scalar row-fold path takes over (no UB limit). 8192 covers every realistic 1-D case (test
// set maxes at M=4096, varFirstDim=1000) with margin and stays UB-safe even for int64 indices.
constexpr uint32_t SCALAR_SLICE1_CAP = 8192;
constexpr SortConfig scatterSortConfig{SortType::RADIX_SORT, false};  // radix sort of int32 index keys

// Subword (int8/uint8) <-> int32 conversion via MicroAPI. High-level Cast<int32,int8> is broken on
// dav_3510 (produces garbage for signed int8); the established ops-nn pattern is UNPACK4_B8 (widen,
// sign-correct by the RegTensor<T> type) and PACK4_B32 (narrow, low-byte/wrapping). Mirrors
// scatter_add_common.h CastToInt32 / CastToOrigin.
template <typename T>
__aicore__ inline void SubwordWidenToI32(const LocalTensor<int32_t>& dst, const LocalTensor<T>& src, uint32_t n)
{
    constexpr uint32_t VL_B32 = GetVecLen() / sizeof(uint32_t);
    __local_mem__ T* srcAddr = (__local_mem__ T*)src.GetPhyAddr();
    __local_mem__ int32_t* dstAddr = (__local_mem__ int32_t*)dst.GetPhyAddr();
    uint16_t loopTimes = (n + VL_B32 - 1) / VL_B32;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = n;
        for (uint16_t i = 0; i < loopTimes; i++) {
            auto dstReg = MicroAPI::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
            auto srcReg = MicroAPI::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
            preg = MicroAPI::UpdateMask<int32_t>(sregMask);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr, srcReg);
            MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_NORM>(
                dstAddr, (MicroAPI::RegTensor<int32_t>&)srcValue, dstReg, preg);
        }
    }
}

template <typename T>
__aicore__ inline void SubwordNarrowFromI32(const LocalTensor<T>& dst, const LocalTensor<int32_t>& src, uint32_t n)
{
    constexpr uint32_t VL_B32 = GetVecLen() / sizeof(uint32_t);
    __local_mem__ int32_t* srcAddr = (__local_mem__ int32_t*)src.GetPhyAddr();
    __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
    uint16_t loopTimes = (n + VL_B32 - 1) / VL_B32;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> srcValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = n;
        for (uint16_t i = 0; i < loopTimes; i++) {
            auto srcReg = MicroAPI::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
            auto dstReg = MicroAPI::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
            preg = MicroAPI::UpdateMask<int32_t>(sregMask);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(srcValue, srcAddr, srcReg);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK4_B32>(
                dstAddr, (MicroAPI::RegTensor<T>&)srcValue, dstReg, preg);
        }
    }
}

// sliceSize==1 fast path lives in its own file (crisp, distinct class -- single-core scalar reduction). Pulled
// in here, after the subword widen/narrow helpers it reuses, so the dispatcher can route to it.
#include "scatter_reduce_common_scalar.h"

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
class ScatterReduceSimt {
public:
    __aicore__ inline ScatterReduceSimt(const ScatterReduceSimtTilingData& tilingData, TPipe& pipe)
        : tiling_(tilingData), pipe_(pipe)
    {}
    // workspace defaults to nullptr: only the MUL sort path consumes it; DIV/MAX/MIN call Init(var,idx,upd).
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace = nullptr);
    __aicore__ inline void Process();
    // ACC: float for fp16, native for fp32/int32, int32 for int8/uint8 (forced by hardware -- vmul has no
    // 8-bit form). Used by the sort-based product and the cross-core partial-combine. Declared before the
    // sort-phase helper declarations below, which take LocalTensor<AccT>& parameters.
    using AccT = typename Conditional<sizeof(PARAMS_T) == 4, PARAMS_T,
        typename Conditional<AscendC::IsSameType<PARAMS_T, half>::value, float, int32_t>::type>::type;
    // sort-based POSITION-split reducer for MUL/MAX/MIN (defined in scatter_reduce_common_sort.h): core 0
    // globally sorts indices into workspace, SyncAll, then every core reduces a count-balanced range of sorted
    // segments (each segment = one var row) with sliceSize column-tiling and a cross-core partial combine
    // (MUL: product; MAX/MIN: max/min -- all associative+commutative, so splittable & combinable). No atomics.
    __aicore__ inline void SortReduceProcess(ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim);
    // SortReduceProcess per-chunk sub-phases (pure code-motion split; same execution order + sync pairing):
    //   SortPhase3aFold     : in-range fold for columns [c0,c0+chunkW). Whole segments -> var; a segment that
    //                         spans this core's right boundary -> in-range partial to partialGm_[core] and the
    //                         owner-split (row,end) is returned for 3b. Runs before SyncAll #2.
    //   SortPhase3bCombine  : owner folds the covering cores' partials (after SyncAll #2) + var, then stores.
    __aicore__ inline void SortPhase3aFold(ADDR_T posBegin, ADDR_T posEnd, ADDR_T M, ADDR_T sliceSize,
        ADDR_T varFirstDim, ADDR_T c0, ADDR_T chunkW, ADDR_T sAlign, LocalTensor<AccT>& accUb,
        LocalTensor<PARAMS_T>& varUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
        TQue<QuePosition::VECIN, 8>& updQue, event_t evMV0, event_t evVM3, event_t evM3M2,
        const DataCopyExtParams& cp, const DataCopyExtParams& cpAcc,
        const DataCopyPadExtParams<PARAMS_T>& pad, int32_t& ownerSplitRow, ADDR_T& ownerSplitEnd);
    __aicore__ inline void SortPhase3bCombine(ADDR_T M, ADDR_T sliceSize, ADDR_T c0, ADDR_T chunkW,
        ADDR_T sAlign, int32_t ownerSplitRow, ADDR_T ownerSplitEnd, LocalTensor<AccT>& accUb,
        LocalTensor<PARAMS_T>& varUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp, event_t evMV0,
        event_t evVM3, event_t evM3M2, event_t evVM2, const DataCopyExtParams& cp,
        const DataCopyExtParams& cpAcc, const DataCopyPadExtParams<PARAMS_T>& pad,
        const DataCopyPadExtParams<AccT>& padAcc);
    // sort-based ROW-split reducer for DIV (defined in scatter_reduce_common_sort.h): same global sort, but
    // every core owns the segments whose START is in its count-balanced position range and divides those whole
    // rows IN PLACE across the full slice -- a divide chain cannot be split across cores (the divisor product
    // overflows), so each row is owned by one core, no combine. Each core walks only O(M/P) positions (the
    // column-split's O(M) walk blew up at large M). Shares Phase 1 via SortIndices.
    __aicore__ inline void SortDivRowProcess(ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim);
    // sliceSize==1 for M (or varFirstDim) beyond the scalar UB cap, ALL modes (MUL/DIV/MAX/MIN): reuse Phase-1
    // SortIndices, then fold each var row's updates with a SCALAR op (no width-1 vector, so sort CAN do
    // sliceSize==1). Row-ownership split (each core owns segments STARTING in its range, whole, no combine, no
    // barrier); DataCopyPad coherent write. AccT intermediate -> correct (no fp16 overflow) for every dtype.
    __aicore__ inline void SortScalarSliceSize1(ADDR_T M, ADDR_T varFirstDim);
    // Phase 1 of both sort reducers: core 0 sorts indices into sortedIdx/originPos workspace, then SyncAll.
    __aicore__ inline void SortIndices(ADDR_T M);
    // SortIndices sub-phases (pure code-motion split of SortIndices; same execution order, same sync pairing):
    //   SortIndicesSmallM  : M<=SORT_TILE fast path -- core 0 single AscendC::Sort + collective DCCI + SyncAll.
    //   SortLocalRuns      : large-M phase 2 -- each core sorts its runLen0-wide chunks (pad tail with SENT),
    //                        writes them into the curSel buffer, then the collective DCCI + SyncAll.
    //   SortMergeTree      : large-M merge tree -- each round each core emits its merge-path slice; flips curSel.
    __aicore__ inline void SortIndicesSmallM(ADDR_T M);
    __aicore__ inline void SortLocalRuns(ADDR_T M, ADDR_T P2, ADDR_T runLen0, int curSel);
    __aicore__ inline void SortMergeTree(ADDR_T M, ADDR_T P2, ADDR_T runLen0, ADDR_T nRounds, int curSel);
    // CHUNK_MAX = widest phase-3 column chunk that fits its UB budget; sliceSize is tiled into CHUNK_MAX columns.
    // (Phase-1 sort UB is bounded by SORT_TILE, not M -- the tiled merge-sort means there is no M gate at all.)
    static constexpr uint32_t P3_COL_BYTES = 2u * sizeof(AccT) + (8u + 1u) * sizeof(PARAMS_T);
    static constexpr ADDR_T CHUNK_MAX = static_cast<ADDR_T>(((176u * 1024u) / P3_COL_BYTES) / 32u * 32u);
    // Shared deep-prefetch fold skeleton for MUL and MAX/MIN (defined in scatter_reduce_common_sort.h): primes
    // the scattered loads, keeps AHEAD in flight, and applies the Mode-selected fold (MUL product / MAX/MIN
    // max-min) per row with each path's exact PipeBarrier rhythm. MulRangeIntoAcc / MaxMinRangeIntoAcc are thin
    // wrappers over it; `tmp` is read only by the MAX/MIN int8 sign-extend (may be empty on the MUL path).
    __aicore__ inline void FoldRangeIntoAcc(LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb,
        LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, 8>& updQue, ADDR_T startJ, ADDR_T endJ,
        ADDR_T rowStride, ADDR_T colOff, ADDR_T width, const DataCopyExtParams& cp,
        const DataCopyPadExtParams<PARAMS_T>& pad);
    // Deep-prefetch LOAD-MANAGEMENT shared by the fold (FoldRangeIntoAcc) and the divide (DivRangeIntoVar)
    // skeletons (defined in scatter_reduce_common_sort.h). These do ONLY the scattered-load queue management
    // (originPos GetValue -> AllocTensor -> DataCopyPad -> EnQue); the DeQue + per-row consume + FreeTensor stay
    // in each caller so each path keeps its own barrier rhythm untouched.
    //   PrimePrefetch  : prime the first `primed` (= min(cnt, AHEAD)) scattered loads before the consume loop.
    //   PrefetchAhead  : inside the consume loop, prefetch the row AHEAD ahead of `j` while it is still in range.
    __aicore__ inline void PrimePrefetch(TQue<QuePosition::VECIN, 8>& updQue, ADDR_T startJ, ADDR_T primed,
        ADDR_T rowStride, ADDR_T colOff, const DataCopyExtParams& cp,
        const DataCopyPadExtParams<PARAMS_T>& pad);
    __aicore__ inline void PrefetchAhead(TQue<QuePosition::VECIN, 8>& updQue, ADDR_T startJ, ADDR_T j,
        ADDR_T cnt, ADDR_T AHEAD, ADDR_T rowStride, ADDR_T colOff, const DataCopyExtParams& cp,
        const DataCopyPadExtParams<PARAMS_T>& pad);
    // Deep-prefetch multiply of updates[originPos[startJ..endJ)] columns [colOff,colOff+width) INTO accUb
    // (already initialised by caller). rowStride is the full row width (updates rows are rowStride apart).
    __aicore__ inline void MulRangeIntoAcc(LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb,
        LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, 8>& updQue, ADDR_T startJ, ADDR_T endJ,
        ADDR_T rowStride, ADDR_T colOff, ADDR_T width, const DataCopyExtParams& cp,
        const DataCopyPadExtParams<PARAMS_T>& pad);
    // DivRangeIntoVar divides accUb (= var[row], in the ACC type) by the segment's updates in place. INTEGER
    // uses the exact int32 vector Div with 0-divisors replaced by 1 (so /0 keeps var, matching the reference,
    // and integer division stays order-independent); FLOAT divides in fp32 (/0 -> inf). updAcc holds the
    // widened update, tmp is the int 0->1 scratch.
    __aicore__ inline void DivRangeIntoVar(LocalTensor<AccT>& accUb, LocalTensor<AccT>& updAcc,
        LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, 8>& updQue,
        ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff, ADDR_T width,
        const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad);
    // MaxMinRangeIntoAcc folds the segment's updates into accUb (already seeded) by Mode's max/min -- the
    // MAX/MIN counterpart of MulRangeIntoAcc for the position-split reducer. Widens like MUL/DIV (int8/uint8
    // -> int32 with int8 sign-extended via tmp so the compare is signed-correct; fp16 -> float). rowAccUb is
    // the per-update widen scratch; tmp the int8 sign-extend scratch. max/min combine across cores, so this is
    // used for both whole segments and per-core partials.
    __aicore__ inline void MaxMinRangeIntoAcc(LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb,
        LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, 8>& updQue, ADDR_T startJ, ADDR_T endJ,
        ADDR_T rowStride, ADDR_T colOff, ADDR_T width,
        const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad);
    // load srcGm[off..] (PARAMS_T) into accUb, widening subword/fp16 to the ACC type.
    __aicore__ inline void LoadWiden(LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb,
        const GlobalTensor<PARAMS_T>& srcGm, ADDR_T off, ADDR_T sliceSize, event_t evMV,
        const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad);
    // narrow accUb to PARAMS_T and store to dstGm[off..].
    __aicore__ inline void NarrowStore(LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb,
        const GlobalTensor<PARAMS_T>& dstGm, ADDR_T off, ADDR_T sliceSize, event_t evVM3,
        const DataCopyExtParams& cp);

private:
    const ScatterReduceSimtTilingData& tiling_;
    TPipe& pipe_;
    GlobalTensor<INDICES_T> idxGm_;
    GlobalTensor<PARAMS_T> xGm_;       // updates
    GlobalTensor<PARAMS_T> outputGm_;  // var (in-place reduce target)
    GlobalTensor<int32_t> sortedIdxGm_;   // workspace: sorted index values [M]
    GlobalTensor<uint32_t> originPosGm_;  // workspace: original update position per sorted slot [M]
    GlobalTensor<int32_t> scratchKeysGm_;   // workspace: K pre-merge sorted index runs [M] (tiled merge-sort)
    GlobalTensor<uint32_t> scratchPosGm_;   // workspace: K pre-merge sorted position runs [M] (tiled merge-sort)
    // split-combine scratch: when one var row's segment spans several cores, each covering core writes its
    // in-range partial product to partialGm_[core] (one CHUNK-wide slot); the row's owner reads back the
    // consecutive covering cores' partials and combines (4 bytes per elem, reinterpreted to the ACC type).
    GlobalTensor<int32_t> partialGm_;     // workspace: per-core partial product [blockNum * chunk]
    uint32_t blockIdx_{0};
    ADDR_T xBlockOffSet_{0};
    ADDR_T currBlockTilingSize_{0};
    ADDR_T totalIdxn_{0};
};


template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    if (tiling_.sliceSize == 0) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    xBlockOffSet_ = static_cast<ADDR_T>(tiling_.blockTilingSize) * blockIdx_;
    currBlockTilingSize_ = (blockIdx_ == tiling_.blockNum - 1) ? static_cast<ADDR_T>(tiling_.tailBlockTilingSize)
                                                               : static_cast<ADDR_T>(tiling_.blockTilingSize);
    idxGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    xGm_.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm_.SetGlobalBuffer((__gm__ PARAMS_T*)var);
    // total index entries across all cores (front cores carry blockTilingSize, last carries the tail)
    totalIdxn_ = (static_cast<ADDR_T>(tiling_.blockTilingSize) * (static_cast<ADDR_T>(tiling_.blockNum) - 1) +
                  static_cast<ADDR_T>(tiling_.tailBlockTilingSize)) / static_cast<ADDR_T>(tiling_.sliceSize);
    // user workspace scratch (all sort paths -- MUL/DIV/MAX/MIN): sortedIdx[M] (int32) then originPos[M]
    // (uint32), each 32B-aligned, then per-core partials (MUL only; DIV/MAX/MIN column-split need none). The
    // host tiling sizes the workspace unconditionally, so MAX/MIN reuse the same region as DIV.
    if constexpr (Mode == MODE_MUL || Mode == MODE_DIV || Mode == MODE_MAX || Mode == MODE_MIN) {
        GM_ADDR userWs = GetUserWorkspace(workspace);
        if (userWs == nullptr) {
            // workspace not provisioned (no kernel entry hits this today -- host sizes it unconditionally):
            // the sort-scratch GlobalTensors below stay unbound. Force blockIdx_ past blockNum so Process's
            // `blockIdx_ >= blockNum` guard returns BEFORE any sort path dereferences them, instead of
            // deferring the null-deref into SortReduceProcess/SortDivRowProcess.
            blockIdx_ = static_cast<uint32_t>(tiling_.blockNum);
            return;
        }
        // +128 margin: the parallel merge-sort pads M up to padM = P2*ceil(M/P2) (< M + P2 <= M + coreNum),
        // so each region must hold padM, not just M -- otherwise the padded tail overflows into the next region.
        const ADDR_T mAlign = (totalIdxn_ + 7) / 8 * 8 + 128;  // 32B align (8 x 4B) + padding headroom
        // layout: sortedIdx[M] | originPos[M] | scratchKeys[M] | scratchPos[M] | per-core partials.
        // scratchKeys/scratchPos hold the K pre-merge sorted runs that the tiled merge-sort folds into
        // sortedIdx/originPos (they are free during phase 3, where only partials are used).
        sortedIdxGm_.SetGlobalBuffer((__gm__ int32_t*)userWs);
        originPosGm_.SetGlobalBuffer((__gm__ uint32_t*)(userWs + mAlign * sizeof(int32_t)));
        scratchKeysGm_.SetGlobalBuffer((__gm__ int32_t*)(userWs + 2 * mAlign * sizeof(int32_t)));
        scratchPosGm_.SetGlobalBuffer((__gm__ uint32_t*)(userWs + 3 * mAlign * sizeof(int32_t)));
        partialGm_.SetGlobalBuffer((__gm__ int32_t*)(userWs + 4 * mAlign * sizeof(int32_t)));
    }
}

// Sort-based MUL path (global sort + count-balanced split + sl column-tiling) lives in its own file --
// it is not SIMT. Defines ScatterReduceSimt::SortReduceProcess and its helpers.
#include "scatter_reduce_common_sort.h"


template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Process()
{
    if (tiling_.sliceSize == 0 || blockIdx_ >= tiling_.blockNum) {
        return;
    }
    const ADDR_T sliceSize = static_cast<ADDR_T>(tiling_.sliceSize);
    const ADDR_T varFirstDim = static_cast<ADDR_T>(tiling_.varFirstDim);
    if constexpr (Mode == MODE_MUL || Mode == MODE_MAX || Mode == MODE_MIN) {
        // Product/max/min are order-independent (assoc.+comm.), so MUL/MAX/MIN share ONE path: global sort (core 0) +
        // count-balanced segment split (all cores), no atomics. Its only ceiling is fitting core 0's UB --
        // the M-index sort plus a single column chunk of the per-segment product -- both derived from the
        // buffer layout and UB size (no magic thresholds). sliceSize is column-tiled so it never limits; M
        // is bounded only by the single-core sort capacity.
        // sliceSize==1, small (M + varFirstDim fit single-core UB): 1D scalar reduce (product/max/min). One core
        // stages it all in UB and scalar-folds in fp32 -- crisply bounded, beats the multi-core sort's setup for
        // the tiny work. Cap is UB capacity (not a magic 256), so all realistic 1-D cases land here.
        if (sliceSize == 1 && totalIdxn_ > 0 && varFirstDim > 0 &&
            (totalIdxn_ + varFirstDim) <= static_cast<ADDR_T>(SCALAR_SLICE1_CAP)) {
            if (blockIdx_ == 0) {
                ScalarReduce1D<Mode, PARAMS_T, INDICES_T, ADDR_T>(pipe_, idxGm_, xGm_, outputGm_,
                                                            totalIdxn_, varFirstDim);
            }
            return;  // single core owns the whole 1D reduction (separate file: scatter_reduce_common_scalar.h)
        }
        // sliceSize>=2 -> sort with the column-tiled VECTOR fold (any M).
        if (sliceSize >= 2 && totalIdxn_ > 0 && varFirstDim > 0) {
            SortReduceProcess(totalIdxn_, sliceSize, varFirstDim);
            return;
        }
        // sliceSize==1 too big for the scalar UB cap -> sort + SCALAR row-fold. The vector fold crashes at
        // width 1, but a scalar product per sorted row segment does not -- so sort handles sliceSize==1 here,
        // no UB limit and no SIMT atomic CAS storm. Replaces the old fp16-overflowing SIMT-atomic fallback.
        if (sliceSize == 1 && totalIdxn_ > 0 && varFirstDim > 0) {
            SortScalarSliceSize1(totalIdxn_, varFirstDim);
            return;
        }
    }
    if constexpr (Mode == MODE_DIV) {
        // sliceSize==1 small -> single-core scalar sequential divide (ScalarDiv1D: /0 keeps var for int, ->inf
        // for float, reference-exact). Large -> sort + scalar row-fold (sequential divide, same /0 rule).
        if (sliceSize == 1 && totalIdxn_ > 0 && varFirstDim > 0 &&
            (totalIdxn_ + varFirstDim) <= static_cast<ADDR_T>(SCALAR_SLICE1_CAP)) {
            if (blockIdx_ == 0) {
                ScalarDiv1D<PARAMS_T, INDICES_T, ADDR_T>(pipe_, idxGm_, xGm_, outputGm_,
                                                         totalIdxn_, varFirstDim);
            }
            return;
        }
        if (sliceSize >= 2 && totalIdxn_ > 0 && varFirstDim > 0) {
            SortDivRowProcess(totalIdxn_, sliceSize, varFirstDim);   // sliceSize>=2: row-split in-place divide
            return;
        }
        if (sliceSize == 1 && totalIdxn_ > 0 && varFirstDim > 0) {
            SortScalarSliceSize1(totalIdxn_, varFirstDim);           // sliceSize==1 large: sort + scalar divide
            return;
        }
    }
    // Empty (M==0 or varFirstDim==0): nothing to reduce -> fall through and return (no-op). No SIMT path remains.
}
}  // namespace ScatterReduceCommon

#endif  // SCATTER_REDUCE_COMMON_SIMT_H
