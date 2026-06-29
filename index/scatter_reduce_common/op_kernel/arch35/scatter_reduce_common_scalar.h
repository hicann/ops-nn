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
 * \file scatter_reduce_common_scalar.h
 * \brief sliceSize==1 fast path for scatter-reduce MUL: a single-core SCALAR reduction. A single-element-slice
 *        reduction is a 1D scalar histogram-product, qualitatively different from sliceSize>=2: vectorised
 *        binning crashes at width 1 and the multi-core sort machinery is pure overhead for the tiny work. This
 *        crisp class (sliceSize == 1) gets its own file/path, separate from the sort and SIMT reducers.
 *
 * FRAGMENT header: #include'd from the dispatcher INSIDE namespace ScatterReduceCommon, after
 * SubwordWidenToI32 / SubwordNarrowFromI32 (which it reuses) and the `using namespace AscendC`. Not
 * standalone-compilable -- it inherits those from the include site on purpose, so the scalar path stays a
 * thin leaf with no duplicated includes/namespace.
 */
#ifndef SCATTER_REDUCE_COMMON_SCALAR_H
#define SCATTER_REDUCE_COMMON_SCALAR_H

// sliceSize==1 DIV: one core scalar-divides var by each update in ORIGINAL order. Scalar GetValue/SetValue
// gives reference-exact semantics for free -- no vector Div (which neither truncates toward zero for int nor
// runs at width 1) and no atomic CAS (which livelocks under repeated indices). Per element:
//   - integer (int8/uint8/int32): C++ integer division truncates toward zero, and the int8/uint8 GetValue
//     promotes to a SIGNED/UNSIGNED int correctly (the sign bug the vector path had to fix by hand); /0 KEEPS
//     var (the reference's choice), so we just skip d==0.
//   - float (fp16/fp32): divide in fp32 (matching the reference intermediate); /0 -> +-inf per IEEE.
// ScalarDiv1D consume phase: inputs are already staged in UB by ScalarDiv1D; this owns only the
// MTE2_S -> scalar divide scatter -> S_MTE3 -> store sub-sequence (unchanged order and pairing from the
// original body). Per element divides var by upd in ORIGINAL order: float in fp32 (/0 -> +-inf), integer
// C++ trunc toward zero (/0 keeps var). Mirrors the ScalarReduceDirect split of ScalarReduce1D.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void ScalarDivCore(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm,
    LocalTensor<INDICES_T>& idxUb, LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
    const DataCopyExtParams& cpVar, ADDR_T M, ADDR_T varFirstDim)
{
    event_t e2s = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e2s); WaitFlag<HardEvent::MTE2_S>(e2s);
    constexpr bool isFloat = AscendC::IsSameType<PARAMS_T, half>::value ||
                             AscendC::IsSameType<PARAMS_T, float>::value;
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) { continue; }
        PARAMS_T d = updUb.GetValue(k);
        if constexpr (isFloat) {
            float v = static_cast<float>(varUb.GetValue(r)) / static_cast<float>(d);  // fp32 intermediate
            varUb.SetValue(r, static_cast<PARAMS_T>(v));
        } else {
            if (d != static_cast<PARAMS_T>(0)) {                       // /0 keeps var (reference's choice)
                varUb.SetValue(r, static_cast<PARAMS_T>(varUb.GetValue(r) / d));  // C++ trunc toward zero
            }
        }
    }
    event_t es3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(es3); WaitFlag<HardEvent::S_MTE3>(es3);
    DataCopyPad(outputGm, varUb, cpVar);
}

// fp16 DIV: widen var into an fp32 accumulator, scalar-divide each update in fp32 (ORIGINAL order; /0 -> +-inf
// per IEEE, propagated through the chain), then narrow back to fp16 ONCE. This matches the fp32 reference (golden
// divides in fp32 and casts once) AND the other reduce paths (ScalarReduceWiden / SortDivRowProcess / SortScalar-
// SliceSize1 all fp32-accumulate). The plain per-update fp16 round of ScalarDivCore loses 1-2 ULP on collision
// rows -- the same class of bug that was fixed for MUL by widening into ScalarReduceWiden.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, typename AccT>
__aicore__ inline void ScalarDivWiden(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm,
    LocalTensor<INDICES_T>& idxUb, LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
    LocalTensor<AccT>& accUb, const DataCopyExtParams& cpVar, ADDR_T M, ADDR_T varFirstDim)
{
    event_t e2v = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v);
    Cast(accUb, varUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim));   // fp16 var -> fp32 acc
    event_t evs = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evs); WaitFlag<HardEvent::V_S>(evs);
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) { continue; }
        AccT d = static_cast<AccT>(updUb.GetValue(k));
        accUb.SetValue(r, accUb.GetValue(r) / d);   // fp32 divide in original order; /0 -> +-inf (IEEE)
    }
    event_t esv = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(esv); WaitFlag<HardEvent::S_V>(esv);
    Cast(varUb, accUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim));   // fp32 acc -> fp16 var, once
    event_t evm3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evm3); WaitFlag<HardEvent::V_MTE3>(evm3);
    DataCopyPad(outputGm, varUb, cpVar);
}

// Caller restricts this to blockIdx 0 (single owner, no atomics) -- M is the index count, var is varFirstDim.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void ScalarDiv1D(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm,
    GlobalTensor<PARAMS_T>& xGm, GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim)
{
    // M is bounded by UB capacity: this single-core path stages all M indices + M updates into UB
    // (idxBuf/updBuf below), so mPad*sizeof <= UB(248KB) => M <= ~31K. Hence M*sizeof(INDICES_T) is well
    // within uint32_t and the DataCopyExtParams byte-length casts below cannot overflow (the DIV dispatch
    // has no explicit M cap unlike MUL<=256/MAX-MIN<=512, but the UB stage is the effective bound).
    const uint32_t mPad = (static_cast<uint32_t>(M) + 31) / 32 * 32;
    const uint32_t vPad = (static_cast<uint32_t>(varFirstDim) + 31) / 32 * 32;
    TBuf<TPosition::VECCALC> idxBuf, updBuf, varBuf;
    pipe.InitBuffer(idxBuf, mPad * sizeof(INDICES_T));
    pipe.InitBuffer(updBuf, mPad * sizeof(PARAMS_T));
    pipe.InitBuffer(varBuf, vPad * sizeof(PARAMS_T));
    LocalTensor<INDICES_T> idxUb = idxBuf.Get<INDICES_T>();
    LocalTensor<PARAMS_T> updUb = updBuf.Get<PARAMS_T>();
    LocalTensor<PARAMS_T> varUb = varBuf.Get<PARAMS_T>();
    DataCopyExtParams cpIdx{1, static_cast<uint32_t>(M * sizeof(INDICES_T)), 0, 0, 0};
    DataCopyExtParams cpUpd{1, static_cast<uint32_t>(M * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyExtParams cpVar{1, static_cast<uint32_t>(varFirstDim * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPad(idxUb, idxGm, cpIdx, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
    DataCopyPad(updUb, xGm, cpUpd, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
    DataCopyPad(varUb, outputGm, cpVar, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
    if constexpr (AscendC::IsSameType<PARAMS_T, half>::value) {
        // fp16: accumulate the per-row division chain in fp32 and round to fp16 ONCE (avoids the per-update
        // fp16 rounding that diverges 1-2 ULP from the fp32 reference on collision rows).
        TBuf<TPosition::VECCALC> accBuf;
        pipe.InitBuffer(accBuf, vPad * sizeof(float));
        LocalTensor<float> accUb = accBuf.Get<float>();
        ScalarDivWiden<PARAMS_T, INDICES_T, ADDR_T, float>(pipe, outputGm, idxUb, updUb, varUb, accUb,
                                                           cpVar, M, varFirstDim);
    } else {
        // fp32 (in-place divide is already exact in fp32) and integer (per-step C++ truncation IS the reference).
        ScalarDivCore<PARAMS_T, INDICES_T, ADDR_T>(pipe, outputGm, idxUb, updUb, varUb, cpVar, M, varFirstDim);
    }
}

// One core stages indices + updates + var in UB and reduces sliceSize==1 with scalar GetValue/SetValue: no
// vector Mul (which crashes at width 1), no atomic (single owner), no SyncAll. The var-first init and the
// narrow are VECTORISED over varFirstDim (width > 1, so safe) -- only the data-dependent scatter
// (acc[idx[k]] *= upd[k]) is scalar, keeping cost ~ M rather than 2*varFirstDim. Caller restricts this to
// blockIdx 0 and small M so the single-core scan stays cheaper than the multi-core paths.
// Mode-folded scalar reduce op: MUL=product, MAX/MIN=hardware-atomic-equivalent compare. Plain C++ scalar
// in S/scalar pipe -- max/min propagate NaN like the vector Max/Min the multi-core paths use.
template <uint64_t Mode, typename T> __aicore__ inline T ScalarReduceFold(T a, T b)
{
    // max/min PROPAGATE NaN (codebase convention: NaN detection via self-not-equal `a != a`, true only for NaN
    // -- same idiom as foreach_erf_simt.h `if (x != x)` and hard_shrink.h `Compare(x,x,CMPMODE::NE)`). The bare
    // `a>b?a:b` drops NaN when a is NaN (a>b is false -> returns b); the `(a != a)` guard keeps it. When b is
    // NaN and a finite, `a>b` is already false so b(=NaN) is returned. Compile-time no-op for integer T.
    if constexpr (Mode == MODE_MAX) { return (a != a) ? a : (a > b ? a : b); }
    else if constexpr (Mode == MODE_MIN) { return (a != a) ? a : (a < b ? a : b); }
    else { return a * b; }
}

// ScalarReduce1D phase, fp32/int32 + MAX/MIN: fold DIRECTLY in PARAMS_T (no widen/narrow). Inputs are already
// staged in UB by ScalarReduce1D; this owns only the MTE2_S->scalar scatter->S_MTE3->store sub-sequence.
template <uint64_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void ScalarReduceDirect(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm,
    LocalTensor<INDICES_T>& idxUb, LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
    const DataCopyExtParams& cpVar, ADDR_T M, ADDR_T varFirstDim)
{
    event_t e2s = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e2s); WaitFlag<HardEvent::MTE2_S>(e2s);
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) { continue; }
        varUb.SetValue(r, ScalarReduceFold<Mode>(varUb.GetValue(r), updUb.GetValue(k)));
    }
    event_t es3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(es3); WaitFlag<HardEvent::S_MTE3>(es3);
    DataCopyPad(outputGm, varUb, cpVar);
}

// ScalarReduce1D phase, fp16/int8/uint8: widen var into the AccT accumulator, scalar-scatter, narrow back.
// Inputs are already staged in UB by ScalarReduce1D; this owns the MTE2_V widen -> V_S -> scalar scatter ->
// S_V -> narrow -> V_MTE3 -> store sub-sequence (unchanged order and pairing from the original branch).
template <uint64_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T, typename AccT>
__aicore__ inline void ScalarReduceWiden(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm,
    LocalTensor<INDICES_T>& idxUb, LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
    LocalTensor<AccT>& accUb, const DataCopyExtParams& cpVar, ADDR_T M, ADDR_T varFirstDim)
{
    event_t e2v = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v);
    if constexpr (sizeof(PARAMS_T) == 1) {  // int8/uint8 -> int32 (MicroAPI; high-level Cast broken)
        SubwordWidenToI32(accUb, varUb, static_cast<uint32_t>(varFirstDim));
    } else {  // fp16 -> float
        Cast(accUb, varUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim));
    }
    event_t evs = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evs); WaitFlag<HardEvent::V_S>(evs);
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) { continue; }
        accUb.SetValue(r, ScalarReduceFold<Mode>(accUb.GetValue(r), static_cast<AccT>(updUb.GetValue(k))));
    }
    event_t esv = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(esv); WaitFlag<HardEvent::S_V>(esv);
    if constexpr (sizeof(PARAMS_T) == 1) {  // int32 -> int8/uint8 (low-byte wrap)
        SubwordNarrowFromI32(varUb, accUb, static_cast<uint32_t>(varFirstDim));
    } else {  // float -> half
        Cast(varUb, accUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim));
    }
    event_t evm3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evm3); WaitFlag<HardEvent::V_MTE3>(evm3);
    DataCopyPad(outputGm, varUb, cpVar);
}

template <uint64_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void ScalarReduce1D(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm,
    GlobalTensor<PARAMS_T>& xGm, GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim)
{
    using AccT = typename Conditional<sizeof(PARAMS_T) == 4, PARAMS_T,
        typename Conditional<AscendC::IsSameType<PARAMS_T, half>::value, float, int32_t>::type>::type;
    const uint32_t mPad = (static_cast<uint32_t>(M) + 31) / 32 * 32;
    const uint32_t vPad = (static_cast<uint32_t>(varFirstDim) + 31) / 32 * 32;
    TBuf<TPosition::VECCALC> idxBuf, updBuf, varBuf, accBuf;
    pipe.InitBuffer(idxBuf, mPad * sizeof(INDICES_T));
    pipe.InitBuffer(updBuf, mPad * sizeof(PARAMS_T));
    pipe.InitBuffer(varBuf, vPad * sizeof(PARAMS_T));
    pipe.InitBuffer(accBuf, vPad * sizeof(AccT));
    LocalTensor<INDICES_T> idxUb = idxBuf.Get<INDICES_T>();
    LocalTensor<PARAMS_T> updUb = updBuf.Get<PARAMS_T>();
    LocalTensor<PARAMS_T> varUb = varBuf.Get<PARAMS_T>();
    LocalTensor<AccT> accUb = accBuf.Get<AccT>();
    DataCopyExtParams cpIdx{1, static_cast<uint32_t>(M * sizeof(INDICES_T)), 0, 0, 0};
    DataCopyExtParams cpUpd{1, static_cast<uint32_t>(M * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyExtParams cpVar{1, static_cast<uint32_t>(varFirstDim * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPad(idxUb, idxGm, cpIdx, DataCopyPadExtParams<INDICES_T>{false, 0, 0, 0});
    DataCopyPad(updUb, xGm, cpUpd, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
    DataCopyPad(varUb, outputGm, cpVar, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
    // MAX/MIN never overflow (result is one of the inputs) -> fold DIRECTLY in PARAMS_T, no widen/narrow.
    // The MUL widen (SubwordWidenToI32) does not sign-extend int8, which breaks signed max/min comparison
    // (a negative var widened as unsigned compares larger than a positive update). fp32/int32 also fold direct.
    if constexpr (AscendC::IsSameType<PARAMS_T, AccT>::value || Mode == MODE_MAX || Mode == MODE_MIN) {
        ScalarReduceDirect<Mode, PARAMS_T, INDICES_T, ADDR_T>(
            pipe, outputGm, idxUb, updUb, varUb, cpVar, M, varFirstDim);
    } else {                                                                     // fp16/int8/uint8: widen acc
        ScalarReduceWiden<Mode, PARAMS_T, INDICES_T, ADDR_T, AccT>(
            pipe, outputGm, idxUb, updUb, varUb, accUb, cpVar, M, varFirstDim);
    }
}

#endif  // SCATTER_REDUCE_COMMON_SCALAR_H
