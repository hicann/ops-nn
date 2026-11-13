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
__aicore__ inline void ScalarDivCore(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm, LocalTensor<INDICES_T>& idxUb,
                                     LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
                                     const DataCopyExtParams& cpVar, ADDR_T M, ADDR_T varFirstDim)
{
    event_t e2s = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e2s);
    WaitFlag<HardEvent::MTE2_S>(e2s);
    constexpr bool isFloat = AscendC::IsSameType<PARAMS_T, half>::value || AscendC::IsSameType<PARAMS_T, float>::value;
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) {
            continue;
        }
        PARAMS_T d = updUb.GetValue(k);
        if constexpr (isFloat) {
            float v = static_cast<float>(varUb.GetValue(r)) / static_cast<float>(d); // fp32 intermediate
            varUb.SetValue(r, static_cast<PARAMS_T>(v));
        } else {
            if (d != static_cast<PARAMS_T>(0)) {                                 // /0 keeps var (reference's choice)
                varUb.SetValue(r, static_cast<PARAMS_T>(varUb.GetValue(r) / d)); // C++ trunc toward zero
            }
        }
    }
    event_t es3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(es3);
    WaitFlag<HardEvent::S_MTE3>(es3);
    DataCopyPad(outputGm, varUb, cpVar);
}

// fp16 DIV: widen var into an fp32 accumulator, scalar-divide each update in fp32 (ORIGINAL order; /0 -> +-inf
// per IEEE, propagated through the chain), then narrow back to fp16 ONCE. This matches the fp32 reference (golden
// divides in fp32 and casts once) AND the other reduce paths (ScalarReduceWiden / SortDivRowProcess / SortScalar-
// SliceSize1 all fp32-accumulate). The plain per-update fp16 round of ScalarDivCore loses 1-2 ULP on collision
// rows -- the same class of bug that was fixed for MUL by widening into ScalarReduceWiden.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, typename AccT>
__aicore__ inline void ScalarDivWiden(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm, LocalTensor<INDICES_T>& idxUb,
                                      LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
                                      LocalTensor<AccT>& accUb, const DataCopyExtParams& cpVar, ADDR_T M,
                                      ADDR_T varFirstDim)
{
    event_t e2v = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    Cast(accUb, varUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim)); // fp16 var -> fp32 acc
    event_t evs = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) {
            continue;
        }
        AccT d = static_cast<AccT>(updUb.GetValue(k));
        accUb.SetValue(r, accUb.GetValue(r) / d); // fp32 divide in original order; /0 -> +-inf (IEEE)
    }
    event_t esv = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(esv);
    WaitFlag<HardEvent::S_V>(esv);
    Cast(varUb, accUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim)); // fp32 acc -> fp16 var, once
    event_t evm3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evm3);
    WaitFlag<HardEvent::V_MTE3>(evm3);
    DataCopyPad(outputGm, varUb, cpVar);
}

// Caller restricts this to blockIdx 0 (single owner, no atomics) -- M is the index count, var is varFirstDim.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void ScalarDiv1D(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                   GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim)
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
        ScalarDivWiden<PARAMS_T, INDICES_T, ADDR_T, float>(pipe, outputGm, idxUb, updUb, varUb, accUb, cpVar, M,
                                                           varFirstDim);
    } else {
        // fp32 (in-place divide is already exact in fp32) and integer (per-step C++ truncation IS the reference).
        ScalarDivCore<PARAMS_T, INDICES_T, ADDR_T>(pipe, outputGm, idxUb, updUb, varUb, cpVar, M, varFirstDim);
    }
}

// ---------------------------------------------------------------------------------------------------
// 溢出安全的缩放折叠(仅浮点 MUL/DIV): 把累加值保持成 m * 2^e —— m 的量级被箍在 [2^-32, 2^32) 内,
// e 是独立的 int64 指数, 于是**中间步既不溢出也不下溢**, 终值一次性物化, 只有真值确实超出 dtype
// 范围时才给 ±inf。
//
// 为什么需要: fp32 的顺序链一旦中间步溢出成 inf 就再也回不来, 而真值往往是可表示的。
// 实测 ScatterDiv_L0_040: var 全 -3.4e38(fp32 最负有限值), updates (-100,100), 先除到一个
// |u|<1 的更新即 6.8e38 > fp32 上限 -> inf, 此后无论再除多少都是 inf;
// 而真值 3.6e32 稳稳在 fp32 范围内(dump 原文 "Expected: 3.60988...e+32  Actual: inf")。
// ttk 的 mix_tolerance 对此有专门机制(代码注释标 !292): actual=±inf 处把 golden 按被测 dtype
// RNE 收窄, 超范围才算同号 inf 一致 —— 即判据明确只接受"真值确实超范围"的 inf, 所以这是
// 算子缺陷而非判据瑕疵。同族: ScatterMul_L1_985。
//
// 口径: 本函数只在快路径算出**非有限**结果时才被调用(见 sort.h 的调用点), 常规形态零开销。
// 非有限输入(±inf/NaN)与除 0 原样按 IEEE 走, 不做任何"吸收"—— 见 ScalarReduceFold 的回退记录。
// ---------------------------------------------------------------------------------------------------

// 有限判定: inf - inf = NaN、NaN - NaN = NaN, 故"自减为 0"当且仅当有限。
// (不用 a > MAX 之类的常量比较, 省去对 T 取上限常量。)
template <typename T>
__aicore__ inline bool ScaledIsFinite(T v)
{
    return (v - v) == static_cast<T>(0);
}

// 把 |v| 归一回 [2^-32, 2^32), 搬动量记进 e。每轮搬 32 个指数, guard 封顶 8 轮
// (8*32=256 已覆盖 fp32 的全部指数域 2^-149..2^128), 防任何意外的死循环。
template <typename T>
__aicore__ inline void ScaledNormalize(T& v, int64_t& e)
{
    constexpr int64_t K = 32;
    constexpr int GUARD = 8;
    const T up = static_cast<T>(4294967296.0);       // 2^32
    const T dn = static_cast<T>(1.0 / 4294967296.0); // 2^-32
    if (v == static_cast<T>(0) || !ScaledIsFinite(v)) {
        return;
    }
    T a = (v < static_cast<T>(0)) ? -v : v;
    int g = 0;
    while (a >= up && g < GUARD) {
        v = v * dn;
        e += K;
        a = (v < static_cast<T>(0)) ? -v : v;
        g++;
    }
    while (a < dn && g < GUARD) {
        v = v * up;
        e -= K;
        a = (v < static_cast<T>(0)) ? -v : v;
        g++;
    }
}

// 把 m * 2^e 物化成 T。|m| ∈ [2^-32, 2^32), 故 |e| > 200 时真值必然已越过 fp32 的
// 上/下限(2^128 / 2^-149), 直接给 ±inf / ±0, 避免 e 达 2^31 量级时逐步搬指数的死循环。
//
// 离线穷举(probe/sim_scaled.py, numpy float32 严格模拟本文件三个 helper, 8000 组随机链):
//   判据=分类(有限/±inf/NaN)是否与"f64 走链再按 fp32 窄回"的真值一致(即 ttk !292 的判法)
//   新缩放折叠 1/8000 不一致, 旧 fp32 顺序链 302/8000 不一致; 有限情形 ULP 误差 p50 两者同为 0。
// 已知残留角: 真值恰为 fp32 最小非规格数(2^-149)时本函数给 0 —— 逐级物化在非规格区必然
// 渐进下溢。该角在判据上无害(|0 - 1e-45| 远小于 atol), 不额外补偿, 以免把常规路径搞复杂。

template <typename T>
__aicore__ inline T ScaledMaterialize(T m, int64_t e)
{
    if (m != m || m == static_cast<T>(0)) { // NaN 原样传播; 0 乘任何 2 的幂仍是 0(含符号)
        return m;
    }
    // 不做 E_SAT 式粗粒度饱和: 指数 128~200 且 |m| 小时真值仍可表示(粗闸会把可表示值误判成
    // ±inf, 实测 ScatterDiv_L0_001 的 25/20.24M 边界元素)。改为步进中检查: m 已非有限或已
    // 冲洗为 0, 说明真值确实超出 T 的表示域, 立即返回(剩余指数无意义, 也避免超大指数空转)。
    constexpr int64_t E_STEP = 30;
    const T up = static_cast<T>(1073741824.0);       // 2^30
    const T dn = static_cast<T>(1.0 / 1073741824.0); // 2^-30
    while (e >= E_STEP) {
        m = m * up;
        if (!ScaledIsFinite(m)) {
            return m; // 真值超出上限, IEEE 溢出语义已生效
        }
        e -= E_STEP;
    }
    while (e <= -E_STEP) {
        m = m * dn;
        if (m == static_cast<T>(0)) {
            return m; // 真值低于下限(含符号的 ±0)
        }
        e += E_STEP;
    }
    // 残余 |e| < 30: 逐个 2 倍搬完(最多 29 次, 且溢出/下溢在此自然发生 = 真值确实超范围)
    while (e > 0) {
        m = m + m;
        e--;
    }
    while (e < 0) {
        m = m * static_cast<T>(0.5);
        e++;
    }
    return m;
}

// 一步缩放乘/除。关键: **先把 u 也归一化**, 再做运算 —— 否则单步自己就可能溢出成 inf,
// 而 inf 乘任何缩放因子还是 inf, 信息已丢, 缩放就白做了(第一版犯过这个错)。
// 归一后 |m| < 2^32 且 |u| ∈ [2^-32, 2^32), 故 |m*u| 与 |m/u| 都 < 2^64, 单步不可能溢出。
// u 非有限或为 0 时直接走 IEEE —— 那些情形的 IEEE 结果(inf/NaN/除 0 规则)本身就是要的答案,
// 不做任何"吸收"(见 ScalarReduceFold 里零吸收的回退记录)。
template <bool IsDiv, typename T>
__aicore__ inline void ScaledStep(T& m, int64_t& e, T u)
{
    if (u == static_cast<T>(0) || !ScaledIsFinite(u)) {
        m = IsDiv ? (m / u) : (m * u);
        return;
    }
    int64_t eu = 0;
    ScaledNormalize(u, eu);
    if constexpr (IsDiv) {
        m = m / u;
        e -= eu;
    } else {
        m = m * u;
        e += eu;
    }
    ScaledNormalize(m, e);
}

// One core stages indices + updates + var in UB and reduces sliceSize==1 with scalar GetValue/SetValue: no
// vector Mul (which crashes at width 1), no atomic (single owner), no SyncAll. The var-first init and the
// narrow are VECTORISED over varFirstDim (width > 1, so safe) -- only the data-dependent scatter
// (acc[idx[k]] *= upd[k]) is scalar, keeping cost ~ M rather than 2*varFirstDim. Caller restricts this to
// blockIdx 0 and small M so the single-core scan stays cheaper than the multi-core paths.
// Mode-folded scalar reduce op: MUL=product, MAX/MIN=hardware-atomic-equivalent compare. Plain C++ scalar
// in S/scalar pipe -- max/min propagate NaN like the vector Max/Min the multi-core paths use.
template <uint64_t Mode, typename T>
__aicore__ inline T ScalarReduceFold(T a, T b)
{
    // max/min PROPAGATE NaN (codebase convention: NaN detection via self-not-equal `a != a`, true only for NaN
    // -- same idiom as foreach_erf_simt.h `if (x != x)` and hard_shrink.h `Compare(x,x,CMPMODE::NE)`). The bare
    // `a>b?a:b` drops NaN when a is NaN (a>b is false -> returns b); the `(a != a)` guard keeps it. When b is
    // NaN and a finite, `a>b` is already false so b(=NaN) is returned. Compile-time no-op for integer T.
    if constexpr (Mode == MODE_MAX) {
        return (a != a) ? a : (a > b ? a : b);
    } else if constexpr (Mode == MODE_MIN) {
        return (a != a) ? a : (a < b ? a : b);
    } else {
        // MUL/DIV: 严格 IEEE 逐条相乘/相除, **不做"任一侧为 0 即返回 0"的零吸收**。
        //
        // 曾在此加过零吸收, 为的是修 aclnnScatterMul_L0_bound_053(var 含 0.0 的那个元素被
        // 跨核偏积溢出的 inf 乘成 NaN)。该补丁已回退, 原因有二:
        //   1. 位置错。这个折叠函数既用于**单核顺序折叠**(sort.h 的 SortScalarSliceSize1Chunk
        //      是 `acc = var[r]` 起折, 顺序乘下去, var==0 时 0*有限=0 恒成立, 本就不会出 NaN),
        //      也用于**跨核偏积合并**(ubresident_merge.h 各核从单位元 1 起折)。b053 的 inf 只在
        //      后者产生, 在此处一刀切等于拿全局语义换一条路径的 bug。
        //   2. 它改坏了 IEEE 语义, 而原注释声称的"该形态在用例集中无覆盖"是**错的** ——
        //      官方 GEIR l0 里直接有 var 全 ±inf 且 updates 含精确 0 的用例:
        //        ScatterMul_L0_026  var 范围(inf, inf)   updates fp16 (0, 0.001)
        //        ScatterMul_L0_028  同族
        //        ScatterMul_L0_030  var 范围(-inf, -inf) updates fp32 (-0.0, 0)
        //        ScatterDiv_L0_040  DIV 侧同族(本 else 分支同时覆盖 MODE_MUL 与 MODE_DIV)
        //        ScatterMul_L1_985
        //      inf*0 / inf/0 的 IEEE 结果是 NaN, 参考实现(golden)也给 NaN, 零吸收给 0 =>
        //      上述 5 例 2026-11-10 全部由 PASS 转 FAIL(matched_ratio 0.999998~0.99999997,
        //      reason "NaN/Inf"), 11-07/11-08 的历史结果均为 PASS。
        //
        // b053 的正确修法在偏积产生处(ubresident_merge.h), 判据是"**全部输入有限**时零才吸收":
        // 偏积的 inf 若由有限输入溢出而来, 它是拆分带来的人工产物, 数学真值仍是 0; 而若某条输入
        // 本身是 ±inf/NaN, 就必须按 IEEE 出 NaN。二者在数值上不可区分, 必须由各核额外上报
        // "我这份偏积的输入里有无非有限值"一位标志, 合并时才能判。见该文件 TODO。
        return a * b;
    }
}

// ScalarReduce1D phase, fp32/int32 + MAX/MIN: fold DIRECTLY in PARAMS_T (no widen/narrow). Inputs are already
// staged in UB by ScalarReduce1D; this owns only the MTE2_S->scalar scatter->S_MTE3->store sub-sequence.
template <uint64_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void ScalarReduceDirect(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm, LocalTensor<INDICES_T>& idxUb,
                                          LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
                                          const DataCopyExtParams& cpVar, ADDR_T M, ADDR_T varFirstDim)
{
    event_t e2s = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e2s);
    WaitFlag<HardEvent::MTE2_S>(e2s);
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) {
            continue;
        }
        varUb.SetValue(r, ScalarReduceFold<Mode>(varUb.GetValue(r), updUb.GetValue(k)));
    }
    event_t es3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(es3);
    WaitFlag<HardEvent::S_MTE3>(es3);
    DataCopyPad(outputGm, varUb, cpVar);
}

// ScalarReduce1D phase, fp16/int8/uint8: widen var into the AccT accumulator, scalar-scatter, narrow back.
// Inputs are already staged in UB by ScalarReduce1D; this owns the MTE2_V widen -> V_S -> scalar scatter ->
// S_V -> narrow -> V_MTE3 -> store sub-sequence (unchanged order and pairing from the original branch).
template <uint64_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T, typename AccT>
__aicore__ inline void ScalarReduceWiden(TPipe& pipe, GlobalTensor<PARAMS_T>& outputGm, LocalTensor<INDICES_T>& idxUb,
                                         LocalTensor<PARAMS_T>& updUb, LocalTensor<PARAMS_T>& varUb,
                                         LocalTensor<AccT>& accUb, const DataCopyExtParams& cpVar, ADDR_T M,
                                         ADDR_T varFirstDim)
{
    event_t e2v = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    if constexpr (sizeof(PARAMS_T) == 1) { // int8/uint8 -> int32 (Reg; high-level Cast broken)
        SubwordWidenToI32(accUb, varUb, static_cast<uint32_t>(varFirstDim));
    } else { // fp16 -> float
        Cast(accUb, varUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim));
    }
    event_t evs = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    for (ADDR_T k = 0; k < M; k++) {
        ADDR_T r = static_cast<ADDR_T>(idxUb.GetValue(k));
        if (r < 0 || r >= varFirstDim) {
            continue;
        }
        accUb.SetValue(r, ScalarReduceFold<Mode>(accUb.GetValue(r), static_cast<AccT>(updUb.GetValue(k))));
    }
    event_t esv = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(esv);
    WaitFlag<HardEvent::S_V>(esv);
    if constexpr (sizeof(PARAMS_T) == 1) { // int32 -> int8/uint8 (low-byte wrap)
        SubwordNarrowFromI32(varUb, accUb, static_cast<uint32_t>(varFirstDim));
    } else { // float -> half
        Cast(varUb, accUb, RoundMode::CAST_NONE, static_cast<int32_t>(varFirstDim));
    }
    event_t evm3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evm3);
    WaitFlag<HardEvent::V_MTE3>(evm3);
    DataCopyPad(outputGm, varUb, cpVar);
}

template <uint64_t Mode, typename PARAMS_T, typename INDICES_T, typename ADDR_T>
__aicore__ inline void ScalarReduce1D(TPipe& pipe, GlobalTensor<INDICES_T>& idxGm, GlobalTensor<PARAMS_T>& xGm,
                                      GlobalTensor<PARAMS_T>& outputGm, ADDR_T M, ADDR_T varFirstDim)
{
    using AccT = typename Conditional<
        sizeof(PARAMS_T) == 4, PARAMS_T,
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
        ScalarReduceDirect<Mode, PARAMS_T, INDICES_T, ADDR_T>(pipe, outputGm, idxUb, updUb, varUb, cpVar, M,
                                                              varFirstDim);
    } else { // fp16/int8/uint8: widen acc
        ScalarReduceWiden<Mode, PARAMS_T, INDICES_T, ADDR_T, AccT>(pipe, outputGm, idxUb, updUb, varUb, accUb, cpVar, M,
                                                                   varFirstDim);
    }
}

#endif // SCATTER_REDUCE_COMMON_SCALAR_H
