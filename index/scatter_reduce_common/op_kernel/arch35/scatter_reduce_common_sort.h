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
 * \file scatter_reduce_common_sort.h
 * \brief Main scatter-reduce MUL path: global sort (core 0) + count-balanced segment split across cores +
 *        sliceSize column-tiling. Order-independent product, no atomics. This is NOT SIMT -- it is the
 *        sort-based reducer, kept in its own file. ScatterReduceSimt member-function DEFINITIONS for
 *        MulRangeIntoAcc / LoadWiden / NarrowStore / SortReduceProcess live here.
 *
 * FRAGMENT header: #include'd from the dispatcher (scatter_reduce_common_simt.h) AFTER the ScatterReduceSimt
 * class declaration and the subword widen/narrow helpers it reuses. Not standalone-compilable.
 */
#ifndef SCATTER_REDUCE_COMMON_SORT_H
#define SCATTER_REDUCE_COMMON_SORT_H

#include <cfloat>

// Deep-prefetch fold skeleton shared by MUL and MAX/MIN: prime PREFETCH_DEPTH-1 scattered loads, then walk
// the range keeping AHEAD loads in flight to hide the DMA latency, applying the Mode-selected fold per row. The
// prefetch mechanics (prime DataCopyPad + per-iteration AHEAD prefetch DataCopyPad/AllocTensor/EnQue + DeQue/
// FreeTensor) were duplicated verbatim between MulRangeIntoAcc and MaxMinRangeIntoAcc; only the in-loop fold
// (and MAX/MIN's trailing PipeBarrier) differed. FoldRangeIntoAcc carries that single skeleton; the per-row
// fold and the EXACT barrier rhythm of each path are selected by `if constexpr (Mode == MODE_MUL)` -- MUL's
// native branch emits no barrier, MAX/MIN folds always emit the trailing PipeBarrier<PIPE_V>() they had at the
// end of every iteration. `tmp` is only read by the MAX/MIN int8 sign-extend (unused, may be empty, for MUL).
// Shared deep-prefetch load management (see header decl): prime the first `primed` scattered loads. Identical
// prime block formerly inlined verbatim in FoldRangeIntoAcc and DivRangeIntoVar.
// int8 由 SubwordWidenToI32 零扩展而来, 需手工补符号: v -= (v >> 7) * 256。
// 7 = 8位宽减符号位, 256 = 2^8, 均由 int8 的位宽决定, 非经验值。
constexpr int32_t INT8_SIGN_SHIFT = 8 * static_cast<int32_t>(sizeof(int8_t)) - 1;
constexpr int32_t INT8_SIGN_SPAN = 1 << (8 * static_cast<int32_t>(sizeof(int8_t)));

// 浮点除法取 0 ULP 档: 默认的 DivAlgo::INTRINSIC 是硬件近似除, 实测 fp32 单次除法约 6.4% 的
// 输入偏 1 ULP; 本算子对重复索引是链式相除, 偏差逐次累积, 与正确舍入的竞品相比会被 cross_check
// 的比值判据判红。FTZ_FALSE 保留次正规数, 与 IEEE 一致。
static constexpr AscendC::DivConfig PRECISE_DIV = {AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::PrimePrefetch(
    TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T primed, ADDR_T rowStride, ADDR_T colOff,
    const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    for (ADDR_T p = 0; p < primed; p++) {
        uint32_t pp = originPosGm_.GetValue(startJ + p);
        LocalTensor<PARAMS_T> b = updQue.template AllocTensor<PARAMS_T>();
        DataCopyPad(b, xGm_[static_cast<ADDR_T>(pp) * rowStride + colOff], cp, pad);
        updQue.EnQue(b);
    }
}

// Shared deep-prefetch load management (see header decl): inside the consume loop, prefetch the row AHEAD ahead
// of `j` while still in range. Identical AHEAD block formerly inlined verbatim in the two consume loops.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::PrefetchAhead(
    TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T j, ADDR_T cnt, ADDR_T AHEAD,
    ADDR_T rowStride, ADDR_T colOff, const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    if (j + AHEAD < cnt) {
        uint32_t posN = originPosGm_.GetValue(startJ + j + AHEAD);
        LocalTensor<PARAMS_T> bn = updQue.template AllocTensor<PARAMS_T>();
        DataCopyPad(bn, xGm_[static_cast<ADDR_T>(posN) * rowStride + colOff], cp, pad);
        updQue.EnQue(bn);
    }
}

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::FoldRangeIntoAcc(
    LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
    TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff,
    ADDR_T width, const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    if (startJ >= endJ) {
        return;
    }
    constexpr ADDR_T AHEAD = static_cast<ADDR_T>(PREFETCH_AHEAD); // 留一笔给正在折叠的行
    const ADDR_T cnt = endJ - startJ;
    const ADDR_T primed = (cnt < AHEAD) ? cnt : AHEAD;
    PrimePrefetch(updQue, startJ, primed, rowStride, colOff, cp, pad);
    for (ADDR_T j = 0; j < cnt; j++) {
        LocalTensor<PARAMS_T> curRaw = updQue.template DeQue<PARAMS_T>();
        PrefetchAhead(updQue, startJ, j, cnt, AHEAD, rowStride, colOff, cp, pad);
        if constexpr (Mode == MODE_MUL) {
            // ---- MUL fold (was MulRangeIntoAcc): native vmul has no trailing barrier; widen path barriers ----
            if constexpr (AscendC::IsSameType<PARAMS_T, AccT>::value) {
                Mul(accUb, accUb, curRaw, static_cast<int32_t>(width));
            } else {
                if constexpr (sizeof(PARAMS_T) == 1) { // int8/uint8 -> int32 via Reg
                    SubwordWidenToI32(rowAccUb, curRaw, static_cast<uint32_t>(width));
                } else { // fp16 -> float
                    Cast(rowAccUb, curRaw, RoundMode::CAST_NONE, static_cast<int32_t>(width));
                }
                PipeBarrier<PIPE_V>();
                Mul(accUb, accUb, rowAccUb, static_cast<int32_t>(width));
                PipeBarrier<PIPE_V>();
            }
        } else {
            // ---- MAX/MIN fold (was MaxMinRangeIntoAcc): every branch ends with a trailing PipeBarrier ----
            if constexpr (AscendC::IsSameType<PARAMS_T, AccT>::value) { // fp32 / int32: native
                if constexpr (Mode == MODE_MAX) {
                    Max(accUb, accUb, curRaw, static_cast<int32_t>(width));
                } else {
                    Min(accUb, accUb, curRaw, static_cast<int32_t>(width));
                }
            } else if constexpr (sizeof(PARAMS_T) == 1) { // int8/uint8 -> int32
                SubwordWidenToI32(rowAccUb, curRaw, static_cast<uint32_t>(width));
                if constexpr (AscendC::IsSameType<PARAMS_T, int8_t>::value) { // sign-extend int8: v -= 256*(v>>7)
                    PipeBarrier<PIPE_V>();
                    ShiftRight(tmp, rowAccUb, INT8_SIGN_SHIFT, static_cast<int32_t>(width));
                    Muls(tmp, tmp, INT8_SIGN_SPAN, static_cast<int32_t>(width));
                    Sub(rowAccUb, rowAccUb, tmp, static_cast<int32_t>(width));
                }
                PipeBarrier<PIPE_V>();
                if constexpr (Mode == MODE_MAX) {
                    Max(accUb, accUb, rowAccUb, static_cast<int32_t>(width));
                } else {
                    Min(accUb, accUb, rowAccUb, static_cast<int32_t>(width));
                }
            } else { // fp16 -> float
                Cast(rowAccUb, curRaw, RoundMode::CAST_NONE, static_cast<int32_t>(width));
                PipeBarrier<PIPE_V>();
                if constexpr (Mode == MODE_MAX) {
                    Max(accUb, accUb, rowAccUb, static_cast<int32_t>(width));
                } else {
                    Min(accUb, accUb, rowAccUb, static_cast<int32_t>(width));
                }
            }
            PipeBarrier<PIPE_V>();
        }
        updQue.FreeTensor(curRaw);
    }
}

// Deep-prefetch multiply: accUb *= product of updates[originPos[startJ..endJ)]. accUb is initialised by the
// caller (var row, or the first update). Thin wrapper over the shared FoldRangeIntoAcc prefetch skeleton (the
// MODE_MUL fold). `tmp` is unused on the MUL path, so the empty MUL-side scratch is forwarded.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::MulRangeIntoAcc(
    LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
    TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff,
    ADDR_T width, const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    FoldRangeIntoAcc(accUb, rowAccUb, tmp, updQue, startJ, endJ, rowStride, colOff, width, cp, pad);
}

// Deep-prefetch max/min fold: accUb = max/min(accUb, updates[originPos[startJ..endJ)]) over columns
// [colOff,colOff+width). Thin wrapper over the shared FoldRangeIntoAcc prefetch skeleton (the MAX/MIN fold:
// int8/uint8 widen to int32 with int8 sign-extended via tmp; fp16 -> float; fp32/int32 native; trailing
// PipeBarrier per iteration). max/min are associative+commutative, so this partial combines across cores.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::MaxMinRangeIntoAcc(
    LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
    TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff,
    ADDR_T width, const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    FoldRangeIntoAcc(accUb, rowAccUb, tmp, updQue, startJ, endJ, rowStride, colOff, width, cp, pad);
}

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::LoadWiden(
    LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb, const GlobalTensor<PARAMS_T>& srcGm, ADDR_T off,
    ADDR_T sliceSize, event_t evMV, const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    if constexpr (AscendC::IsSameType<PARAMS_T, AccT>::value) {
        DataCopyPad(accUb.template ReinterpretCast<PARAMS_T>(), srcGm[off], cp, pad);
        SetFlag<HardEvent::MTE2_V>(evMV);
        WaitFlag<HardEvent::MTE2_V>(evMV);
    } else {
        DataCopyPad(varUb, srcGm[off], cp, pad);
        SetFlag<HardEvent::MTE2_V>(evMV);
        WaitFlag<HardEvent::MTE2_V>(evMV);
        if constexpr (sizeof(PARAMS_T) == 1) { // int8/uint8 -> int32 via Reg (high-level Cast broken)
            SubwordWidenToI32(accUb, varUb, static_cast<uint32_t>(sliceSize));
        } else { // fp16 -> float
            Cast(accUb, varUb, RoundMode::CAST_NONE, static_cast<int32_t>(sliceSize));
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::NarrowStore(
    LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb, const GlobalTensor<PARAMS_T>& dstGm, ADDR_T off,
    ADDR_T sliceSize, event_t evVM3, const DataCopyExtParams& cp)
{
    if constexpr (AscendC::IsSameType<PARAMS_T, AccT>::value) {
        SetFlag<HardEvent::V_MTE3>(evVM3);
        WaitFlag<HardEvent::V_MTE3>(evVM3);
        DataCopyPad(dstGm[off], accUb.template ReinterpretCast<PARAMS_T>(), cp);
    } else {
        if constexpr (sizeof(PARAMS_T) == 1) { // int32 -> int8/uint8 via Reg (low-byte wrap)
            SubwordNarrowFromI32(varUb, accUb, static_cast<uint32_t>(sliceSize));
        } else { // float -> half
            Cast(varUb, accUb, RoundMode::CAST_NONE, static_cast<int32_t>(sliceSize));
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evVM3);
        WaitFlag<HardEvent::V_MTE3>(evVM3);
        DataCopyPad(dstGm[off], varUb, cp);
    }
}

// TILED merge-sort capacity unit: <= the measured single-call AscendC::Sort limit (~9700), 32-aligned.
// 宽索引档的低位宽度, 必须与 host tiling 的 INDEX_LO_BITS 一致。取 30 而非 31: 归并树用
// INT32_MAX(2^31-1) 作补齐哨兵, lo 若取 31 位则最大合法 lo 恰等于哨兵, 段边界判定会把真实数据
// 当成填充。30 位下哨兵稳稳高于所有 lo, 且桶数(=ceil(dim0/2^30))在任何现实 dim0 下都只有个位数。
constexpr int32_t INDEX_LO_BITS = 30;
constexpr int32_t INDEX_LO_MASK_SHIFT = 32 - INDEX_LO_BITS; // 左移再逻辑右移, 清掉高 2 位

constexpr uint32_t SORT_TILE = 8192;
constexpr uint32_t SORT_KMAX = 64; // tiles merged in one pass; M up to KMAX*TILE = 512K (far beyond any case)

// 宽索引档: 把排序 key 截到低 INDEX_LO_BITS 位(左移再逻辑右移清掉高位)。regbase(MicroAPI VF)写法,
// 与本目录 SubwordWidenToI32 同构: 全部包在 __VEC_SCOPE__ 内, 尾块用 UpdateMask 只算有效 lane,
// 不在 kernel 侧补 pad。用 uint32 视图保证右移是**逻辑**移位(int32 会符号扩展, 高位清不掉)。
// 窄档 + int64 索引专用: 在**截断之前**判越界, 把越界项的 key 钉成 OOB_KEY。
//
// 为什么必须做: 排序只保证"key 相同的排在一起", 不保证"同一索引的多条相邻"。窄档下 int64 的
// 越界大值(如 2^32+5)被 Cast 截成低 32 位后可能落进合法区间(5), 与合法的 5 撞 key ->
// 排序后可能出现 5, 2^32+5, 5 -> 同一行被切成两段。越界项虽会被消费端守卫跳过, 但**分段已被打断**:
//   * MUL/MAX/MIN 可结合可交换, 分两段各自折叠仍正确;
//   * DIV 走行归属(一行只能一个核独占, 连除不可拆), 同一行出现两段可能被两个核同时认领 -> 竞争写。
// 钉成 OOB_KEY(=2^30, 大于所有合法 key、小于归并哨兵 INT32_MAX)后, 越界项整体聚到末尾,
// 合法索引的 key 互不碰撞, 同索引必然连续成段。
//
// int32 索引不需要本处理: 值本身就装得下, key = 原值, 天然无碰撞(负数排最前、>=dim0 的排最后,
// 都不会插进合法段中间)。宽档同样不需要: 分区按**真值**分桶, 越界项进溢出桶。
__aicore__ inline void BuildSortKeyI64(const LocalTensor<int32_t>& keys, __local_mem__ int32_t* rawI32, uint32_t n,
                                       int32_t varFirstDim, bool wide)
{
    constexpr uint32_t VL_B32 = GetVecLen() / sizeof(uint32_t);
    constexpr int32_t OOB_KEY = 1 << INDEX_LO_BITS; // 大于所有合法 key, 小于归并哨兵 INT32_MAX
    (void)varFirstDim;
    __local_mem__ int32_t* dstAddr = (__local_mem__ int32_t*)keys.GetPhyAddr();
    uint16_t loopTimes = static_cast<uint16_t>((n + VL_B32 - 1) / VL_B32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> lo;
        Reg::RegTensor<int32_t> hi;
        Reg::RegTensor<int32_t> t0;
        Reg::RegTensor<int32_t> t1;
        Reg::RegTensor<int32_t> key;
        Reg::MaskReg preg;
        uint32_t sregMask = n;
        for (uint16_t i = 0; i < loopTimes; i++) {
            preg = Reg::UpdateMask<int32_t>(sregMask);
            // int64 缓冲按 int32 视图去交织: 偶数位=低 32 位(lo), 奇数位=高 32 位(hi)。
            // 全程不经 Cast(int64->int32) —— 该转换是截断还是饱和从未验证。
            Reg::DataCopy<int32_t, Reg::LoadDist::DIST_DINTLV_B32>(lo, hi, rawI32 + i * VL_B32 * 2);
            if (wide) {
                // 宽档: key = 低 INDEX_LO_BITS 位(桶内偏移), 高位由桶号承载; 越界项由第 2 趟稳定分区
                // 按**真值**归入溢出桶, 故此处不需判越界。
                Reg::ShiftLefts(reinterpret_cast<Reg::RegTensor<uint32_t>&>(key),
                                reinterpret_cast<Reg::RegTensor<uint32_t>&>(lo),
                                static_cast<int16_t>(INDEX_LO_MASK_SHIFT), preg);
                Reg::ShiftRights(reinterpret_cast<Reg::RegTensor<uint32_t>&>(key),
                                 reinterpret_cast<Reg::RegTensor<uint32_t>&>(key),
                                 static_cast<int16_t>(INDEX_LO_MASK_SHIFT), preg);
            } else {
                // 窄档: 把越界项的 key 钉到 OOB_KEY, 使其整体聚到末尾, **不插进合法段中间**。
                // 这一步是必需的: 排序只保证"key 相同的排一起", 不保证"同一索引的多条相邻";
                // 越界大值(2^32+7)截断后 key=7 会与合法索引 7 撞车, 块内顺序任意 -> 同一行被切成
                // 多段 -> 跨核合并/行归属逻辑失效(实测表现为设备死等, 无 AICORE error)。
                //
                // 纯算术实现(不用 Compares/Select: 其掩码极性与参数序未经验证):
                //     nz  = min(|hi|, 1)              hi != 0 -> 1
                //     ns  = (uint32)lo >> 31          lo 符号位 -> 1
                //     key = max( min(lo, OOB), OOB * max(nz, ns) )
                // 合法(hi=0, lo>=0, lo<dim0<=OOB): max(lo, 0) = lo
                // hi!=0 或 lo<0: 结果恒为 OOB_KEY
                // hi=0 但 lo>=dim0: 结果 >= dim0, 本就不与合法 key 冲突
                Reg::Abs(t0, hi, preg);
                Reg::Mins(t0, t0, static_cast<int32_t>(1), preg); // nz
                Reg::ShiftRights(reinterpret_cast<Reg::RegTensor<uint32_t>&>(t1),
                                 reinterpret_cast<Reg::RegTensor<uint32_t>&>(lo), static_cast<int16_t>(31), preg);
                Reg::Maxs(t0, t0, static_cast<int32_t>(0), preg);
                Reg::Max(t0, t0, t1, preg); // max(nz, ns)
                Reg::Muls(t0, t0, static_cast<int32_t>(OOB_KEY), preg);
                Reg::Mins(key, lo, static_cast<int32_t>(OOB_KEY), preg);
                Reg::Max(key, key, t0, preg);
            }
            auto dstReg = Reg::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
            Reg::DataCopy<int32_t, Reg::StoreDist::DIST_NORM>(dstAddr, key, dstReg, preg);
        }
    }
}

__aicore__ inline void MaskKeyToLoBits(const LocalTensor<int32_t>& keys, uint32_t n)
{
    constexpr uint32_t VL_B32 = GetVecLen() / sizeof(uint32_t);
    __local_mem__ uint32_t* addr = (__local_mem__ uint32_t*)keys.GetPhyAddr();
    uint16_t loopTimes = static_cast<uint16_t>((n + VL_B32 - 1) / VL_B32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> v;
        Reg::MaskReg preg;
        uint32_t sregMask = n;
        for (uint16_t i = 0; i < loopTimes; i++) {
            auto ar = Reg::CreateAddrReg<uint32_t>(i, static_cast<uint16_t>(VL_B32));
            preg = Reg::UpdateMask<uint32_t>(sregMask);
            Reg::DataCopy<uint32_t, Reg::LoadDist::DIST_NORM>(v, addr, ar);
            // 移位量必须是 int16_t: regbase 的 ShiftLefts/ShiftRights 对标量参数有
            // static_assert(SupportType<U, int16_t>()), 传 uint32_t 会在实例化处报
            // "current scalarValue data type is not supported"(错误信息指向 impl 头文件, 不指向调用点)。
            Reg::ShiftLefts(v, v, static_cast<int16_t>(INDEX_LO_MASK_SHIFT), preg);
            Reg::ShiftRights(v, v, static_cast<int16_t>(INDEX_LO_MASK_SHIFT), preg);
            Reg::DataCopy<uint32_t, Reg::StoreDist::DIST_NORM>(addr, v, ar, preg);
        }
    }
}

// Phase 1 of every sort-based reducer: produce, in GM, sortedIdx[M] (ascending int32 keys) + originPos[M]
// (original update slot per sorted slot), then SyncAll. M<=SORT_TILE: one core, one AscendC::Sort (fast path,
// already optimal for small M). M>SORT_TILE: a TRUE PARALLEL MERGE SORT -- all P2 (=largest pow2<=blockNum)
// cores sort an equal chunk in parallel, then a binary merge tree where, EVERY round, every core produces its
// own contiguous runLen0-wide slice of the round output via a merge-path co-rank (so the merge runs on all
// cores too, not just core 0). Chunks pad to runLen0 with INT32_MAX so all runs share a length and a core's
// slice never spans more than one merge pair. No capacity cap, no gate, any M. Shared by SortReduceProcess
// (MUL/MAX/MIN) and SortDivRowProcess (DIV).
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline ADDR_T ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::RowOf(ADDR_T k)
{
    // **行号身份一律以原始索引为准**, 不读排序 key。
    //
    // 排序 key 只承担一件事: 把"可能相同"的索引排到相邻位置。它是 32 位的(硬件 Sort 的 key 只能 32 位),
    // 因此 int64 索引进 key 时会丢高位 —— 但这不影响正确性, 因为:
    //   * 同一个索引的多条必然得到同一 key -> 必然相邻 -> 段判定(比较真值)把它们合并;      [不漏]
    //   * 不同索引撞上同一 key(如 5 与 2^32+5)只会让两行恰好相邻, 段判定比较**真值**后自然分开,
    //     越界的那条再被消费端的 r<0 || r>=varFirstDim 守卫跳过。                          [不错]
    // 若改读 key, 上面第二种情况会把 2^32+5 洗白成合法行号 5, 静默写错行 —— 这正是本次重构要根除的。
    // 代价: 每次多一条标量 GM 读(originPos -> idx)。以正确性换, 且不引入任何 64 位矢量运算。
    return static_cast<ADDR_T>(idxGm_.GetValue(static_cast<ADDR_T>(originPosGm_.GetValue(k))));
}

// 宽索引档第 2 趟基数排序: 按 hi=idx>>30 做稳定分区。core 0 单核执行(桶数为个位数, 且宽档本身是
// 极端大表场景, 正确性优先), 完成后 DCCI 刷写 + SyncAll, 与 SortIndices 的收尾同一套一致性保证。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::StablePartitionByHi(ADDR_T M)
{
    // +1 = 溢出桶: 越界索引(r<0 或 r>=varFirstDim)不参与合法分桶。它们**必须**有归宿 ——
    // 索引值是输入数据, 内核语义是"越界跳过"(与 A2 的行区间归属判断等价), 所以输入里完全可能出现
    // 任意 int64 值(负数 / 2^40 / INT64_MAX)。这些值的 hi 远超 bucketNum, 若直接拿去索引计数数组
    // 会越界写栈(静默踩内存)。统一归入末尾的溢出桶, 排到最后, 之后照样被消费者的边界守卫跳过。
    constexpr int32_t MAX_BUCKETS = 64 + 1; // 64 个合法桶(首维上限 2^36) + 1 个溢出桶
    const ADDR_T bucketNum = static_cast<ADDR_T>(tiling_.bucketNum);
    const ADDR_T varFirstDim = static_cast<ADDR_T>(tiling_.varFirstDim);
    const ADDR_T ovf = bucketNum; // 溢出桶下标
    // 双保险: 调用点已保证只有宽档进来; 此处再挡一次, 但**不能依赖它**(见调用点注释)。
    // SyncAll 要求参与核集合完全一致; 窄档下各消费路径(尤其 SortScalarSliceSize1 的行归属)
    // 本就不做集合同步, 在此凭空插一次全局同步会让先到的核等一个永远不来的核 -> 设备侧挂死
    // (无 AICORE error, 表现为跑批 Timeout)。实测: fc_alloob_s1big_i64 / fc_crossbucket_flo64 复现。
    if (bucketNum <= 1) {
        return;
    }
    if (blockIdx_ == 0) {
        ADDR_T cnt[MAX_BUCKETS];
        ADDR_T off[MAX_BUCKETS];
        for (ADDR_T b = 0; b <= bucketNum; b++) {
            cnt[b] = 0;
        }
        for (ADDR_T k = 0; k < M; k++) { // 计数
            const ADDR_T r = RowOf(k);
            const ADDR_T b = (r < 0 || r >= varFirstDim) ? ovf : (r >> INDEX_LO_BITS);
            cnt[b] += 1;
        }
        ADDR_T acc = 0;
        for (ADDR_T b = 0; b <= bucketNum; b++) { // 前缀和 -> 每桶输出起点(含溢出桶, 排在最后)
            off[b] = acc;
            acc += cnt[b];
        }
        for (ADDR_T k = 0; k < M; k++) { // 按序 scatter: 同桶内保持原有(lo 升序)次序 => 稳定
            const ADDR_T r = RowOf(k);
            const ADDR_T b = (r < 0 || r >= varFirstDim) ? ovf : (r >> INDEX_LO_BITS);
            const ADDR_T dst = off[b];
            off[b] = dst + 1;
            scratchKeysGm_.SetValue(dst, sortedIdxGm_.GetValue(k));
            scratchPosGm_.SetValue(dst, originPosGm_.GetValue(k));
        }
        for (ADDR_T k = 0; k < M; k++) { // 写回
            sortedIdxGm_.SetValue(k, scratchKeysGm_.GetValue(k));
            originPosGm_.SetValue(k, scratchPosGm_.GetValue(k));
        }
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(sortedIdxGm_);
    }
    AscendC::SyncAll();
}

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortIndices(ADDR_T M)
{
    const ADDR_T blockN = static_cast<ADDR_T>(tiling_.blockNum);

    // ---------- small-M fast path: a single AscendC::Sort on core 0 ----------
    if (M <= static_cast<ADDR_T>(SORT_TILE)) {
        SortIndicesSmallM(M);
        // ⚠️ 调用点必须包在 wideIndex 判断里, **不能进函数再靠提前 return 绕开**:
        // StablePartitionByHi 末尾有 AscendC::SyncAll(), 而同步原语的对称性要在**调用层**保证 ——
        // 窄档各消费路径本就不做集合同步, 只要有核走了不同分支, 就会出现"部分核发了同步、
        // 部分核没发"→ 永久死等(无 AICORE error, 表现为跑批 Timeout)。
        // 实测: 函数内提前 return 仍挂(fc_overint32_s4_i32 等), 调用点包起来后 4/4 PASS。
        if (tiling_.wideIndex != 0) {
            StablePartitionByHi(M);
        }
        return;
    }

    // ---------- large M: parallel merge sort ----------
    ADDR_T P2 = 1; // number of runs = largest pow2 <= blockNum ...
    while (P2 * 2 <= blockN) {
        P2 *= 2;
    }
    while ((M + P2 - 1) / P2 > static_cast<ADDR_T>(SORT_TILE)) {
        P2 *= 2;
    } // ... raised so each chunk fits one Sort
    const ADDR_T runLen0 = (M + P2 - 1) / P2; // padded chunk length (every run is this long)
    ADDR_T nRounds = 0;
    {
        ADDR_T q = P2;
        while (q > 1) {
            q /= 2;
            nRounds++;
        }
    }
    // ping-pong A(scratch)=sel 0, B(sortedIdx/origin)=sel 1. Pick start so the last round lands in B.
    int curSel = ((nRounds & 1) == 0) ? 1 : 0;

    SortLocalRuns(M, P2, runLen0, curSel);
    SortMergeTree(M, P2, runLen0, nRounds, curSel);
    if (tiling_.wideIndex != 0) { // 同上: 同步对称性在调用层保证
        StablePartitionByHi(M);
    }
    // result is in buffer curSel == 1 (sortedIdx/originPos) by the start-parity choice.
}

// SortIndices small-M fast path: core 0 runs one AscendC::Sort, writes sortedIdx/originPos, then the
// collective DCCI + SyncAll (so every core's phase-3 scalar reads see fresh data).
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortIndicesSmallM(ADDR_T M)
{
    constexpr int32_t shiftOff = 32 / sizeof(int32_t);
    constexpr uint32_t SORT_PAD = 2 * 32;
    constexpr uint32_t tPad = (SORT_TILE + 31) / 32 * 32;
    if (blockIdx_ == 0) {
        TBuf<TPosition::VECCALC> bRaw, bI32, bSorted, bOrigin;
        pipe_.InitBuffer(bRaw, tPad * sizeof(INDICES_T) + SORT_PAD);
        pipe_.InitBuffer(bI32, tPad * sizeof(int32_t) + SORT_PAD);
        pipe_.InitBuffer(bSorted, (tPad + shiftOff) * sizeof(int32_t) + SORT_PAD);
        pipe_.InitBuffer(bOrigin, tPad * sizeof(uint32_t) + SORT_PAD);
        LocalTensor<INDICES_T> rawUb = bRaw.Get<INDICES_T>();
        LocalTensor<int32_t> i32Ub = bI32.Get<int32_t>();
        LocalTensor<uint32_t> originUb = bOrigin.Get<uint32_t>();
        LocalTensor<int32_t> shiftSorted = bSorted.Get<int32_t>()[shiftOff];
        DataCopyPadExtParams<INDICES_T> padIdx{false, 0, 0, 0};
        const event_t e2v = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t v2m3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        DataCopyExtParams cpIdx{1, static_cast<uint32_t>(M * sizeof(INDICES_T)), 0, 0, 0};
        if constexpr (sizeof(INDICES_T) == 8) {
            // int64 索引: 直接由原始缓冲构造 key(DIST_DINTLV_B32 取低字), **不走 Cast(int64->int32)**。
            DataCopyPad(rawUb, idxGm_, cpIdx, padIdx);
            SetFlag<HardEvent::MTE2_V>(e2v);
            WaitFlag<HardEvent::MTE2_V>(e2v);
            BuildSortKeyI64(i32Ub, (__local_mem__ int32_t*)rawUb.GetPhyAddr(), static_cast<uint32_t>(M),
                            static_cast<int32_t>(tiling_.varFirstDim), tiling_.wideIndex != 0);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(i32Ub, idxGm_, cpIdx, padIdx);
            SetFlag<HardEvent::MTE2_V>(e2v);
            WaitFlag<HardEvent::MTE2_V>(e2v);
            if (tiling_.wideIndex != 0) { // int32 索引 + 宽档: 值已是 32 位, 掩低位即精确桶内偏移
                MaskKeyToLoBits(i32Ub, static_cast<uint32_t>(M));
                PipeBarrier<PIPE_V>();
            }
        }
        AscendC::Sort<int32_t, true, scatterSortConfig>(shiftSorted, originUb, i32Ub, static_cast<uint32_t>(M));
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(M * sizeof(int32_t)), 0, 0, 0};
        SetFlag<HardEvent::V_MTE3>(v2m3);
        WaitFlag<HardEvent::V_MTE3>(v2m3);
        DataCopyPad(sortedIdxGm_, shiftSorted, cpOut);
        DataCopyPad(originPosGm_, originUb, cpOut);
    }
    // core 0 wrote sortedIdx/originPos via MTE3; phase 3 reads them with scalar GetValue on EVERY core.
    // Without flushing core 0's writes to GM and invalidating the other cores' stale cache lines, those
    // scalar reads can return stale data -> a few wrong originPos -> ~99% precision (same hazard the
    // merge-sort path guards at line 277; the single-core fast path needs it too).
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        sortedIdxGm_);
    AscendC::SyncAll();
}

// SortIndices large-M phase 2: each core sorts its runLen0-wide chunks (runs r = blockIdx_, blockIdx_+blockN,
// ...), pads each run's tail to runLen0 with SENT, writes them to the curSel buffer, then DCCI + SyncAll.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortLocalRuns(ADDR_T M, ADDR_T P2,
                                                                                           ADDR_T runLen0, int curSel)
{
    constexpr int32_t shiftOff = 32 / sizeof(int32_t);
    constexpr uint32_t SORT_PAD = 2 * 32;
    constexpr uint32_t tPad = (SORT_TILE + 31) / 32 * 32;
    constexpr int32_t SENT = 2147483647; // INT32_MAX padding key: sorts after every in-range index
    const ADDR_T blockN = static_cast<ADDR_T>(tiling_.blockNum);

    // ---- local sort: core handles runs r = blockIdx_, blockIdx_+blockN, ... ; pad tail to runLen0 with SENT ----
    {
        TBuf<TPosition::VECCALC> bRaw, bI32, bSorted, bOrigin;
        pipe_.InitBuffer(bRaw, tPad * sizeof(INDICES_T) + SORT_PAD);
        pipe_.InitBuffer(bI32, tPad * sizeof(int32_t) + SORT_PAD);
        pipe_.InitBuffer(bSorted, (tPad + shiftOff) * sizeof(int32_t) + SORT_PAD);
        pipe_.InitBuffer(bOrigin, tPad * sizeof(uint32_t) + SORT_PAD);
        LocalTensor<INDICES_T> rawUb = bRaw.Get<INDICES_T>();
        LocalTensor<int32_t> i32Ub = bI32.Get<int32_t>();
        LocalTensor<uint32_t> originUb = bOrigin.Get<uint32_t>();
        LocalTensor<int32_t> shiftSorted = bSorted.Get<int32_t>()[shiftOff];
        DataCopyPadExtParams<INDICES_T> padIdx{false, 0, 0, 0};
        const event_t e2v = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        const event_t v2s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        const event_t s2m3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        const event_t m3m2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
        // 循环尾除了 MTE3->MTE2(保护下一轮输入搬入), 还必须挡住下一轮对 shiftSorted/originUb 的写:
        // 下一轮用 V(Sort/Adds) 和 S(补 SENT) 写这两个 UB, 而本轮的 DataCopyPad(MTE3) 仍在读它们。
        // 只发 MTE3->MTE2 只同步了 MTE2 管道, V/S 会跑在前面覆盖缓冲 -> 写出坏的排序键/位置。
        const event_t m3v = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
        const event_t m3s2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
        for (ADDR_T r = static_cast<ADDR_T>(blockIdx_); r < P2; r += blockN) {
            const ADDR_T cs = r * runLen0;
            const ADDR_T cl = (cs >= M) ? 0 : ((M - cs < runLen0) ? (M - cs) : runLen0);
            if (cl > 0) {
                DataCopyExtParams cpIdx{1, static_cast<uint32_t>(cl * sizeof(INDICES_T)), 0, 0, 0};
                if constexpr (sizeof(INDICES_T) == 8) {
                    // 同 SortIndicesSmallM: int64 索引由原始缓冲直接构造 key, 不经 Cast。
                    DataCopyPad(rawUb, idxGm_[cs], cpIdx, padIdx);
                    SetFlag<HardEvent::MTE2_V>(e2v);
                    WaitFlag<HardEvent::MTE2_V>(e2v);
                    BuildSortKeyI64(i32Ub, (__local_mem__ int32_t*)rawUb.GetPhyAddr(), static_cast<uint32_t>(cl),
                                    static_cast<int32_t>(tiling_.varFirstDim), tiling_.wideIndex != 0);
                    PipeBarrier<PIPE_V>();
                } else {
                    DataCopyPad(i32Ub, idxGm_[cs], cpIdx, padIdx);
                    SetFlag<HardEvent::MTE2_V>(e2v);
                    WaitFlag<HardEvent::MTE2_V>(e2v);
                    if (tiling_.wideIndex != 0) {
                        MaskKeyToLoBits(i32Ub, static_cast<uint32_t>(cl));
                        PipeBarrier<PIPE_V>();
                    }
                }
                AscendC::Sort<int32_t, true, scatterSortConfig>(shiftSorted, originUb, i32Ub,
                                                                static_cast<uint32_t>(cl));
                Adds(originUb.template ReinterpretCast<int32_t>(), originUb.template ReinterpretCast<int32_t>(),
                     static_cast<int32_t>(cs), static_cast<int32_t>(cl));
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_S>(v2s);
            WaitFlag<HardEvent::V_S>(v2s);
            for (ADDR_T p = cl; p < runLen0; p++) {
                shiftSorted.SetValue(p, SENT);
                originUb.SetValue(p, 0u);
            }
            DataCopyExtParams cpOut{1, static_cast<uint32_t>(runLen0 * sizeof(int32_t)), 0, 0, 0};
            SetFlag<HardEvent::S_MTE3>(s2m3);
            WaitFlag<HardEvent::S_MTE3>(s2m3);
            if (curSel == 1) {
                DataCopyPad(sortedIdxGm_[cs], shiftSorted, cpOut);
                DataCopyPad(originPosGm_[cs], originUb, cpOut);
            } else {
                DataCopyPad(scratchKeysGm_[cs], shiftSorted, cpOut);
                DataCopyPad(scratchPosGm_[cs], originUb, cpOut);
            }
            SetFlag<HardEvent::MTE3_MTE2>(m3m2);
            WaitFlag<HardEvent::MTE3_MTE2>(m3m2);
            // 本轮 MTE3 读完 shiftSorted/originUb 之前, 不许下一轮的 V(Sort/Adds) 和 S(补SENT) 覆写它们
            SetFlag<HardEvent::MTE3_V>(m3v);
            WaitFlag<HardEvent::MTE3_V>(m3v);
            SetFlag<HardEvent::MTE3_S>(m3s2);
            WaitFlag<HardEvent::MTE3_S>(m3s2);
        }
    }
    // flush this core's runs to GM + invalidate stale lines so other cores' scalar reads next see fresh data
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        sortedIdxGm_);
    AscendC::SyncAll();
}

// SortIndices large-M merge tree: each round, each core produces its own slice [s*runLen0,(s+1)*runLen0) of
// the round output via a merge-path co-rank, ping-pongs the curSel buffer, DCCI + SyncAll per round.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortMergeTree(ADDR_T M, ADDR_T P2,
                                                                                           ADDR_T runLen0,
                                                                                           ADDR_T nRounds, int curSel)
{
    constexpr uint32_t SORT_PAD = 2 * 32;
    constexpr uint32_t tPad = (SORT_TILE + 31) / 32 * 32;
    constexpr int32_t SENT = 2147483647; // INT32_MAX padding key: sorts after every in-range index
    const ADDR_T blockN = static_cast<ADDR_T>(tiling_.blockNum);

    // ---- merge tree: each round, each core produces its slice [s*runLen0,(s+1)*runLen0) via merge-path ----
    TBuf<TPosition::VECCALC> bOK, bOP;
    pipe_.InitBuffer(bOK, tPad * sizeof(int32_t) + SORT_PAD);
    pipe_.InitBuffer(bOP, tPad * sizeof(uint32_t) + SORT_PAD);
    LocalTensor<int32_t> okUb = bOK.Get<int32_t>();
    LocalTensor<uint32_t> opUb = bOP.Get<uint32_t>();
    const event_t s2m3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
    const event_t m3s = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
    for (ADDR_T rr = 0; rr < nRounds; rr++) {
        const ADDR_T curRunLen = runLen0 << rr;
        const ADDR_T pairOut = curRunLen * 2;
        GlobalTensor<int32_t> ck = (curSel == 1) ? sortedIdxGm_ : scratchKeysGm_;
        GlobalTensor<uint32_t> cp = (curSel == 1) ? originPosGm_ : scratchPosGm_;
        for (ADDR_T s = static_cast<ADDR_T>(blockIdx_); s < P2; s += blockN) {
            const ADDR_T oBase = s * runLen0;
            const ADDR_T pairBase = (oBase / pairOut) * pairOut;
            const ADDR_T leftBase = pairBase;
            const ADDR_T rightBase = pairBase + curRunLen;
            const ADDR_T localLo = oBase - pairBase;
            // merge-path co-rank: largest i in [iMin,iMax] with left[i-1] <= right[localLo-i]
            ADDR_T iMin = (localLo > curRunLen) ? (localLo - curRunLen) : 0;
            ADDR_T iMax = (localLo < curRunLen) ? localLo : curRunLen;
            while (iMin < iMax) {
                const ADDR_T mid = (iMin + iMax + 1) / 2;
                const int32_t lp = ck.GetValue(leftBase + mid - 1);
                const ADDR_T rj = localLo - mid;
                const int32_t rv = (rj >= curRunLen) ? SENT : ck.GetValue(rightBase + rj);
                if (lp <= rv) {
                    iMin = mid;
                } else {
                    iMax = mid - 1;
                }
            }
            ADDR_T li = leftBase + iMin;
            ADDR_T ri = rightBase + (localLo - iMin);
            const ADDR_T lend = leftBase + curRunLen;
            const ADDR_T rend = rightBase + curRunLen;
            for (ADDR_T o = 0; o < runLen0; o++) {
                bool takeL;
                if (li >= lend) {
                    takeL = false;
                } else if (ri >= rend) {
                    takeL = true;
                } else {
                    takeL = (ck.GetValue(li) <= ck.GetValue(ri));
                }
                if (takeL) {
                    okUb.SetValue(o, ck.GetValue(li));
                    opUb.SetValue(o, cp.GetValue(li));
                    li++;
                } else {
                    okUb.SetValue(o, ck.GetValue(ri));
                    opUb.SetValue(o, cp.GetValue(ri));
                    ri++;
                }
            }
            DataCopyExtParams cpS{1, static_cast<uint32_t>(runLen0 * sizeof(int32_t)), 0, 0, 0};
            SetFlag<HardEvent::S_MTE3>(s2m3);
            WaitFlag<HardEvent::S_MTE3>(s2m3);
            if (curSel == 1) { // out is the other buffer (scratch)
                DataCopyPad(scratchKeysGm_[oBase], okUb, cpS);
                DataCopyPad(scratchPosGm_[oBase], opUb, cpS);
            } else {
                DataCopyPad(sortedIdxGm_[oBase], okUb, cpS);
                DataCopyPad(originPosGm_[oBase], opUb, cpS);
            }
            SetFlag<HardEvent::MTE3_S>(m3s);
            WaitFlag<HardEvent::MTE3_S>(m3s);
        }
        // flush this round's writes to GM + invalidate, so the next round's cross-core scalar reads are fresh
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(sortedIdxGm_);
        curSel ^= 1;
        AscendC::SyncAll();
    }
}
// 本核是否整体接管该段。段完整落在范围内自然整体处理; MUL 浮点路径则一律整体接管(行归属),
// 因为 updates 单独连乘会脱离 var 的量级约束而溢出 —— 同 SortDivRowProcess 的取舍。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TakeSegmentWhole(ADDR_T segEnd,
                                                                                              ADDR_T posEnd)
{
    if constexpr (Mode == MODE_MUL && AscendC::IsSameType<AccT, float>::value) {
        return true;
    }
    return segEnd <= posEnd;
}

// 取较小值(避免三目表达式占行, 且语义直白)。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline ADDR_T ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::MinAddr(ADDR_T a, ADDR_T b)
{
    return (a < b) ? a : b;
}

// MUL 浮点走行归属时, 左边缘那段由前一个核整体接管, 本核直接跳过它; 其余 Mode 原样返回。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline ADDR_T ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SkipSpilledLeftEdge(ADDR_T k, ADDR_T M)
{
    if constexpr (Mode == MODE_MUL && AscendC::IsSameType<AccT, float>::value) {
        while (k > 0 && k < M && RowOf(k) == RowOf(k - 1)) {
            k++;
        }
    }
    return k;
}

// Sort-based position-split reducer for MUL/MAX/MIN (all three are associative+commutative, so a row's
// updates can be split across cores and combined). Phase1 (SortIndices) groups equal indices contiguously.
// Phase3: every core walks the sorted array, owns the segments whose start falls in its count-balanced
// position range [blockIdx*M/P,(blockIdx+1)*M/P), and for each owned segment (one var row) folds that row's
// updates (MUL: product; MAX/MIN: max/min) into var[row] -- a single contention-free write. A segment that
// spills past the range boundary is split: each covering core writes an in-range partial, the owner folds
// them (3b). The count-balanced split is what keeps each core's segment walk O(M/P) (not O(M)) and spreads
// a hot all-to-one row across cores -- the two things the DIV column/row split cannot both do.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortReduceProcess(ADDR_T M,
                                                                                               ADDR_T sliceSize,
                                                                                               ADDR_T varFirstDim)
{
    // ACC: float for fp16 (match golden fp32 intermediate); native for fp32/int32; int32 for int8/uint8.
    // The int8/uint8 widening is forced by HARDWARE -- vmul has no 8-bit form -- not by math (mod 256 is
    // preserved through int32 wrapping). int8/uint8 widen/narrow via Reg (high-level Cast i8<->i32 is
    // broken on this device); fp16 via Cast; fp32/int32 are native.
    using ACC = AccT;
    SortIndices(M);
    pipe_.Reset(); // free phase-1 sort/merge UB before the phase-3 buffers (else they coexist -> UB overflow)

    // -------- Phase 3: every core reduces its count-balanced segments --------
    const ADDR_T posBegin = blockIdx_ * M / static_cast<ADDR_T>(tiling_.blockNum);
    const ADDR_T posEnd = (blockIdx_ + 1) * M / static_cast<ADDR_T>(tiling_.blockNum);
    // Under the tiling invariant blockNum <= M every core owns >=1 sorted position (posBegin < posEnd). We do
    // NOT early-return on an empty range: the per-chunk SyncAll barriers below are collective, so a core that
    // bailed here would hang every core that still has work. If a future tiling change ever breaks the
    // invariant, ownsRange keeps the empty core OUT of the fold (which would otherwise emit a spurious
    // left-edge partial) but still INSIDE every barrier -- degrading to alive-but-wrong, never to deadlock.
    const bool ownsRange = posBegin < posEnd;
    // Deep-prefetch double-buffer for the update-row loads (scattered, latency-bound: keep PREFETCH_DEPTH
    // DMAs in flight). The product is PER-COLUMN independent, so a wide row is processed in column chunks of
    // CHUNK = min(sliceSize, CHUNK_MAX), where CHUNK_MAX is the widest chunk that fits the phase-3 UB budget.
    // This makes sliceSize unbounded -- the chunk is a UB-derived tile, not a cap.
    // 列切分上限由 host 按 UB 实测容量反解后下发(见 tiling 的 ubChunkMax), kernel 只做取小。
    const ADDR_T CHUNK = MinAddr(sliceSize, static_cast<ADDR_T>(tiling_.ubChunkMax));
    const uint32_t cPad = (static_cast<uint32_t>(CHUNK) + 31) / 32 * 32;
    const ADDR_T sAlign = (CHUNK + 7) / 8 * 8; // per-core partial slot stride (one chunk wide)
    TBuf<TPosition::VECCALC> bAcc, bVar, bRowAcc;
    TQue<QuePosition::VECIN, PREFETCH_DEPTH> updQue;
    pipe_.InitBuffer(bAcc, cPad * sizeof(ACC));
    pipe_.InitBuffer(bVar, cPad * sizeof(PARAMS_T));
    pipe_.InitBuffer(bRowAcc, cPad * sizeof(ACC));
    pipe_.InitBuffer(updQue, PREFETCH_DEPTH, cPad * sizeof(PARAMS_T));
    LocalTensor<ACC> accUb = bAcc.Get<ACC>();
    LocalTensor<PARAMS_T> varUb = bVar.Get<PARAMS_T>();
    LocalTensor<ACC> rowAccUb = bRowAcc.Get<ACC>();
    // MAX/MIN int8 needs a signed-extend scratch (the int32 widen is zero-extended).
    TBuf<TPosition::VECCALC> bTmp;
    pipe_.InitBuffer(bTmp, cPad * sizeof(int32_t));
    LocalTensor<int32_t> tmp = bTmp.Get<int32_t>();
    const event_t evMV0 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
    const event_t evVM3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    const event_t evM3M2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
    const event_t evVM2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
    DataCopyPadExtParams<PARAMS_T> pad{false, 0, 0, 0};
    DataCopyPadExtParams<ACC> padAcc{false, 0, 0, 0};

    // every core runs the SAME chunk count (sliceSize/CHUNK is tiling-derived) so the per-chunk SyncAll
    // stays collective; blockNum <= M guarantees posBegin < posEnd, so no core skips the barrier.
    for (ADDR_T c0 = 0; c0 < sliceSize; c0 += CHUNK) {
        const ADDR_T chunkW = (sliceSize - c0 < CHUNK) ? (sliceSize - c0) : CHUNK;
        DataCopyExtParams cp{1, static_cast<uint32_t>(chunkW * sizeof(PARAMS_T)), 0, 0, 0};
        DataCopyExtParams cpAcc{1, static_cast<uint32_t>(chunkW * sizeof(ACC)), 0, 0, 0};
        ADDR_T ownerSplitRow = -1;
        ADDR_T ownerSplitEnd = 0;
        // empty-range core (only reachable if blockNum <= M is ever violated): skip the fold but still hit the
        // barriers below. ownerSplitRow stays -1, so SortPhase3bCombine is a no-op for it too.
        if (ownsRange) {
            SortPhase3aFold(posBegin, posEnd, M, sliceSize, varFirstDim, c0, chunkW, sAlign, accUb, varUb, rowAccUb,
                            tmp, updQue, evMV0, evVM3, evM3M2, cp, cpAcc, pad, ownerSplitRow, ownerSplitEnd);
        }
        // flush this core's MTE3 partial writes to GM before the barrier, so the owner core's MTE2 reads in
        // 3b see fresh data (same cross-core consistency guard as SortIndices' sortedIdxGm_ flush).
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(partialGm_);
        AscendC::SyncAll(); // #2: every contributing core's partial for this chunk is in the workspace
        SortPhase3bCombine(M, sliceSize, c0, chunkW, sAlign, ownerSplitRow, ownerSplitEnd, accUb, varUb, rowAccUb, tmp,
                           evMV0, evVM3, evM3M2, evVM2, cp, cpAcc, pad, padAcc);
        // barrier before the next chunk reuses the partial workspace: this chunk's 3b reads of partialGm_
        // must finish before the next chunk's 3a overwrites the same per-core slots (different cores). The
        // last chunk has no successor, so skip it (single-chunk cases pay no extra barrier).
        if (c0 + CHUNK < sliceSize) {
            AscendC::SyncAll();
        }
    }
}

// SortReduceProcess Phase 3a: in-range fold for columns [c0,c0+chunkW). Whole segments fold var-first and
// write var; a segment that spans this core's right boundary writes a pure in-range partial to
// partialGm_[blockIdx] and returns its (row,end) in ownerSplitRow/ownerSplitEnd for phase 3b. The left-edge
// 左边缘段(起点落在前一个核的范围内)的折叠: 本核只把范围内的部分算成 partial 落 workspace,
// 由持有该段起点的核在 3b 合并。抽出以免 SortPhase3aFold 继续膨胀。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::FoldSpilledLeftEdge(
    ADDR_T& k, ADDR_T posEnd, ADDR_T varFirstDim, ADDR_T sliceSize, ADDR_T c0, ADDR_T chunkW, ADDR_T sAlign,
    LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
    TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, event_t evMV0, event_t evVM3, event_t evM3M2,
    const DataCopyExtParams& cp, const DataCopyExtParams& cpAcc, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    using ACC = AccT;
    ADDR_T r = RowOf(k);
    ADDR_T s = k;
    while (k < posEnd && RowOf(k) == r) {
        k++;
    }
    if (r >= 0 && static_cast<ADDR_T>(r) < varFirstDim) {
        uint32_t p0 = originPosGm_.GetValue(s);
        LoadWiden(accUb, varUb, xGm_, static_cast<ADDR_T>(p0) * sliceSize + c0, chunkW, evMV0, cp, pad);
        if constexpr ((Mode == MODE_MAX || Mode == MODE_MIN) && AscendC::IsSameType<PARAMS_T, int8_t>::value) {
            ShiftRight(tmp, accUb, INT8_SIGN_SHIFT, static_cast<int32_t>(chunkW));
            Muls(tmp, tmp, INT8_SIGN_SPAN, static_cast<int32_t>(chunkW));
            Sub(accUb, accUb, tmp, static_cast<int32_t>(chunkW));
            PipeBarrier<PIPE_V>();
        }
        if constexpr (Mode == MODE_MUL) {
            MulRangeIntoAcc(accUb, rowAccUb, tmp, updQue, s + 1, k, sliceSize, c0, chunkW, cp, pad);
        } else {
            MaxMinRangeIntoAcc(accUb, rowAccUb, tmp, updQue, s + 1, k, sliceSize, c0, chunkW, cp, pad);
        }
        SetFlag<HardEvent::V_MTE3>(evVM3);
        WaitFlag<HardEvent::V_MTE3>(evVM3); // pure partial, no narrow
        DataCopyPad(partialGm_[static_cast<ADDR_T>(blockIdx_) * sAlign].template ReinterpretCast<ACC>(), accUb, cpAcc);
        SetFlag<HardEvent::MTE3_MTE2>(evM3M2);
        WaitFlag<HardEvent::MTE3_MTE2>(evM3M2); // store before reuse
    }
}

// segment that spilled in from before posBegin writes a partial too (owned by the earlier core).
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortPhase3aFold(
    ADDR_T posBegin, ADDR_T posEnd, ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim, ADDR_T c0, ADDR_T chunkW,
    ADDR_T sAlign, LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb, LocalTensor<AccT>& rowAccUb,
    LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, event_t evMV0, event_t evVM3,
    event_t evM3M2, const DataCopyExtParams& cp, const DataCopyExtParams& cpAcc,
    const DataCopyPadExtParams<PARAMS_T>& pad, ADDR_T& ownerSplitRow, ADDR_T& ownerSplitEnd)
{
    using ACC = AccT;
    // -------- Phase 3a: in-range product for columns [c0,c0+chunkW). Whole segments -> var; segments
    // spanning a core boundary -> partial to partialGm_[core], combined by the owner in phase 3b. --------
    // MUL 浮点走行归属: 左边缘段整体归前一个核, 本核直接跳过(见 TakeSegmentWhole 的说明)。
    ADDR_T k = SkipSpilledLeftEdge(posBegin, M);
    if (k > 0 && k < posEnd && RowOf(k) == RowOf(k - 1)) { // left edge spilled in
        FoldSpilledLeftEdge(k, posEnd, varFirstDim, sliceSize, c0, chunkW, sAlign, accUb, varUb, rowAccUb, tmp, updQue,
                            evMV0, evVM3, evM3M2, cp, cpAcc, pad);
    }
    while (k < posEnd && k < M) {
        ADDR_T r = RowOf(k);
        ADDR_T segStart = k;
        while (k < M && RowOf(k) == r) {
            k++;
        }
        ADDR_T segEnd = k;
        if (r < 0 || static_cast<ADDR_T>(r) >= varFirstDim) {
            if (segEnd > posEnd) {
                break;
            }
            continue;
        }
        const ADDR_T base = static_cast<ADDR_T>(r) * sliceSize + c0;
        // MUL 浮点采用行归属: 跨出本核范围的段也由起始核整体折叠(var-first), 不拆成跨核 partial。
        // 理由与 SortDivRowProcess 相同 —— updates 单独连乘会冲出 fp32 范围("the divisor product
        // overflows"), 而 var 从头参与时中间值受其量级约束。拆分虽能分摊热点行, 但会把本可表示的
        // 结果算成 ±inf, 与参考实现(var-first)的折叠顺序产生系统性分歧。
        if (TakeSegmentWhole(segEnd, posEnd)) { // var-first fold, write var
            LoadWiden(accUb, varUb, outputGm_, base, chunkW, evMV0, cp, pad);
            if constexpr ((Mode == MODE_MAX || Mode == MODE_MIN) && AscendC::IsSameType<PARAMS_T, int8_t>::value) {
                ShiftRight(tmp, accUb, INT8_SIGN_SHIFT, static_cast<int32_t>(chunkW));
                Muls(tmp, tmp, INT8_SIGN_SPAN, static_cast<int32_t>(chunkW));
                Sub(accUb, accUb, tmp, static_cast<int32_t>(chunkW));
                PipeBarrier<PIPE_V>();
            }
            if constexpr (Mode == MODE_MUL) {
                MulRangeIntoAcc(accUb, rowAccUb, tmp, updQue, segStart, segEnd, sliceSize, c0, chunkW, cp, pad);
            } else {
                MaxMinRangeIntoAcc(accUb, rowAccUb, tmp, updQue, segStart, segEnd, sliceSize, c0, chunkW, cp, pad);
            }
            NarrowStore(accUb, varUb, outputGm_, base, chunkW, evVM3, cp);
            // the store (MTE3) reads accUb/varUb; the NEXT segment's load (MTE2) reuses them -> wait
            SetFlag<HardEvent::MTE3_MTE2>(evM3M2);
            WaitFlag<HardEvent::MTE3_MTE2>(evM3M2);
            if (segEnd > posEnd)
                break; // 该段已整体处理完, 后续位置归下一个核
        } else {       // owner of a split row -> pure in-range partial, finish in 3b
            uint32_t p0 = originPosGm_.GetValue(segStart);
            LoadWiden(accUb, varUb, xGm_, static_cast<ADDR_T>(p0) * sliceSize + c0, chunkW, evMV0, cp, pad);
            if constexpr ((Mode == MODE_MAX || Mode == MODE_MIN) && AscendC::IsSameType<PARAMS_T, int8_t>::value) {
                ShiftRight(tmp, accUb, INT8_SIGN_SHIFT, static_cast<int32_t>(chunkW));
                Muls(tmp, tmp, INT8_SIGN_SPAN, static_cast<int32_t>(chunkW));
                Sub(accUb, accUb, tmp, static_cast<int32_t>(chunkW));
                PipeBarrier<PIPE_V>();
            }
            if constexpr (Mode == MODE_MUL) {
                MulRangeIntoAcc(accUb, rowAccUb, tmp, updQue, segStart + 1, posEnd, sliceSize, c0, chunkW, cp, pad);
            } else {
                MaxMinRangeIntoAcc(accUb, rowAccUb, tmp, updQue, segStart + 1, posEnd, sliceSize, c0, chunkW, cp, pad);
            }
            ownerSplitRow = r;
            ownerSplitEnd = segEnd;
            break;
        }
    }
}
// SortReduceProcess Phase 3b: the owner of a split row folds the covering cores' in-range partials (blockIdx+1
// .. while their range start is inside the segment) plus var[r], then narrow-stores var[r]. A no-op when this
// core did not own a split row.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortPhase3bCombine(
    ADDR_T M, ADDR_T sliceSize, ADDR_T c0, ADDR_T chunkW, ADDR_T sAlign, ADDR_T ownerSplitRow, ADDR_T ownerSplitEnd,
    LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
    event_t evMV0, event_t evVM3, event_t evM3M2, event_t evVM2, const DataCopyExtParams& cp,
    const DataCopyExtParams& cpAcc, const DataCopyPadExtParams<PARAMS_T>& pad, const DataCopyPadExtParams<AccT>& padAcc)
{
    using ACC = AccT;
    // -------- Phase 3b: owner folds the covering cores' partials (blockIdx+1.. while their range start is
    // inside the segment -- direct test, the segEnd*P/M inverse is off by one at floor boundaries) + var --
    if (ownerSplitRow >= 0) {
        const ADDR_T r = ownerSplitRow;
        const ADDR_T base = static_cast<ADDR_T>(r) * sliceSize + c0;
        const ADDR_T bn = static_cast<ADDR_T>(tiling_.blockNum);
        for (ADDR_T c = static_cast<ADDR_T>(blockIdx_) + 1; c < bn && c * M / bn < ownerSplitEnd; c++) {
            DataCopyPad(rowAccUb, partialGm_[c * sAlign].template ReinterpretCast<ACC>(), cpAcc, padAcc);
            SetFlag<HardEvent::MTE2_V>(evMV0);
            WaitFlag<HardEvent::MTE2_V>(evMV0);
            // combine this covering core's partial (already in the signed ACC, so no re-sign-extend)
            if constexpr (Mode == MODE_MUL) {
                Mul(accUb, accUb, rowAccUb, static_cast<int32_t>(chunkW));
            } else if constexpr (Mode == MODE_MAX) {
                Max(accUb, accUb, rowAccUb, static_cast<int32_t>(chunkW));
            } else {
                Min(accUb, accUb, rowAccUb, static_cast<int32_t>(chunkW));
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(evVM2);
            WaitFlag<HardEvent::V_MTE2>(evVM2); // combine read before reload
        }
        LoadWiden(rowAccUb, varUb, outputGm_, base, chunkW, evMV0, cp, pad); // fold in var[r]
        if constexpr ((Mode == MODE_MAX || Mode == MODE_MIN) && AscendC::IsSameType<PARAMS_T, int8_t>::value) {
            ShiftRight(tmp, rowAccUb, INT8_SIGN_SHIFT, static_cast<int32_t>(chunkW));
            Muls(tmp, tmp, INT8_SIGN_SPAN, static_cast<int32_t>(chunkW));
            Sub(rowAccUb, rowAccUb, tmp, static_cast<int32_t>(chunkW));
            PipeBarrier<PIPE_V>();
        }
        if constexpr (Mode == MODE_MUL) {
            Mul(accUb, accUb, rowAccUb, static_cast<int32_t>(chunkW));
        } else if constexpr (Mode == MODE_MAX) {
            Max(accUb, accUb, rowAccUb, static_cast<int32_t>(chunkW));
        } else {
            Min(accUb, accUb, rowAccUb, static_cast<int32_t>(chunkW));
        }
        PipeBarrier<PIPE_V>();
        NarrowStore(accUb, varUb, outputGm_, base, chunkW, evVM3, cp);
        // the store (MTE3) reads accUb/varUb; the next chunk's loads (MTE2) reuse them -> wait
        SetFlag<HardEvent::MTE3_MTE2>(evM3M2);
        WaitFlag<HardEvent::MTE3_MTE2>(evM3M2);
    }
}

// Deep-prefetch sequential divide: accUb /= updates[originPos[startJ..endJ)] in sorted order, in the ACC
// type (float for fp16/fp32, int32 for int8/uint8/int32 -- the same ACC as MUL, so LoadWiden/NarrowStore are
// reused). Two reference-matching subtleties:
//   - INTEGER division must be EXACT trunc-toward-zero (the reference uses int64, so fp32 would lose
//     precision above 2^24); the hardware int32 vector Div is exact integer division, so divide natively.
//   - INTEGER /0 must KEEP var unchanged (the reference's choice for the C++ UB). Each 0 divisor is replaced
//     by 1 (var/1 = var) via safe = upd + (upd==0). With /0 a no-op, integer division is order-independent
//     (a theorem), so the sorted order matches the reference bit-for-bit.
// FLOAT /0 -> +-inf exactly like the reference, so no divisor fix-up. Either way var is divided directly --
// no product is formed (it would overflow where the reference's sequential divide stays bounded).
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::DivRangeIntoVar(
    LocalTensor<AccT>& accUb, LocalTensor<AccT>& updAcc, LocalTensor<int32_t>& tmp,
    TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff,
    ADDR_T width, const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad)
{
    if (startJ >= endJ) {
        return;
    }
    constexpr ADDR_T AHEAD = static_cast<ADDR_T>(PREFETCH_AHEAD); // 留一笔给正在相除的行
    const ADDR_T cnt = endJ - startJ;
    const ADDR_T primed = (cnt < AHEAD) ? cnt : AHEAD;
    PrimePrefetch(updQue, startJ, primed, rowStride, colOff, cp, pad); // origin read from GM (O(M/P) walk)
    for (ADDR_T j = 0; j < cnt; j++) {
        LocalTensor<PARAMS_T> curRaw = updQue.template DeQue<PARAMS_T>();
        PrefetchAhead(updQue, startJ, j, cnt, AHEAD, rowStride, colOff, cp, pad);
        if constexpr (AscendC::IsSameType<AccT, int32_t>::value) { // int8/uint8/int32: exact int32 Div
            if constexpr (sizeof(PARAMS_T) == 1) {
                SubwordWidenToI32(updAcc, curRaw, static_cast<uint32_t>(width)); // int8/uint8 -> int32 (zero-ext)
                if constexpr (AscendC::IsSameType<PARAMS_T, int8_t>::value) {
                    PipeBarrier<PIPE_V>(); // sign-extend int8: v-=256*(v>>7)
                    ShiftRight(tmp, updAcc, INT8_SIGN_SHIFT, static_cast<int32_t>(width));
                    Muls(tmp, tmp, INT8_SIGN_SPAN, static_cast<int32_t>(width));
                    Sub(updAcc, updAcc, tmp, static_cast<int32_t>(width));
                }
            } else {
                Adds(updAcc, curRaw, 0, static_cast<int32_t>(width)); // int32: copy out of the queue
            }
            PipeBarrier<PIPE_V>();
            // 除数判零用精确等值比较 + Select(同 ops-math truncate_div/mod 的做法)。不可用 Abs 等算术
            // 变换: |INT_MIN| 不可表示, Abs 溢出回绕后判零失效, 除数会被误换成 1。
            // 掩码借 tmp 的存储: int8 符号扩展已用完 tmp, 二者不重叠。
            LocalTensor<uint8_t> zmask = tmp.template ReinterpretCast<uint8_t>();
            CompareScalar(zmask, updAcc, static_cast<AccT>(0), CMPMODE::NE, static_cast<int32_t>(width));
            PipeBarrier<PIPE_V>();
            Select(updAcc, zmask, updAcc, static_cast<AccT>(1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   static_cast<int32_t>(width)); // upd!=0 保留, ==0 换成 1(除完即原值, 契约: 保持 var)
            PipeBarrier<PIPE_V>();
            Div(accUb, accUb, updAcc, static_cast<int32_t>(width));     // exact int32 trunc; /0 kept var
        } else {                                                        // fp16/fp32: float Div, /0 -> inf
            if constexpr (AscendC::IsSameType<PARAMS_T, AccT>::value) { // fp32
                Div<AccT, PRECISE_DIV>(accUb, accUb, curRaw, static_cast<int32_t>(width));
            } else { // fp16 -> float
                Cast(updAcc, curRaw, RoundMode::CAST_NONE, static_cast<int32_t>(width));
                PipeBarrier<PIPE_V>();
                Div<AccT, PRECISE_DIV>(accUb, accUb, updAcc, static_cast<int32_t>(width));
            }
        }
        PipeBarrier<PIPE_V>();
        updQue.FreeTensor(curRaw);
    }
}

// Sort-based ROW-split reducer for DIV. Phase1 (SortIndices) groups equal indices. Phase3: every core owns
// the segments whose START falls in its count-balanced position range [blockIdx*M/P,(blockIdx+1)*M/P) and
// divides var[row] by that row's updates IN PLACE across the FULL slice (column-tiled). A row's whole segment
// is owned by ONE core -- DIV cannot split a divide chain across cores (that needs the divisor product, which
// overflows), so a segment extending past posEnd is taken WHOLE by its starting core, and a segment spilled in
// from before posBegin belongs to the earlier core and is skipped. Unlike the old column-split (every core
// walked all M segments -- O(M) scalar, catastrophic at large M, and narrow slices left most cores idle),
// each core walks only its O(M/P) positions. No atomics, no cross-core combine, no barrier. A single
// all-to-one hot row still serialises on its one owning core (the divide chain is fundamentally sequential).
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortDivRowProcess(ADDR_T M,
                                                                                               ADDR_T sliceSize,
                                                                                               ADDR_T varFirstDim)
{
    using ACC = AccT;
    SortIndices(M); // integer /0 keeps var (a no-op) so DIV is order-independent -- no within-segment sort
    pipe_.Reset();  // free phase-1 sort/merge UB before the phase-3 buffers (else they coexist -> UB overflow)

    const ADDR_T bn = static_cast<ADDR_T>(tiling_.blockNum);
    const ADDR_T posBegin = static_cast<ADDR_T>(blockIdx_) * M / bn; // count-balanced position range
    const ADDR_T posEnd = (static_cast<ADDR_T>(blockIdx_) + 1) * M / bn;
    if (posBegin >= posEnd) {
        return;
    }
    // 列切分上限同样取 host 下发值(见 tiling 的 ubChunkMax), kernel 只做取小。
    const ADDR_T CHUNK = MinAddr(sliceSize, static_cast<ADDR_T>(tiling_.ubChunkMax));
    const uint32_t cPad = (static_cast<uint32_t>(CHUNK) + 31) / 32 * 32;
    // The sorted index / originPos arrays are read straight from GM (sortedIdxGm_/originPosGm_) -- the row-split
    // walk only touches this core's O(M/P) positions, so UB staging (which the old column-split needed for its
    // O(M) per-core walk) buys nothing and would cost M*8 UB, breaking the single-core sort capacity at large M.
    TBuf<TPosition::VECCALC> bAcc, bVar, bUpd, bTmp;
    TQue<QuePosition::VECIN, PREFETCH_DEPTH> updQue;
    pipe_.InitBuffer(bAcc, cPad * sizeof(ACC));
    pipe_.InitBuffer(bVar, cPad * sizeof(PARAMS_T));
    pipe_.InitBuffer(bUpd, cPad * sizeof(ACC));     // widened update (and 0->1 safe divisor for int)
    pipe_.InitBuffer(bTmp, cPad * sizeof(int32_t)); // int 0->1 / sign-extend scratch (unused for float)
    pipe_.InitBuffer(updQue, PREFETCH_DEPTH, cPad * sizeof(PARAMS_T));
    LocalTensor<ACC> accUb = bAcc.Get<ACC>();
    LocalTensor<PARAMS_T> varUb = bVar.Get<PARAMS_T>();
    LocalTensor<ACC> updAcc = bUpd.Get<ACC>();
    LocalTensor<int32_t> tmp = bTmp.Get<int32_t>();
    const event_t evMV0 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
    const event_t evVM3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    const event_t evM3M2 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
    DataCopyPadExtParams<PARAMS_T> pad{false, 0, 0, 0};

    for (ADDR_T c0 = 0; c0 < sliceSize; c0 += CHUNK) {
        const ADDR_T chunkW = (sliceSize - c0 < CHUNK) ? (sliceSize - c0) : CHUNK;
        DataCopyExtParams cp{1, static_cast<uint32_t>(chunkW * sizeof(PARAMS_T)), 0, 0, 0};
        // own segments whose START is in [posBegin,posEnd): skip a segment spilled in from before posBegin
        // (it belongs to the earlier core), then process each owned segment WHOLE (even if it runs past posEnd).
        ADDR_T k = posBegin;
        if (k > 0 && RowOf(k) == RowOf(k - 1)) {
            ADDR_T rs = RowOf(k);
            while (k < M && RowOf(k) == rs) {
                k++;
            }
        }
        while (k < posEnd && k < M) {
            ADDR_T r = RowOf(k);
            ADDR_T segStart = k;
            while (k < M && RowOf(k) == r) {
                k++;
            } // segEnd may exceed posEnd: owned whole
            ADDR_T segEnd = k;
            if (r < 0 || static_cast<ADDR_T>(r) >= varFirstDim) {
                continue;
            }
            const ADDR_T base = static_cast<ADDR_T>(r) * sliceSize + c0;
            LoadWiden(accUb, varUb, outputGm_, base, chunkW, evMV0, cp, pad); // acc = var[r] (widened)
            if constexpr (AscendC::IsSameType<PARAMS_T, int8_t>::value) {     // sign-extend int8 var
                ShiftRight(tmp, accUb, INT8_SIGN_SHIFT, static_cast<int32_t>(chunkW));
                Muls(tmp, tmp, INT8_SIGN_SPAN, static_cast<int32_t>(chunkW));
                Sub(accUb, accUb, tmp, static_cast<int32_t>(chunkW));
                PipeBarrier<PIPE_V>();
            }
            DivRangeIntoVar(accUb, updAcc, tmp, updQue, segStart, segEnd, sliceSize, c0, chunkW, cp, pad);
            NarrowStore(accUb, varUb, outputGm_, base, chunkW, evVM3, cp); // var[r] = acc
            // store (MTE3) reads accUb/varUb; the next segment's load (MTE2) reuses them -> wait
            SetFlag<HardEvent::MTE3_MTE2>(evM3M2);
            WaitFlag<HardEvent::MTE3_MTE2>(evM3M2);
        }
    }
}

// sliceSize==1, ALL modes (MUL/DIV/MAX/MIN), large M/varFirstDim (beyond the single-core scalar UB cap): sort
// the indices (Phase 1), then SCALAR-fold each var row. Mirrors SortDivRowProcess's row-ownership split -- each
// core owns the segments whose START is in its count-balanced position range and folds them WHOLE; rows are
// single-owner so there is NO cross-core combine and NO barrier after SortIndices. The fold is a plain scalar
// reduce in AccT (fp32 for fp16 -> no fp16 mid-chain overflow; int32 for int8/uint8 -> the scalar int8->int32
// cast sign-extends correctly, unlike the broken vector Cast, and mod-2^k matches the reference; native for
// fp32/int32). MUL=product, MAX/MIN=running max/min (never overflows), DIV=sequential in-place divide (/0 ->
// inf for float, keep-var for int -- matches the reference, never forms the divisor product). Read/written one
// element at a time via GetValue/SetValue/DataCopyPad -> no UB capacity limit, no atomic CAS storm. This is
// what lets sort cover sliceSize==1 (the vector fold's width-1 crash is sidestepped) for every mode.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::SortScalarSliceSize1(ADDR_T M,
                                                                                                  ADDR_T varFirstDim)
{
    constexpr bool isFloat = AscendC::IsSameType<PARAMS_T, half>::value || AscendC::IsSameType<PARAMS_T, float>::value;
    SortIndices(M); // Phase 1: sort indices into sortedIdxGm_ / originPosGm_, then collective SyncAll
    pipe_.Reset();
    // MULTI-CORE fold (row-ownership): each core folds the rows whose segment STARTS in its count-balanced
    // position range, WHOLE (no cross-core combine). The expensive O(M) product is parallelised across cores.
    // The write uses DataCopyPad (MTE/DMA, coherent) instead of scalar SetValue -- scalar SetValue goes through
    // L1 (read-modify-write the cache line), so neighbouring cores writing adjacent var rows lose each other's
    // bytes. DataCopyPad writes straight to GM (byte-precise DMA), the same coherent path the sliceSize>=2 sort
    // uses -- verified coherent at the adjacent-core boundary even for sub-32B unaligned single-element writes.
    //
    // Reads are scalar GetValue: the sequential sortedIdx/originPos scans are hardware-prefetched, and the xGm
    // gather is SCATTERED (originPos permutes positions) so it cannot be batched cheaply. A measured DataCopyPad
    // tile-staging variant (the "A opt") REGRESSED to 0.72-0.81x -- it added MTE2 DMA + per-tile sync overhead on
    // reads that were already cheap, without touching the dominant scattered gather. Scalar reads kept.
    TBuf<TPosition::VECCALC> bRes;
    pipe_.InitBuffer(bRes, 32);
    LocalTensor<PARAMS_T> resUb = bRes.Get<PARAMS_T>();
    const event_t evSM3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
    const event_t evM3S = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
    const DataCopyExtParams cp1{1, static_cast<uint32_t>(sizeof(PARAMS_T)), 0, 0, 0};
    const ADDR_T bn = static_cast<ADDR_T>(tiling_.blockNum);
    const ADDR_T posBegin = static_cast<ADDR_T>(blockIdx_) * M / bn;
    const ADDR_T posEnd = (static_cast<ADDR_T>(blockIdx_) + 1) * M / bn;
    if (posBegin >= posEnd) {
        return;
    } // no SyncAll past this point -> a quiet core cannot deadlock the rest
    ADDR_T k = posBegin;
    if (k > 0 && RowOf(k) == RowOf(k - 1)) { // segment spilled from earlier core
        const ADDR_T rs = RowOf(k);
        while (k < M && RowOf(k) == rs) {
            k++;
        }
    }
    while (k < posEnd && k < M) {
        const ADDR_T r = RowOf(k);
        const ADDR_T segStart = k;
        while (k < M && RowOf(k) == r) {
            k++;
        }
        const ADDR_T segEnd = k;
        if (r < 0 || static_cast<ADDR_T>(r) >= varFirstDim) {
            continue;
        }
        AccT acc = static_cast<AccT>(outputGm_.GetValue(static_cast<ADDR_T>(r))); // var[r], widened to AccT
        for (ADDR_T j = segStart; j < segEnd; j++) {
            const uint32_t p = originPosGm_.GetValue(j);
            const AccT u = static_cast<AccT>(xGm_.GetValue(static_cast<ADDR_T>(p)));
            if constexpr (Mode == MODE_MUL) {
                acc = acc * u;
            } // fp32 / int product
            else if constexpr (Mode == MODE_MAX) {
                acc = (acc != acc) ? acc : ((acc > u) ? acc : u);
            } // NaN passthrough
            else if constexpr (Mode == MODE_MIN) {
                acc = (acc != acc) ? acc : ((acc < u) ? acc : u);
            } // (a!=a): codebase idiom
            else {
                if constexpr (isFloat) {
                    acc = acc / u;
                } // MODE_DIV: float /0 -> inf
                else {
                    if (u != static_cast<AccT>(0)) {
                        acc = acc / u;
                    }
                }
            } // integer /0 keeps var
        }
        resUb.SetValue(0, static_cast<PARAMS_T>(acc));
        SetFlag<HardEvent::S_MTE3>(evSM3);
        WaitFlag<HardEvent::S_MTE3>(evSM3);
        DataCopyPad(outputGm_[static_cast<ADDR_T>(r)], resUb, cp1); // coherent 1-element write to var[r]
        SetFlag<HardEvent::MTE3_S>(evM3S);
        WaitFlag<HardEvent::MTE3_S>(evM3S);
    }
}

#endif // SCATTER_REDUCE_COMMON_SORT_H
