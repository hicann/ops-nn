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
 *        dispatched from Process():
 *          - scatter_reduce_common_sort.h   : sort + count-balanced multi-core split; sliceSize>=2 column
 *                                             tiling, sliceSize==1 large-M scalar row-fold (all modes)
 *          - scatter_reduce_common_scalar.h : single-core scalar reduction for small sliceSize==1
 *          - scatter_reduce_common_atomic.h : sliceSize==1 MAX/MIN 走硬件原子归约, 跳过全局排序
 */
#ifndef SCATTER_REDUCE_COMMON_SIMT_H
#define SCATTER_REDUCE_COMMON_SIMT_H

#include "scatter_reduce_common_struct.h"
#include "kernel_operator.h"
// GetVRegSize(): 向量寄存器宽度, 供原子路径的向量化筛选算每轮元素数。
// 相对路径按构建树解析(memory: dag-header-compiled-in-two-trees), 写法同 index/arg_max_grad。
#include "../inc/platform.h"

namespace ScatterReduceCommon {
using namespace AscendC;

// sliceSize==1 scalar-path cap: M + varFirstDim must fit the single-core UB staging (idx+upd+var+acc). Above
// this, the sort + scalar row-fold path takes over (no UB limit). 分片阈值不写死, 由 host 按 UB 反解下发 (test
// set maxes at M=4096, varFirstDim=1000) with margin and stays UB-safe even for int64 indices.
constexpr SortConfig scatterSortConfig{SortType::RADIX_SORT, false}; // radix sort of int32 index keys

// Subword (int8/uint8) <-> int32 conversion via Reg. High-level Cast<int32,int8> is broken on
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
        Reg::RegTensor<T> srcValue;
        Reg::MaskReg preg;
        uint32_t sregMask = n;
        for (uint16_t i = 0; i < loopTimes; i++) {
            auto dstReg = Reg::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
            auto srcReg = Reg::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
            preg = Reg::UpdateMask<int32_t>(sregMask);
            Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr, srcReg);
            Reg::DataCopy<int32_t, Reg::StoreDist::DIST_NORM>(dstAddr, (Reg::RegTensor<int32_t>&)srcValue, dstReg,
                                                              preg);
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
        Reg::RegTensor<int32_t> srcValue;
        Reg::MaskReg preg;
        uint32_t sregMask = n;
        for (uint16_t i = 0; i < loopTimes; i++) {
            auto srcReg = Reg::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
            auto dstReg = Reg::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
            preg = Reg::UpdateMask<int32_t>(sregMask);
            Reg::DataCopy<int32_t, Reg::LoadDist::DIST_NORM>(srcValue, srcAddr, srcReg);
            Reg::DataCopy<T, Reg::StoreDist::DIST_PACK4_B32>(dstAddr, (Reg::RegTensor<T>&)srcValue, dstReg, preg);
        }
    }
}

// sliceSize==1 fast path lives in its own file (crisp, distinct class -- single-core scalar reduction). Pulled
// in here, after the subword widen/narrow helpers it reuses, so the dispatcher can route to it.
#include "scatter_reduce_common_scalar.h"

// sliceSize==1 的 MAX/MIN 硬件原子路径(同款 FRAGMENT 约定)。放在 scalar.h 之后, 供 Process() 在
// 落到 sort 行折叠之前优先路由。
#include "scatter_reduce_common_atomic_common.h" // 共用件: 判据/单位元/翻转/命中压缩/UbBufSpec
#include "scatter_reduce_common_atomic.h"        // AtomicReduceRow1D (ss==1)
#include "scatter_reduce_common_ubresident.h"    // UbResidentReduceRow1D (ss==1 与 ss>=2)
#include "scatter_reduce_common_bucket.h"        // BucketReduceWideSpan (宽 span MUL)
#include "scatter_reduce_common_atomic_slicen.h" // AtomicReduceSliceN (ss>=2)

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
class ScatterReduceSimt {
public:
    __aicore__ inline ScatterReduceSimt(const ScatterReduceSimtTilingData& tilingData, TPipe& pipe)
        : tiling_(tilingData), pipe_(pipe)
    {}
    // workspace defaults to nullptr: only the MUL sort path consumes it; DIV/MAX/MIN call Init(var,idx,upd).
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace = nullptr);
    __aicore__ inline void Process();
    // Process 的选路链: 每条候选路径一个 TryXxx, 返回"本核已处理"。原先全挤在 Process 里,
    // 圈复杂度 55(全仓 16650 个 arch35 内核函数里排第 29)。拆法同 sort.h 的 SortIndices。
    __aicore__ inline UbBufSpec MakeUbSpec() const;
    __aicore__ inline bool ProcessReduceModes(ADDR_T sliceSize, ADDR_T varFirstDim);
    __aicore__ inline bool ProcessDivMode(ADDR_T sliceSize, ADDR_T varFirstDim);
    __aicore__ inline bool TryScalarReduce1D(ADDR_T sliceSize, ADDR_T varFirstDim);
    __aicore__ inline bool TryUbResident(ADDR_T sliceSize, ADDR_T varFirstDim, const UbBufSpec& ubSpec);
    __aicore__ inline bool TryBucketReduce(ADDR_T sliceSize, ADDR_T varFirstDim);
    __aicore__ inline bool TryAtomicSliceN(ADDR_T sliceSize, ADDR_T varFirstDim, const UbBufSpec& ubSpec);
    __aicore__ inline bool TryAtomicRow1D(ADDR_T sliceSize, ADDR_T varFirstDim, const UbBufSpec& ubSpec);
    __aicore__ inline bool TrySortSliceN(ADDR_T sliceSize, ADDR_T varFirstDim);
    __aicore__ inline bool TrySortScalar1D(ADDR_T sliceSize, ADDR_T varFirstDim);
    // ACC: float for fp16, native for fp32/int32, int32 for int8/uint8 (forced by hardware -- vmul has no
    // 8-bit form). Used by the sort-based product and the cross-core partial-combine. Declared before the
    // sort-phase helper declarations below, which take LocalTensor<AccT>& parameters.
    using AccT = typename Conditional<
        sizeof(PARAMS_T) == 4, PARAMS_T,
        typename Conditional<AscendC::IsSameType<PARAMS_T, half>::value, float, int32_t>::type>::type;
    // 更新行的深预取深度: 散列搬入是延迟受限的, 多留几笔在途。AHEAD 少一笔, 留给正在折叠的那行。
    // UB_BLOCK_BYTES 定义在 struct.h(host/kernel 共用, 不便引 AscendC 命名空间), 在此绑定校验。
    static_assert(ScatterReduceCommon::UB_BLOCK_BYTES == AscendC::ONE_BLK_SIZE,
                  "UB_BLOCK_BYTES 必须与 AscendC::ONE_BLK_SIZE 一致");
    static constexpr int32_t PREFETCH_DEPTH = static_cast<int32_t>(ScatterReduceCommon::SORT_PREFETCH_DEPTH);
    static constexpr int32_t PREFETCH_AHEAD = PREFETCH_DEPTH - 1;

    // 缩放域折叠生效条件: 浮点 MUL/DIV。与 host 的 SortFoldPlan().scaledFold 同一判据。
    static constexpr bool kScaledFold = (AscendC::IsSameType<PARAMS_T, half>::value ||
                                         AscendC::IsSameType<PARAMS_T, float>::value) &&
                                        (Mode == ScatterReduceCommon::MODE_MUL ||
                                         Mode == ScatterReduceCommon::MODE_DIV);

    // bTmp 切分: 非缩放域一份(int8 符号扩展/整型 0->1); 缩放域另加指数累加、指数增量、两张掩码。
    // 掩码区必须整到 32B —— 比较指令的掩码目标要求 block 对齐, 否则 VEC_ERROR。
    __aicore__ static inline uint32_t MaskElems(uint32_t cPad)
    {
        const uint32_t bytes = (cPad + 7U) / 8U;
        return ((bytes + ScatterReduceCommon::UB_BLOCK_BYTES - 1U) / ScatterReduceCommon::UB_BLOCK_BYTES) *
               (ScatterReduceCommon::UB_BLOCK_BYTES / sizeof(int32_t));
    }
    __aicore__ static inline uint32_t ScaledTmpElems(uint32_t cPad)
    {
        return kScaledFold ? (2U * cPad + 2U * MaskElems(cPad)) : cPad;
    }
    __aicore__ inline void BindScaledState(LocalTensor<int32_t>& tmp, uint32_t cPad)
    {
        if constexpr (kScaledFold) {
            scaledExp_ = tmp;
            scaledDlt_ = tmp[cPad];
            scaledMk_ = tmp[2U * cPad].template ReinterpretCast<uint8_t>();
            scaledMk2_ = tmp[2U * cPad + MaskElems(cPad)].template ReinterpretCast<uint8_t>();
        }
    }
    // sort-based POSITION-split reducer for MUL/MAX/MIN (defined in scatter_reduce_common_sort.h): core 0
    // globally sorts indices into workspace, SyncAll, then every core reduces a count-balanced range of sorted
    // segments (each segment = one var row) with sliceSize column-tiling and a cross-core partial combine
    // (MUL: product; MAX/MIN: max/min -- all associative+commutative, so splittable & combinable). No atomics.
    // 外壳: 把全量 M 切成固定长度的片依次处理, 片间插集合屏障(var 读改写依赖)。
    // 排序 scratch 只按一片定尺, workspace 因此与 M 无关, 支持的更新条数无上限。
    __aicore__ inline void SortReduceProcess(ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim);
    // 单片处理: [base, base+M) 这一片的排序 + 折叠。
    __aicore__ inline void SortReduceProcessChunk(ADDR_T base, ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim);
    // M <= blockNum(每核摊不到 2 个排序位)时的列并行折叠(scatter_reduce_common_sort.h): 位置切分在该
    // 档退化为"每核 1 位 + 跨核分段/部分合并"的重机械, 实测死锁(SMsortD/E/F: MAX, M==blockNum, 索引
    // 分散)。归约按列独立, 列条带单写者, 全程零集合同步。
    __aicore__ inline void SortColumnsParallel(ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim);
    // SortReduceProcess per-chunk sub-phases (pure code-motion split; same execution order + sync pairing):
    //   SortPhase3aFold     : in-range fold for columns [c0,c0+chunkW). Whole segments -> var; a segment that
    //                         spans this core's right boundary -> in-range partial to partialGm_[core] and the
    //                         owner-split (row,end) is returned for 3b. Runs before SyncAll #2.
    //   SortPhase3bCombine  : owner folds the covering cores' partials (after SyncAll #2) + var, then stores.
    __aicore__ inline void SortPhase3aFold(ADDR_T posBegin, ADDR_T posEnd, ADDR_T M, ADDR_T sliceSize,
                                           ADDR_T varFirstDim, ADDR_T c0, ADDR_T chunkW, ADDR_T sAlign,
                                           LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb,
                                           LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
                                           TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, event_t evMV0,
                                           event_t evVM3, event_t evM3M2, const DataCopyExtParams& cp,
                                           const DataCopyExtParams& cpAcc, const DataCopyPadExtParams<PARAMS_T>& pad,
                                           ADDR_T& ownerSplitRow, ADDR_T& ownerSplitEnd, event_t evVS, event_t evSM3);
    __aicore__ inline void SortPhase3bCombine(ADDR_T M, ADDR_T sliceSize, ADDR_T c0, ADDR_T chunkW, ADDR_T sAlign,
                                              ADDR_T ownerSplitRow, ADDR_T ownerSplitEnd, LocalTensor<AccT>& accUb,
                                              LocalTensor<PARAMS_T>& varUb, LocalTensor<AccT>& rowAccUb,
                                              LocalTensor<int32_t>& tmp, event_t evMV0, event_t evVM3, event_t evM3M2,
                                              event_t evVM2, const DataCopyExtParams& cp,
                                              const DataCopyExtParams& cpAcc, const DataCopyPadExtParams<PARAMS_T>& pad,
                                              const DataCopyPadExtParams<AccT>& padAcc, event_t evVS, event_t evSM3);
    // sort-based ROW-split reducer for DIV (defined in scatter_reduce_common_sort.h): same global sort, but
    // every core owns the segments whose START is in its count-balanced position range and divides those whole
    // rows IN PLACE across the full slice -- a divide chain cannot be split across cores (the divisor product
    // overflows), so each row is owned by one core, no combine. Each core walks only O(M/P) positions (the
    // column-split's O(M) walk blew up at large M). Shares Phase 1 via SortIndices.
    // 外壳: 把全量 M 切成固定长度的片依次处理, 片间插集合屏障(var 读改写依赖)。
    // 排序 scratch 只按一片定尺, workspace 因此与 M 无关, 支持的更新条数无上限。
    __aicore__ inline void SortDivRowProcess(ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim);
    // 单片处理: [base, base+M) 这一片的排序 + 折叠。
    __aicore__ inline void SortDivRowProcessChunk(ADDR_T base, ADDR_T M, ADDR_T sliceSize, ADDR_T varFirstDim);
    // sliceSize==1 for M (or varFirstDim) beyond the scalar UB cap, ALL modes (MUL/DIV/MAX/MIN): reuse Phase-1
    // SortIndices, then fold each var row's updates with a SCALAR op (no width-1 vector, so sort CAN do
    // sliceSize==1). Row-ownership split (each core owns segments STARTING in its range, whole, no combine, no
    // barrier); DataCopyPad coherent write. AccT intermediate -> correct (no fp16 overflow) for every dtype.
    // 外壳: 把全量 M 切成固定长度的片依次处理, 片间插集合屏障(var 读改写依赖)。
    // 排序 scratch 只按一片定尺, workspace 因此与 M 无关, 支持的更新条数无上限。
    __aicore__ inline void SortScalarSliceSize1(ADDR_T M, ADDR_T varFirstDim);
    // 单片处理: [base, base+M) 这一片的排序 + 折叠。
    __aicore__ inline void SortScalarSliceSize1Chunk(ADDR_T base, ADDR_T M, ADDR_T varFirstDim);
    // Phase 1 of both sort reducers: core 0 sorts indices into sortedIdx/originPos workspace, then SyncAll.
    // 分片排序: 只对 idx[base, base+M) 这一片排序, 结果落在固定大小的 scratch 里。
    // 排序 scratch 是 O(片长) 而非 O(全量 M), workspace 因此恒定, 算子支持的更新条数无上限。
    // 位置数组存**全局**下标(base + 片内局部位置), 折叠阶段据此取 updates 行。
    __aicore__ inline void SortIndices(ADDR_T base, ADDR_T M);
    // 取排序槽 k 对应的**完整**行号。窄档直接读 sortedIdx(它就是完整 index); 宽档 sortedIdx 只存
    // lo(低 30 位), 完整值由 originPos 回读原始 int64 索引得到 —— 这样消费者无需桶边界表, 也天然
    // 避免"跨桶 lo 相同被误判为同一行"的坑。
    __aicore__ inline ADDR_T RowOf(ADDR_T k);
    // SortIndices sub-phases (pure code-motion split of SortIndices; same execution order, same sync pairing):
    //   SortIndicesSmallM  : M<=tiling_.sortTile fast path -- core 0 single AscendC::Sort + collective DCCI + SyncAll.
    //   SortLocalRuns      : large-M phase 2 -- each core sorts its runLen0-wide chunks (pad tail with SENT),
    //                        writes them into the curSel buffer, then the collective DCCI + SyncAll.
    //   SortMergeTree      : large-M merge tree -- each round each core emits its merge-path slice; flips curSel.
    __aicore__ inline void SortIndicesSmallM(ADDR_T base, ADDR_T M);
    __aicore__ inline void SortLocalRuns(ADDR_T base, ADDR_T M, ADDR_T P2, ADDR_T runLen0, int curSel);
    __aicore__ inline void SortMergeTree(ADDR_T M, ADDR_T P2, ADDR_T runLen0, ADDR_T nRounds, int curSel);
    // 左边缘段的折叠(定义在 scatter_reduce_common_sort.h): 只算范围内部分, 落 partial 供 3b 合并。
    __aicore__ inline void FoldSpilledLeftEdge(
        ADDR_T& k, ADDR_T posEnd, ADDR_T varFirstDim, ADDR_T sliceSize, ADDR_T c0, ADDR_T chunkW, ADDR_T sAlign,
        LocalTensor<AccT>& accUb, LocalTensor<PARAMS_T>& varUb, LocalTensor<AccT>& rowAccUb, LocalTensor<int32_t>& tmp,
        TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, event_t evMV0, event_t evVM3, event_t evM3M2,
        const DataCopyExtParams& cp, const DataCopyExtParams& cpAcc, const DataCopyPadExtParams<PARAMS_T>& pad);
    // 跳过归前一个核的左边缘段(定义在 scatter_reduce_common_sort.h)。
    __aicore__ inline ADDR_T SkipSpilledLeftEdge(ADDR_T k, ADDR_T M);
    // 本核是否整体接管该段(定义在 scatter_reduce_common_sort.h)。
    __aicore__ inline bool TakeSegmentWhole(ADDR_T segEnd, ADDR_T posEnd);
    // 取较小值。
    __aicore__ inline ADDR_T MinAddr(ADDR_T a, ADDR_T b);
    // phase-3 的列切分上限不在 kernel 侧定死: host 用 GetCoreMemSize 取到 UB 实测容量, 按类型折算
    // 每列存活开销(accUb+rowAccUb+varUb+updQue深度8+tmp)后反解, 经 tiling 的 ubChunkMax 下发。
    // sliceSize 只是被切成若干个这样的 chunk, 不构成上限, 故任一轴再大也不会不支持。
    // Shared deep-prefetch fold skeleton for MUL and MAX/MIN (defined in scatter_reduce_common_sort.h): primes
    // the scattered loads, keeps AHEAD in flight, and applies the Mode-selected fold (MUL product / MAX/MIN
    // max-min) per row with each path's exact PipeBarrier rhythm. MulRangeIntoAcc / MaxMinRangeIntoAcc are thin
    // wrappers over it; `tmp` is read only by the MAX/MIN int8 sign-extend (may be empty on the MUL path).
    __aicore__ inline void FoldRangeIntoAcc(LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb,
                                            LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue,
                                            ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff, ADDR_T width,
                                            const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad);
    // Deep-prefetch LOAD-MANAGEMENT shared by the fold (FoldRangeIntoAcc) and the divide (DivRangeIntoVar)
    // skeletons (defined in scatter_reduce_common_sort.h). These do ONLY the scattered-load queue management
    // (originPos GetValue -> AllocTensor -> DataCopyPad -> EnQue); the DeQue + per-row consume + FreeTensor stay
    // in each caller so each path keeps its own barrier rhythm untouched.
    //   PrimePrefetch  : prime the first `primed` (= min(cnt, AHEAD)) scattered loads before the consume loop.
    //   PrefetchAhead  : inside the consume loop, prefetch the row AHEAD ahead of `j` while it is still in range.
    __aicore__ inline void PrimePrefetch(TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T primed,
                                         ADDR_T rowStride, ADDR_T colOff, const DataCopyExtParams& cp,
                                         const DataCopyPadExtParams<PARAMS_T>& pad);
    __aicore__ inline void PrefetchAhead(TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ, ADDR_T j,
                                         ADDR_T cnt, ADDR_T AHEAD, ADDR_T rowStride, ADDR_T colOff,
                                         const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad);
    // Deep-prefetch multiply of updates[originPos[startJ..endJ)] columns [colOff,colOff+width) INTO accUb
    // (already initialised by caller). rowStride is the full row width (updates rows are rowStride apart).
    __aicore__ inline void MulRangeIntoAcc(LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb,
                                           LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue,
                                           ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff, ADDR_T width,
                                           const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad);
    // DivRangeIntoVar divides accUb (= var[row], in the ACC type) by the segment's updates in place. INTEGER
    // uses the exact int32 vector Div with 0-divisors replaced by 1 (so /0 keeps var, matching the reference,
    // and integer division stays order-independent); FLOAT divides in fp32 (/0 -> inf). updAcc holds the
    // widened update, tmp is the int 0->1 scratch.
    __aicore__ inline void DivRangeIntoVar(LocalTensor<AccT>& accUb, LocalTensor<AccT>& updAcc,
                                           LocalTensor<int32_t>& tmp, TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue,
                                           ADDR_T startJ, ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff, ADDR_T width,
                                           const DataCopyExtParams& cp, const DataCopyPadExtParams<PARAMS_T>& pad);
    // MaxMinRangeIntoAcc folds the segment's updates into accUb (already seeded) by Mode's max/min -- the
    // MAX/MIN counterpart of MulRangeIntoAcc for the position-split reducer. Widens like MUL/DIV (int8/uint8
    // -> int32 with int8 sign-extended via tmp so the compare is signed-correct; fp16 -> float). rowAccUb is
    // the per-update widen scratch; tmp the int8 sign-extend scratch. max/min combine across cores, so this is
    // used for both whole segments and per-core partials.
    __aicore__ inline void MaxMinRangeIntoAcc(LocalTensor<AccT>& accUb, LocalTensor<AccT>& rowAccUb,
                                              LocalTensor<int32_t>& tmp,
                                              TQue<QuePosition::VECIN, PREFETCH_DEPTH>& updQue, ADDR_T startJ,
                                              ADDR_T endJ, ADDR_T rowStride, ADDR_T colOff, ADDR_T width,
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
    GlobalTensor<PARAMS_T> xGm_;            // updates
    GlobalTensor<PARAMS_T> outputGm_;       // var (in-place reduce target)
    GlobalTensor<INDICES_T> sortedIdxGm_;   // workspace: sorted index values [M] (key = 索引真值)
    GlobalTensor<int64_t> originPosGm_;     // workspace: original update position per sorted slot [M]
    GlobalTensor<INDICES_T> scratchKeysGm_; // workspace: K pre-merge sorted index runs [M] (tiled merge-sort)
    GlobalTensor<int64_t> scratchPosGm_;    // workspace: K pre-merge sorted position runs [M] (tiled merge-sort)
    // split-combine scratch: when one var row's segment spans several cores, each covering core writes its
    // in-range partial product to partialGm_[core] (one CHUNK-wide slot); the row's owner reads back the
    // consecutive covering cores' partials and combines (4 bytes per elem, reinterpreted to the ACC type).
    GlobalTensor<int32_t> partialGm_; // workspace: per-core partial product [blockNum * chunk]
    // 缩放域折叠的状态(仅浮点 MUL/DIV 绑定): 指数累加、指数增量、两张比较掩码。
    // 全部从 bTmp 一块里切, 由分配点一次绑定 —— 做成成员是为了不给五个折叠调用点扩散签名。
    LocalTensor<int32_t> scaledExp_;
    LocalTensor<int32_t> scaledDlt_;
    LocalTensor<uint8_t> scaledMk_;
    LocalTensor<uint8_t> scaledMk2_;
    // 分桶路径复用整块排序 scratch(分桶不排序, 该区全程空闲), 需要知道它有多大。mAlign 只在 Init
    // 里算一次并在此留存 —— 不在第二处重算 padM: 两处算出不同 padM 会让各子区基址错位(见 Init 内
    // "分片阈值必须与 SortIndices 同源"那段), 这是同一个坑。
    ADDR_T wsUserBytes_{0};
    GM_ADDR userWsBase_{nullptr};
    uint32_t blockIdx_{0};
    ADDR_T xBlockOffSet_{0};
    ADDR_T currBlockTilingSize_{0};
    ADDR_T totalIdxn_{0};
};

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                  GM_ADDR updates, GM_ADDR workspace)
{
    if (tiling_.sliceSize == 0) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    xBlockOffSet_ = static_cast<ADDR_T>(tiling_.blockTilingSize) * blockIdx_;
    currBlockTilingSize_ = (blockIdx_ == tiling_.blockNum - 1) ? static_cast<ADDR_T>(tiling_.tailBlockTilingSize) :
                                                                 static_cast<ADDR_T>(tiling_.blockTilingSize);
    idxGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    xGm_.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm_.SetGlobalBuffer((__gm__ PARAMS_T*)var);
    // total index entries across all cores (front cores carry blockTilingSize, last carries the tail)
    totalIdxn_ = (static_cast<ADDR_T>(tiling_.blockTilingSize) * (static_cast<ADDR_T>(tiling_.blockNum) - 1) +
                  static_cast<ADDR_T>(tiling_.tailBlockTilingSize)) /
                 static_cast<ADDR_T>(tiling_.sliceSize);
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
        // The parallel merge-sort (SortIndices) pads M up to padM = P2*runLen0, where P2 is raised PAST
        // blockNum until runLen0 = ceil(M/P2) <= tiling_.sortTile. padM - M can reach ~P2 (thousands for M > ~1M),
        // so each region MUST be padM wide. The old "+128" margin assumed padM - M <= coreNum, which is FALSE
        // once P2 grows past coreNum -- the sort then wrote into the next region -> VEC_ERROR / wrong output on
        // large-index cases. Compute the exact padM here (mirrors SortIndices and the host tiling).
        // 分片阈值必须与 SortIndices 同源: 都读 host 下发的 tiling_.sortTile(host 按 UB 实测容量反解)。
        // 写死常量会与 SortIndices 算出不同的 padM(阈值越小 -> P2 越大 -> padM 越大), 差值可达 P2 量级,
        // 远超下面 +128 的余量, 于是各子区基址错位、排序写出区间重叠 -> 越界写/静默错值。
        const ADDR_T sortTile = static_cast<ADDR_T>(tiling_.sortTile);
        // scratch 只按**一片**定尺(与 host 的 workspace 公式同源): 全量 M 被切成若干片,
        // 每片独立排序+折叠, 复用同一块 scratch, workspace 因此与 M 无关。
        const ADDR_T chunkM = MinAddr(totalIdxn_, static_cast<ADDR_T>(tiling_.sortChunkM));
        ADDR_T padM = chunkM;
        if (chunkM > sortTile) {
            ADDR_T p2 = 1;
            while (p2 * 2 <= static_cast<ADDR_T>(tiling_.blockNum)) {
                p2 *= 2;
            }
            while ((chunkM + p2 - 1) / p2 > sortTile) {
                p2 *= 2;
            }
            padM = p2 * ((chunkM + p2 - 1) / p2);
        }
        const ADDR_T mAlign = (padM + 7) / 8 * 8 + 128; // 32B align (8 x 4B) on top of the true padM
        // layout: sortedIdx[M] | originPos[M] | scratchKeys[M] | scratchPos[M] | per-core partials.
        // scratchKeys/scratchPos hold the K pre-merge sorted runs that the tiled merge-sort folds into
        // sortedIdx/originPos (they are free during phase 3, where only partials are used).
        // layout: sortedIdx[mAlign x sizeof(INDICES_T)] | originPos[mAlign x 4]
        //       | scratchKeys[mAlign x sizeof(INDICES_T)] | scratchPos[mAlign x 4] | partials
        // key 是索引真值, 故两个 key 区按索引类型宽度计; host 的 workspace 公式必须同源。
        constexpr ADDR_T KEY_SZ = static_cast<ADDR_T>(sizeof(INDICES_T));
        constexpr ADDR_T POS_SZ = static_cast<ADDR_T>(sizeof(int64_t));
        sortedIdxGm_.SetGlobalBuffer((__gm__ INDICES_T*)userWs);
        originPosGm_.SetGlobalBuffer((__gm__ int64_t*)(userWs + mAlign * KEY_SZ));
        scratchKeysGm_.SetGlobalBuffer((__gm__ INDICES_T*)(userWs + mAlign * (KEY_SZ + POS_SZ)));
        scratchPosGm_.SetGlobalBuffer((__gm__ int64_t*)(userWs + mAlign * (2 * KEY_SZ + POS_SZ)));
        partialGm_.SetGlobalBuffer((__gm__ int32_t*)(userWs + mAlign * 2 * (KEY_SZ + POS_SZ)));
        // 分桶路径要整块复用这片 scratch。容量按 host 的 workspace 公式同源取:
        // 四个排序子区(sortedIdx/originPos/scratchKeys/scratchPos) + per-core partials。
        // partials 段按 host 公式是 blockNum * sliceAlign * 4, 这里保守只认前四段, 不把余量算进来。
        userWsBase_ = userWs;
        wsUserBytes_ = mAlign * 2 * (KEY_SZ + POS_SZ);
    }
}

// Sort-based MUL path (global sort + count-balanced split + sl column-tiling) lives in its own file --
// it is not SIMT. Defines ScatterReduceSimt::SortReduceProcess and its helpers.
#include "scatter_reduce_common_sort.h"

#include "scatter_reduce_common_dispatch.h" // Process 与各 TryXxx 选路候选
} // namespace ScatterReduceCommon

#endif // SCATTER_REDUCE_COMMON_SIMT_H
