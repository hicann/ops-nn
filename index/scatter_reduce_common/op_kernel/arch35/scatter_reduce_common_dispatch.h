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
 * \file scatter_reduce_common_dispatch.h
 * \brief ScatterReduceSimt 的选路链: Process 与各 TryXxx 候选路径的定义。
 *
 * FRAGMENT header: #include'd from scatter_reduce_common_simt.h INSIDE namespace ScatterReduceCommon,
 * 在类声明之后。由 simt.h 拆出(原单文件 569 行, 超 CodeCheck 500 行阈值)。
 */
#ifndef SCATTER_REDUCE_COMMON_DISPATCH_H
#define SCATTER_REDUCE_COMMON_DISPATCH_H
// host 下发的每块 UB 缓冲字节数, 原样透传给 InitBuffer(内核不再自己乘)。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline UbBufSpec ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::MakeUbSpec() const
{
    return UbBufSpec{static_cast<uint32_t>(tiling_.atomicIdxBytes), static_cast<uint32_t>(tiling_.atomicUpdBytes),
                     static_cast<uint32_t>(tiling_.atomicPosBytes), static_cast<uint32_t>(tiling_.atomicSlotSpan)};
}

// Product/max/min are order-independent (assoc.+comm.), so MUL/MAX/MIN share ONE path: global sort (core 0) +
// count-balanced segment split (all cores), no atomics. Its only ceiling is fitting core 0's UB --
// the M-index sort plus a single column chunk of the per-segment product -- both derived from the
// buffer layout and UB size (no magic thresholds). sliceSize is column-tiled so it never limits; M
// is bounded only by the single-core sort capacity.
// sliceSize==1, small (M + varFirstDim fit single-core UB): 1D scalar reduce (product/max/min). One core
// stages it all in UB and scalar-folds in fp32 -- crisply bounded, beats the multi-core sort's setup for
// the tiny work. Cap is UB capacity (not a magic 256), so all realistic 1-D cases land here.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TryScalarReduce1D(ADDR_T sliceSize,
                                                                                               ADDR_T varFirstDim)
{
    if (!(sliceSize == 1 && totalIdxn_ > 0 && varFirstDim > 0 && tiling_.scalarPath != 0)) {
        return false;
    }
    if (blockIdx_ == 0) {
        ScalarReduce1D<Mode, PARAMS_T, INDICES_T, ADDR_T>(pipe_, idxGm_, xGm_, outputGm_, totalIdxn_, varFirstDim);
    }
    return true; // single core owns the whole 1D reduction (separate file: scatter_reduce_common_scalar.h)
}

// MAX/MIN + dtype 有硬件原子(或 uint8 走 workspace 合并) + **触达行区间装得进 UB**: 每核持私有 UB 副本,
// 命中全部在 UB 内折叠, 收尾只做一次整份 var 的原子写。每命中成本从 42.8ns(一次 32B GM
// 原子事务)降到约 2~4ns(UB 内一次标量比较)。入口是纯容量判定, 装不下返回 false 回落下面
// 的逐命中原子写, 不拒收。详见 scatter_reduce_common_ubresident.h 的头注(含为何这条路成立
// 而分桶/每核重扫不成立: 不多扫一趟 M、不落盘、搬运不乘核数、无集合性 SyncAll)。
// uint8 无硬件原子, 但本路径的跨核合并有 workspace 变体(不需要 x^0x80 变换), 故判据用
// UsableRow1D 把 uint8 一并纳入 —— 实测 uint8 的 4 例(L1_337/L1_291/L1_187/L1_233)
// 在 v1 只接硬件原子时是 1.0x 未生效, 接上 workspace 合并后才能吃到这条路。
// MUL(2026-10-31)同样接入: 乘法序无关, 按 M 分并行成立; 无硬件原子乘, 一律走 workspace
// 合并(slot 按 ACC 域), 判据不再看 dtype。排序路径的全局排序恰是 MUL 不达标集(120 例,
// 108 例 ss==1)的主开销 —— 48 例索引全越界(排序后折叠零命中)、48 例热点行(span 1~101 行),
// 全部由"预扫 span + UB 驻留折叠"消解。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TryUbResident(ADDR_T sliceSize,
                                                                                           ADDR_T varFirstDim,
                                                                                           const UbBufSpec& ubSpec)
{
    if constexpr (((Mode == MODE_MAX || Mode == MODE_MIN) && IsAtomicMaxMinUsableRow1D<PARAMS_T>()) ||
                  Mode == MODE_MUL) {
        if (!(totalIdxn_ > 0 && varFirstDim > 0 && sliceSize > 0)) {
            return false;
        }
        const uint32_t tileU = (tiling_.atomicTile < ATOMIC_TILE_MAX) ? static_cast<uint32_t>(tiling_.atomicTile) :
                                                                        ATOMIC_TILE_MAX;
        return UbResidentReduceRow1D<Mode, PARAMS_T, INDICES_T, ADDR_T>(
            pipe_, idxGm_, xGm_, outputGm_, totalIdxn_, varFirstDim, sliceSize, static_cast<ADDR_T>(blockIdx_),
            static_cast<ADDR_T>(tiling_.blockNum), (tileU == 0U) ? 1U : tileU, static_cast<ADDR_T>(tiling_.ubUsable),
            userWsBase_, wsUserBytes_, ubSpec);
    } else {
        return false;
    }
}

// 宽 span 且命中近似均匀的 MUL: 分桶归约(scatter_reduce_common_bucket.h)。排在 UB 驻留之后:
// 触达区间装得下的负载早已被前者截走, 到这里的都是 span 远超 UB 的档(L1_650/662: 索引全
// int32 值域均匀, 有效命中 ~2%)。均衡门控(单桶命中 > 8x 均值)整体回落排序 —— 对抗性热点分布
// 在数据上与被否决版遇到的同构, 仍归排序的按命中数均衡分段。入口纯容量/shape 判定。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TryBucketReduce(ADDR_T sliceSize,
                                                                                             ADDR_T varFirstDim)
{
    if constexpr (Mode == MODE_MUL) {
        if (!(totalIdxn_ > 0 && varFirstDim > 0 && sliceSize > 0)) {
            return false;
        }
        return BucketReduceWideSpan<Mode, PARAMS_T, INDICES_T, ADDR_T>(
            pipe_, idxGm_, xGm_, outputGm_, totalIdxn_, varFirstDim, sliceSize, static_cast<ADDR_T>(blockIdx_),
            static_cast<ADDR_T>(tiling_.blockNum), static_cast<ADDR_T>(tiling_.ubUsable), userWsBase_, wsUserBytes_);
    } else {
        return false;
    }
}

// sliceSize>=2 + MAX/MIN + dtype 有硬件原子: 先试原子切片写回(每命中搬 sliceSize 个连续元素,
// DMA 粒度比 ss==1 更大更划算)。切片过宽装不下槽缓冲时返回 false, 回落排序路径, 不拒收。
// uint8 经 x^0x80 借 int8 原子, XFORM 现已在 SliceN 里接完整, 故 ss>=2 同样用 UsableRow1D 判据。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TryAtomicSliceN(ADDR_T sliceSize,
                                                                                             ADDR_T varFirstDim,
                                                                                             const UbBufSpec& ubSpec)
{
    if constexpr ((Mode == MODE_MAX || Mode == MODE_MIN) && IsAtomicMaxMinUsableRow1D<PARAMS_T>()) {
        if (!(sliceSize >= 2 && totalIdxn_ > 0 && varFirstDim > 0)) {
            return false;
        }
        // 规格与 ss==1 同源: host 按实测 UB 反解(含 uint8 XFORM 的额外缓冲)后下发。
        const uint32_t tileSn = static_cast<uint32_t>(tiling_.atomicTile);
        const uint32_t slotsSn = static_cast<uint32_t>(tiling_.atomicSlots);
        return AtomicReduceSliceN<Mode, PARAMS_T, INDICES_T, ADDR_T>(
            pipe_, idxGm_, xGm_, outputGm_, totalIdxn_, varFirstDim, sliceSize, static_cast<ADDR_T>(blockIdx_),
            static_cast<ADDR_T>(tiling_.blockNum), tileSn, slotsSn, ubSpec);
    } else {
        return false;
    }
}

// sliceSize==1 + MAX/MIN + dtype 有硬件原子: 按 M 均分各核原子写回, 跳过全局排序。
// 排序存在的理由是"把同一行的更新聚到一起再折叠", 而硬件原子归约在写回时就完成了这件事,
// 于是 54MB 排序 workspace 与多趟归并一并省掉。MAX/MIN 可结合可交换且结果恒为某个输入
// (scalar.h: "MAX/MIN never overflow"), 所以乱序归约与参考实现一致, 不引入不确定性。
// 不支持原子的 dtype(int64)照旧落到 sort 行折叠。
// uint8 经 x^0x80 保序映射借 int8 原子(XFORM), 故此处用 UsableRow1D 判据。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TryAtomicRow1D(ADDR_T sliceSize,
                                                                                            ADDR_T varFirstDim,
                                                                                            const UbBufSpec& ubSpec)
{
    if constexpr ((Mode == MODE_MAX || Mode == MODE_MIN) && IsAtomicMaxMinUsableRow1D<PARAMS_T>()) {
        if (!(sliceSize == 1 && totalIdxn_ > 0 && varFirstDim > 0)) {
            return false;
        }
        // 规格取 min(tiling 下发值, 编译期上限): 下发值来自 host 按实测 UB 的反解,
        // 上限只因批内元数据数组开在标量栈上、尺寸必须是编译期常量。
        const uint32_t tileN = (tiling_.atomicTile < ATOMIC_TILE_MAX) ? static_cast<uint32_t>(tiling_.atomicTile) :
                                                                        ATOMIC_TILE_MAX;
        const uint32_t slotsN = (tiling_.atomicSlots < SLOT_NUM_MAX) ? static_cast<uint32_t>(tiling_.atomicSlots) :
                                                                       SLOT_NUM_MAX;
        AtomicReduceRow1D<Mode, PARAMS_T, INDICES_T, ADDR_T>(
            pipe_, idxGm_, xGm_, outputGm_, totalIdxn_, varFirstDim, static_cast<ADDR_T>(blockIdx_),
            static_cast<ADDR_T>(tiling_.blockNum), (tileN == 0U) ? 1U : tileN, (slotsN == 0U) ? 1U : slotsN, ubSpec);
        return true;
    } else {
        return false;
    }
}

// sliceSize>=2 -> sort with the column-tiled VECTOR fold (any M).
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TrySortSliceN(ADDR_T sliceSize,
                                                                                           ADDR_T varFirstDim)
{
    if (!(sliceSize >= 2 && totalIdxn_ > 0 && varFirstDim > 0)) {
        return false;
    }
    SortReduceProcess(totalIdxn_, sliceSize, varFirstDim);
    return true;
}

// sliceSize==1 too big for the scalar UB cap -> sort + SCALAR row-fold. The vector fold crashes at
// width 1, but a scalar product per sorted row segment does not -- so sort handles sliceSize==1 here,
// no UB limit and no SIMT atomic CAS storm. Replaces the old fp16-overflowing SIMT-atomic fallback.
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::TrySortScalar1D(ADDR_T sliceSize,
                                                                                             ADDR_T varFirstDim)
{
    if (!(sliceSize == 1 && totalIdxn_ > 0 && varFirstDim > 0)) {
        return false;
    }
    SortScalarSliceSize1(totalIdxn_, varFirstDim);
    return true;
}

// MUL/MAX/MIN 的选路链。**顺序即优先级, 不可调换**:
// 🔴 UB 驻留必须排在 ss>=2 的原子切片与排序**之前** —— 它是最快的一条, 且入口是纯容量判定,
//    装不下自然回落。曾把它放在 ss>=2 分支之后, 结果 ss>=2 永远到不了(排序分支无条件 return)。
//
// 🔴 分桶路径最终否决(2026-10-28, 三轮实测 + 逐阶段截断定位), 故不在本链中; bucket.h 已删。
//   逐阶段耗时(L1_332, 完整 2641370us): 阶段1 2.6% / 阶段2+3 5.2% / **阶段4 92.1%**,
//   而阶段4 里 DMA 只占 0.1%, 92% 全在标量折叠循环体。
//   真因: 该例索引数据范围是 (0,1) —— 9782 万个索引全指向第 0/1 行, 全部命中落进
//   **1 个桶**(共 280 个), 那个桶归 1 个核 ⇒ 1 核串行折叠 9782 万命中, 55 核空转。
//   实测单核 24.9ns/命中, 比 A 类的 45ns/命中/核 还快 —— 折叠本身没问题, 是并行度坍塌。
//   **分桶按行分并行, 而不达标集全是热点行分布, 所以它结构性不适用。**
//   L0 证伪集 5 例实测(分桶/原子): L0_001 0.039x、L1_135 0.038x、L1_086 0.408x、
//   L1_263 1.065x、L1_332 1.215x —— 最好只 1.215x(判据需中位 1.8x)。
//   设计稿"命中为 0 时开销退化成扫描一遍, 与原子路径等价"这句**是错的**: 分桶要对 M
//   走两趟(计数 + 落 (row,value) 对), 外加 counts/lcnt 清零与 4 次集合性 SyncAll,
//   而原子路径只走一趟向量化筛选。零命中时前者全额付费、后者几乎免费。
//   被否决的尝试留痕: 桶号整除改移位(0.2%)、阶段4 整桶连续读(1.00x)、桶窗口压 32KB(未生效)。
//   正解是 UbResidentReduceRow1D 的按 M 分并行 + 判据放宽到"触达行区间装得下"
//   (44/44 例的触达区间只有 1~101 行)。
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::ProcessReduceModes(ADDR_T sliceSize,
                                                                                                ADDR_T varFirstDim)
{
    const UbBufSpec ubSpec = MakeUbSpec();
    if (TryScalarReduce1D(sliceSize, varFirstDim)) {
        return true;
    }
    if (TryUbResident(sliceSize, varFirstDim, ubSpec)) {
        return true;
    }
    if (TryBucketReduce(sliceSize, varFirstDim)) {
        return true;
    }
    if (TryAtomicSliceN(sliceSize, varFirstDim, ubSpec)) {
        return true;
    }
    if (TrySortSliceN(sliceSize, varFirstDim)) {
        return true;
    }
    if (TryAtomicRow1D(sliceSize, varFirstDim, ubSpec)) {
        return true;
    }
    return TrySortScalar1D(sliceSize, varFirstDim);
}

// sliceSize==1 small -> single-core scalar sequential divide (ScalarDiv1D: /0 keeps var for int, ->inf
// for float, reference-exact). Large -> sort + scalar row-fold (sequential divide, same /0 rule).
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline bool ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::ProcessDivMode(ADDR_T sliceSize,
                                                                                            ADDR_T varFirstDim)
{
    // 三条路径的公共前置条件先提出来: 空形态(M==0 或 varFirstDim==0)一律不接, 由调用方按 no-op 处理
    if (totalIdxn_ <= 0 || varFirstDim <= 0) {
        return false;
    }
    if (sliceSize == 1 && tiling_.scalarPath != 0) {
        if (blockIdx_ == 0) {
            ScalarDiv1D<PARAMS_T, INDICES_T, ADDR_T>(pipe_, idxGm_, xGm_, outputGm_, totalIdxn_, varFirstDim);
        }
        return true;
    }
    if (sliceSize >= 2) {
        SortDivRowProcess(totalIdxn_, sliceSize, varFirstDim); // sliceSize>=2: row-split in-place divide
        return true;
    }
    if (sliceSize == 1) {
        SortScalarSliceSize1(totalIdxn_, varFirstDim); // sliceSize==1 large: sort + scalar divide
        return true;
    }
    return false;
}

template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Process()
{
    if (tiling_.sliceSize == 0 || blockIdx_ >= tiling_.blockNum) {
        return;
    }
    const ADDR_T sliceSize = static_cast<ADDR_T>(tiling_.sliceSize);
    const ADDR_T varFirstDim = static_cast<ADDR_T>(tiling_.varFirstDim);
    if constexpr (Mode == MODE_MUL || Mode == MODE_MAX || Mode == MODE_MIN) {
        (void)ProcessReduceModes(sliceSize, varFirstDim);
    } else if constexpr (Mode == MODE_DIV) {
        (void)ProcessDivMode(sliceSize, varFirstDim);
    }
    // Empty (M==0 or varFirstDim==0): nothing to reduce -> fall through and return (no-op). No SIMT path remains.
}

#endif // SCATTER_REDUCE_COMMON_DISPATCH_H
