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
 * \file scatter_reduce_common_struct.h
 * \brief Shared definitions for the non-nd scatter reduce ops (ScatterMax/Min/Mul/Div).
 *        Independent implementation (does not depend on scatter_add). Modern direct-struct tiling.
 */
#ifndef SCATTER_REDUCE_COMMON_STRUCT_H
#define SCATTER_REDUCE_COMMON_STRUCT_H

#include <cstdint>

namespace ScatterReduceCommon {
// reduce mode, passed as a template parameter to the kernel
constexpr uint64_t MODE_MAX = 0;
constexpr uint64_t MODE_MIN = 1;
constexpr uint64_t MODE_MUL = 2;
constexpr uint64_t MODE_DIV = 3;

// tiling key template modes
// UB 访问与掩码均按 block 对齐; 该文件 host/kernel 共用, 故不引 AscendC 命名空间,
// 取值与 AscendC::ONE_BLK_SIZE 一致(kernel 侧有静态断言校验)。
constexpr uint32_t UB_BLOCK_BYTES = 32;
// 位掩码 1 bit/元素, 故一个 block 覆盖 UB_BLOCK_BYTES * 8 个元素。

// uint8 的 x^0x80 翻转每轮搬这么多字节, 需要 x/a/b 三块等宽暂存(共 3 倍)。
// host/kernel 共用这一个常量: host 的 UB 账本按 3 * 该值预留, 内核按该值开缓冲 ——
// 从前 host 写 3UL*16UL*1024UL、内核写自己的 XF_TILE, 是两份字面量, 改一边就对不上账。
constexpr uint32_t XF_FLIP_TILE_BYTES = 16U * 1024U;

constexpr uint64_t TPL_ADDR_32 = 0; // index address fits in uint32
constexpr uint64_t TPL_ADDR_64 = 1; // index address needs uint64

// updates 预取队列深度。内核 ScatterReduceSimt::PREFETCH_DEPTH 取自此处, host 的 UB 账本同源,
// 避免 host 写 1+8、内核写 8 的两份字面量。
constexpr uint64_t SORT_PREFETCH_DEPTH = 8;

// 排序折叠路径的 UB 账本: host 反解 chunkW 容量与内核开 buffer 共用这一份式子
// (此前两处各写一遍, 改一边即错配)。
struct SortFoldBufPlan {
    uint64_t colBytes; // 每列合计字节
    bool needRowAcc;   // 窄类型 widen 暂存 / 缩放域折叠的 updates 尾数
    bool needTmp;      // int8 符号扩展 / 缩放域折叠的指数累加(int32)
    bool scaledFold;   // 缩放域折叠: 另需指数增量(int32) + 两张比较掩码
};

// 账本必须如实反映**内核真实分配**: 三个折叠入口都无条件开 acc/var/rowAcc/tmp/updQue 五块。
// 按通路省 buffer 要内核那半一起改(条件化 InitBuffer), 只改这里会把容量算大 -> UB 越界。
constexpr SortFoldBufPlan SortFoldPlan(uint64_t paramBytes, uint64_t accBytes, uint64_t mode, bool isFloat)
{
    const bool scaledFold = isFloat && (mode == MODE_MUL || mode == MODE_DIV); // 尾数 x 整数指数双寄存
    uint64_t bytes = accBytes * 2UL + paramBytes * (1UL + SORT_PREFETCH_DEPTH) + sizeof(int32_t);
    bytes += scaledFold ? (sizeof(int32_t) + 1UL) : 0UL; // 指数增量 + 两张掩码(按 1B/列 计)
    return SortFoldBufPlan{bytes, true, true, scaledFold};
}

// SIMT atomic tiling data (direct struct; read in kernel via GET_TILING_DATA_WITH_STRUCT).
// semantics: for each index entry m, var[indices[m]] (a slice of sliceSize elems) is reduced with
// updates[m] by the reduce mode. The total update-element space (indicesNum * sliceSize) is split
// across cores (block tiling) then iterated in UB chunks.
struct ScatterReduceSimtTilingData {
    uint64_t blockNum;            // number of cores actually used
    uint64_t blockTilingSize;     // update elements handled per front core
    uint64_t tailBlockTilingSize; // update elements handled on the last used core
    uint64_t sliceSize;           // elements per index slice (product of var tail dims)
    uint64_t varFirstDim;         // var dim0 size (index bound)
    uint64_t scalarPath; // 1 = 整个 var + indices 能同时放进 UB, 走单核标量路径; host 按实测 UB 判定
    uint64_t sortTile;   // 排序分片元素数, host 按 UB 实测容量反解并 32 对齐(不写死)
    uint64_t ubChunkMax; // phase-3 列切分上限, host 按 UB 实测容量与每列存活开销反解
    // ↓ 原子/分桶路径的规格, 一律由 host 按实测 UB 反解下发, 内核不再写死任何预算常量。
    uint64_t atomicTile;  // 原子路径每轮搬入的索引数(idx + updates 两块缓冲由此定)
    uint64_t atomicSlots; // 一批在飞的 granule 槽数(批末等待由此摊薄)
    uint64_t bucketRows; // 分桶路径的桶内 var 行数(= UB 窗口 / 切片字节, 向下取 2 的幂), 0 表示不走分桶
    // 分桶路径的 UB 预算(字节)。必须由 host 下发而不能在内核里用 bucketRows*rowBytes 反推 ——
    // bucketRows 被向下取到 2 的幂(为了让"行号→桶号"退化成一次移位), 反推会把预算算小最多一半,
    // 于是暂存深度被压到门槛以下、该进分桶的用例被挡回老路(L1_086 实测 D 算成 3 而真值是 7)。
    uint64_t ubUsable;
    // ↓ 原子路径每块 UB 缓冲的**字节数**, 由 host 一次算准下发, 内核只透传给 InitBuffer。
    // 从前内核按 tile*sizeof(INDICES_T) / tile*sliceBytes / tile*sizeof(int32_t) 自己乘,
    // 而 host 的预算用 perTile = keyBytes + updCopies*sliceSize*varBytes + sizeof(int32_t)
    // 另算一遍 —— 同一笔账两套算式, 少算一个通道就 UB 越界(G2-kernel-initbuffer-selfsized)。
    // 现在 host 用同一组中间量既算预算又填这几个字段, 二者按构造一致。
    uint64_t atomicIdxBytes; // 索引缓冲: atomicTile * sizeof(索引 dtype)
    uint64_t atomicUpdBytes; // updates 缓冲: atomicTile * sliceSize * sizeof(var dtype)
    uint64_t atomicPosBytes; // 命中位置缓冲: atomicTile * sizeof(int32_t)
    // granule 槽宽(元素数)。内核此前自己算 (sliceSize + 2*GRAN + GRAN-1)/GRAN*GRAN,
    // 与 host ResolveAtomicUbSpec 的 slotSpan 是同一算式的两份拷贝(原注释明写"必须逐字一致")。
    // 改由 host 下发, 内核不再复刻该算式。
    uint64_t atomicSlotSpan;
    // 排序分片长度(条数): 排序 scratch 只按**一片**定尺, 与全量 M 无关。
    //
    // 原先 scratch 是 O(全量 M)(sortedIdx|originPos|scratchKeys|scratchPos, 每条目
    // 2*(keySz+posSz) 字节), 于是 workspace 随 M 线性膨胀, 把算子支持的更新条数
    // 封在 预算/32B 以内(实测 61.7GiB 预算 => M 上限约 2^31, 而官方用例 b053/b051/
    // bound_034 的 M 正好是 2^31, 天花板已被踩穿)。改为分片后 workspace 恒定,
    // M 多大都切成若干片依次排序+折叠, 没有隐含上限。
    // 片长由 host 按固定预算反解下发, 内核不自推(与 ubUsable/atomic* 同一范式)。
    uint64_t sortChunkM;
    // 选路由 host 判定并下发, 内核只照此执行 —— 入口条件集中在一处、可审计, 且结果随 tiling data
    // 出现在跑批输出里, 可以逐例证明"这条用例确实走了那条路"(此前只能靠耗时反推, 曾因此漏掉
    // uint8 门控没放行、优化代码全是死代码的问题)。
    //   0 = 单核 scalar   1 = 原子 sliceSize==1   2 = 原子 sliceSize>=2
    //   3 = 按行归属分桶   4 = 排序(行折叠/列折叠)
    uint64_t pathId;
    // 并行轴: 取两条轴"单核搬运字节"较小者(host 按成本模型推导, 非阈值)。
    // 列轴无偏积无集合同步但参与核数受列数限制; 位置轴全核参与但偏积要往返。0 = 位置轴  1 = 列轴
    uint64_t parallelAxis;
    // 列轴切分: 先定并行粒度(每核一段连续列), 再由 UB 容量决定段内切几刀。
    // 顺序反了(先顶满 UB 再发列块)会让尾轮只有部分核在干活。
    uint64_t colsPerCore; // 每核负责的列数 = ceil(sliceSize / 列轴参与核数)
    uint64_t colChunk;    // 核内每轮处理的列数 = ceil(colsPerCore / ceil(colsPerCore / ubChunkMax))
};
} // namespace ScatterReduceCommon

#endif // SCATTER_REDUCE_COMMON_STRUCT_H
