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
 * \file bn3d_training_reduce_tiling_data.h
 * \brief
 */
// 保护宏不以下划线开头：以 _ 接大写字母的标识符是 C++ 标准保留给实现的，
// 与本算子 op_graph/bn3d_training_reduce_proto.h 的命名口径保持一致。
#ifndef OPS_NORM_BN3D_TRAINING_REDUCE_TILING_DATA_H_
#define OPS_NORM_BN3D_TRAINING_REDUCE_TILING_DATA_H_

// DENSE_CHANNEL 路线 TilingData（两条 Kernel 分支共用）。
//
// 两种受支持的布局都归一化为同一个 R1-A-R0 模型，GM 线性下标同为
//   idx(r1, a, r0) = r1 * (numC * numR0) + a * numR0 + r0
// 因此搬运逻辑（跨 R1 用 srcStride 跳过其他通道、单 (r1, a) 的 R0 元素连续）完全共用：
//
//   * channel-first 密集布局（storage NCDHW / NCHW，TilingKey 100000，numC0 == 0）
//       R1 = N = dim0，A = C = dim1，R0 = product(dim2:)（rank 2 时 R0 = 1）。
//       每个通道归约成 1 个标量。
//
//   * C0 打包布局（storage NDC1HWC0 [N,D,C1,H,W,C0]，TilingKey 200000，numC0 == C0）
//       R1 = N * D，A = C1，R0 = H * W * C0。
//       归约保留 C1 与 C0 两轴，故每个 c1 归约成 numC0 个标量：先按 VL 宽累加
//       （lane L 恒对应 c0 = L % numC0，成立的前提是 numC0 整除 VL_FP32，Host 侧已校验），
//       再把累加器按 numC0 折叠。
//
// 与 INTrainingReduceV2 的关键差异：本算子跨 N 归约、不保留 N，因此每个通道只在
// 全部 R1 × R0 元素累加完毕后收尾一次，而不是每行一次。
struct BN3DTrainingReduceDenseChannelTilingData {
    uint64_t numN;        // R1：NCDHW 为 N；NDC1HWC0 为 N * D
    uint64_t numC;        // A ：NCDHW 为 C；NDC1HWC0 为 C1（有效逻辑通道数）
    uint64_t numR0;       // R0：单个 (r1, a) 的连续归约元素数
    uint64_t r0Align;     // R0 向上对齐到 VL_FP32 的元素数（UB 内行步长，单位为输入元素）
    uint64_t usedCoreNum; // 实际参与计算的核数
    uint64_t cPerCore;    // 每核负责的通道数（前 usedCoreNum-1 个核；尾核可能更少）
    uint64_t cRound;      // 每轮在 UB 暂存并一次性写回的通道数上限
    uint64_t nTile;       // 单次 DataCopyPad 载入的 R1 行数（跨 R1 用 srcStride 跳过其他通道）
    uint64_t isSubR;      // 0：R0 全载；1：R0 需分块（此时 nTile 恒为 1）
    uint64_t r0Factor;    // sub-R 分块的块大小（元素数，VL_FP32 整数倍）
    uint64_t numChunks;   // ceil(numR0 / r0Factor)，Host 保证 <= UINT32_MAX 后 Kernel 才窄化
    uint64_t tailLen;     // numR0 - (numChunks - 1) * r0Factor，尾块有效长度
    uint64_t numC0; // 0：每通道输出 1 个标量；> 0：每通道输出 numC0 个标量（NDC1HWC0 的 C0）
    // ── 多累加槽（缩短 fp32 累加依赖链，对齐竞品的树形归约精度）────────────────
    // 单槽时每个 lane 上是一条长度 = 总 tile 次数的线性依赖链，fp32 相对误差随链长
    // **线性**增长；竞品 torch.sum 走树形归约只随 log 增长。实测（200 条三方比对）
    // 比值与每通道归约元素数单调相关：<1e3 时中位 1.00，>1e6 时中位 215。
    //
    // 做法：把 tile 序列轮转着累进 numAccSlots 个独立槽，最后把各槽两两折叠归并。
    // 链长由 T 降到 T / numAccSlots + log2(numAccSlots)。
    // 槽轮转在**标量侧**完成（accUb + slot * 2 * VL_FP32），__VEC_SCOPE__ 内部一行不改，
    // 因此不触碰 VF 编程模型的两条约束（内部禁数据依赖分支、归纳变量须 uint16_t）。
    uint64_t numAccSlots; // 累加槽数，恒为 2 的幂；1 表示退化为原单槽行为
    uint64_t foldPasses;  // 两两折叠的趟数 = log2(numAccSlots)
};

#endif // OPS_NORM_BN3D_TRAINING_REDUCE_TILING_DATA_H_
