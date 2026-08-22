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
 * \file bn3d_training_reduce_dense_channel.h
 * \brief DENSE_CHANNEL 路线：通道独占，跨 R1 与 R0 归约。两种布局共用同一套搬运逻辑。
 *
 * 两种受支持布局的 GM 下标同为 idx(r1, a, r0) = r1 * (numC * numR0) + a * numR0 + r0，
 * 因此搬运（跨 R1 用 srcStride 跳过其他通道、单 (r1, a) 的 R0 元素连续）完全共用，
 * 仅收尾方式不同，由模板参数 C0_PACKED 在编译期分派：
 *
 *   * C0_PACKED == false（storage NCDHW / NCHW，TilingKey 100000）
 *       R1 = N，A = C，R0 = product(dim2:)；每通道收尾做一次水平 ReduceSum 得 1 个标量。
 *
 *   * C0_PACKED == true（storage NDC1HWC0 = [N,D,C1,H,W,C0]，TilingKey 200000）
 *       R1 = N * D，A = C1，R0 = H * W * C0；归约保留 C1 与 C0 两轴。
 *       因 numC0 整除 VL_FP32（Host 侧已校验），VL 宽累加器的 lane L 恒对应 c0 = L % numC0，
 *       故收尾不是水平全归约，而是把累加器按 numC0 折叠成 numC0 个标量。
 *       ⚠ 折叠的载入 / 写回偏移只按 numC0 递进（k * numC0、slot * numC0，单位 fp32），
 *       故还要求 numC0 * sizeof(float) 是 UB block 的整数倍，否则向量访存非对齐、真机
 *       直接 VEC_ERROR。这一条同样由 Host 侧校验（见 tiling_dense_channel 的 DoOpTiling），
 *       两条合起来把 numC0 限定为 {8, 16, 32, 64}。改动此处折叠逻辑时必须同步该校验。
 *
 * 与 INTrainingReduceV2 的结构差异：
 *   INTrainingReduceV2 输出 per-(n,c)，每行都要做一次水平 ReduceSum；
 *   本算子不保留 N，因此用一对 VL 宽的 fp32 向量累加器跨 R1 和 R0 累加，
 *   整个通道结束后才收尾一次。累加器跨 tile 存活在 UB（RegTensor 出了
 *   __VEC_SCOPE__ 就失效），进出 scope 用 DataCopy<float, DIST_NORM> 全宽读写。
 *
 * Σx 折叠原始 x，Σx² 先逐元素平方再折叠——(a+b)² ≠ a²+b²，两者必须各自持有独立累加器。
 * fp16 / bf16 输入先提升 fp32 再平方，禁止低精度下先平方。
 */
#ifndef BN3D_TRAINING_REDUCE_DENSE_CHANNEL_H_
#define BN3D_TRAINING_REDUCE_DENSE_CHANNEL_H_

#include "bn3d_training_reduce_common.h"

namespace BN3DTrainingReduceOps {
using namespace AscendC;

constexpr uint32_t ALIGN_32_FACTOR = 32;
// 累加器缓存：前 VL_FP32 个 fp32 存 Σx，后 VL_FP32 个存 Σx²，两段互不重叠。
constexpr uint32_t ACC_SLOT_NUM = 2;
// 实际申请的槽数比逻辑槽数多 1：C0 折叠要从 accUb + VL_FP32 + k * numC0 处全宽（VL_FP32）
// 载入，最远读到 3 * VL_FP32 - numC0，多留一个 VL 宽的保护槽以免越出 accBuf_。
// Host 侧 accBytes 按同一常量计算，两侧必须一致。
constexpr uint32_t ACC_GUARD_SLOT_NUM = 1;
// 每个累加槽配一份等宽的 Kahan 补偿量，故 accBuf_ 里累加器区与补偿区各占一半。
constexpr uint32_t NUM_ACC_AND_COMP = 2;
// Kahan 补偿求和的分块大小，单位是 VL 宽 chunk。块内走朴素累加（每 chunk 2 条 Add），
// 块尾才做一次 Kahan 合并（12 条），把补偿开销摊薄 K 倍。
//
// 取 8 的依据是**判据本身**：cross_check 比的是"本算子误差 ÷ A100 竞品误差"，竞品
// torch.sum 走树形归约、误差约 O(log2(N)·ε)。分块 Kahan 的误差是 O(K·ε)（块内链长 K，
// 块间 Kahan 与块数无关），故只要 K <= log2(N) 就与竞品同档。实测失败用例的链长中位数
// 490（log2 ≈ 9）、最长约 61000（log2 ≈ 16），K = 8 全覆盖且留一档余量。
// 再大（如 16）能省一半补偿开销但误差贴到竞品同量级，rmse 是绝对误差之比、余量太薄。
constexpr uint16_t KAHAN_BLOCK_CHUNKS = 8;

template <typename T_X, bool C0_PACKED>
class BN3DTrainingReduceDenseChannel {
public:
    using T_SUM = float; // 两个输出恒 fp32

    __aicore__ inline explicit BN3DTrainingReduceDenseChannel(
        const BN3DTrainingReduceDenseChannelTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();

        numN_ = tilingData->numN;
        numC_ = tilingData->numC;
        numR0_ = tilingData->numR0;
        r0Align_ = tilingData->r0Align;
        usedCoreNum_ = tilingData->usedCoreNum;
        cPerCore_ = tilingData->cPerCore;
        cRound_ = tilingData->cRound;
        nTile_ = static_cast<uint32_t>(tilingData->nTile);
        isSubR_ = tilingData->isSubR;
        r0Factor_ = static_cast<uint32_t>(tilingData->r0Factor);
        numChunks_ = static_cast<uint32_t>(tilingData->numChunks);
        tailLen_ = static_cast<uint32_t>(tilingData->tailLen);
        // 每通道产出的 fp32 个数：非 C0 打包恒为 1；C0 打包为 numC0。
        outPerChannel_ = C0_PACKED ? static_cast<uint32_t>(tilingData->numC0) : 1U;
        // 整通道能被一次 DataCopyPad 搬完（非 sub-R 且 nTile 覆盖全部 R1）→ 走融合快路。
        // 此时 numN_ <= nTile_ <= BLOCK_COUNT_MAX(65535)，故可安全窄化为 uint16_t 作行数。
        fusedSingleTile_ = (isSubR_ == 0) && (nTile_ >= numN_);
        // 多累加槽：把 tile 序列轮转着累进 numAccSlots_ 个独立槽，最后两两折叠归并，
        // 把 fp32 线性依赖链由 T 降到 T / numAccSlots_ + log2(numAccSlots_)。
        // Host 保证 numAccSlots_ 是 2 的幂且 >= 1（含 C == 0 的 no-work 分支）。
        numAccSlots_ = static_cast<uint32_t>(tilingData->numAccSlots == 0 ? 1 : tilingData->numAccSlots);
        foldPasses_ = static_cast<uint16_t>(tilingData->foldPasses);
        // 融合快路整通道只进一次 VF scope、累加器全程在寄存器里，本就没有跨 tile 的
        // UB 累加器可轮转，故不参与多槽（它对应的也正是链最短的小通道场景）。
        if (fusedSingleTile_) {
            numAccSlots_ = 1;
            foldPasses_ = 0;
        }
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum)
    {
        // C == 0 时 Host 已短路，此处不会有归约任务；仍做防御性返回。
        if (usedCoreNum_ == 0) {
            return;
        }

        const uint64_t gmLen = numN_ * numC_ * numR0_;
        // 输出元素数：非 C0 打包为 C（形状 [C]）；C0 打包为 C1 * C0（形状 [1,1,C1,1,1,C0]）。
        const uint64_t outLen = numC_ * outPerChannel_;
        xGm_.SetGlobalBuffer((__gm__ T_X*)x, gmLen);
        sumGm_.SetGlobalBuffer((__gm__ T_SUM*)sum, outLen);
        squareSumGm_.SetGlobalBuffer((__gm__ T_SUM*)squareSum, outLen);

        // 输入 tile：全载路径为 nTile 行 × r0Align；sub-R 路径为单行的一个 r0Factor 分块。
        const uint64_t inTileElems = (isSubR_ != 0) ? static_cast<uint64_t>(r0Factor_) : (nTile_ * r0Align_);
        pipe_.InitBuffer(inQueueX_, DOUBLE_BUFFER_NUM,
                         ops::CeilAlign(inTileElems * sizeof(T_X), static_cast<uint64_t>(BLOCK_SIZE)));
        const uint64_t outTileBytes = ops::CeilAlign(cRound_ * outPerChannel_ * sizeof(T_SUM),
                                                     static_cast<uint64_t>(BLOCK_SIZE));
        pipe_.InitBuffer(outQueueSum_, DOUBLE_BUFFER_NUM, outTileBytes);
        pipe_.InitBuffer(outQueueSquareSum_, DOUBLE_BUFFER_NUM, outTileBytes);
        pipe_.InitBuffer(accBuf_, AccBufSlotNum() * VL_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (usedCoreNum_ == 0 || static_cast<uint64_t>(blockIdx_) >= usedCoreNum_) {
            return;
        }

        // 按通道分核：一个通道只由一个核负责，写回责任唯一，无跨核归并。
        const uint64_t cStart = static_cast<uint64_t>(blockIdx_) * cPerCore_;
        if (cStart >= numC_) {
            return;
        }
        const uint64_t cEnd = (cStart + cPerCore_ > numC_) ? numC_ : (cStart + cPerCore_);

        LocalTensor<float> accLocal = accBuf_.Get<float>();
        __local_mem__ float* accUb = (__local_mem__ float*)accLocal.GetPhyAddr();

        for (uint64_t cBase = cStart; cBase < cEnd; cBase += cRound_) {
            const uint32_t curRound = static_cast<uint32_t>(((cBase + cRound_) > cEnd) ? (cEnd - cBase) : cRound_);

            LocalTensor<T_SUM> sumLocal = outQueueSum_.AllocTensor<T_SUM>();
            LocalTensor<T_SUM> squareSumLocal = outQueueSquareSum_.AllocTensor<T_SUM>();
            __local_mem__ float* sumUb = (__local_mem__ float*)sumLocal.GetPhyAddr();
            __local_mem__ float* squareSumUb = (__local_mem__ float*)squareSumLocal.GetPhyAddr();

            for (uint32_t j = 0; j < curRound; ++j) {
                const uint64_t c = cBase + j;
                // 单 tile 且非 C0 打包时走融合快路，省掉 2 次 __VEC_SCOPE__ 进出与累加器
                // 的 UB 往返（小通道场景下这些固定开销是主导项，见 ProcessChannelFused）。
                if constexpr (!C0_PACKED) {
                    if (fusedSingleTile_) {
                        ProcessChannelFused(c, sumUb, squareSumUb, j);
                        continue;
                    }
                }
                ResetAcc(accUb);
                if (isSubR_ != 0) {
                    AccumulateChannelSubR(c, accUb);
                } else {
                    AccumulateChannelFull(c, accUb);
                }
                // 多槽时先把 numAccSlots_ 个槽两两折叠回槽 0，收尾逻辑保持不变。
                FoldAccSlots(accUb);
                if constexpr (C0_PACKED) {
                    FinalizeChannelC0(accUb, sumUb, squareSumUb, j);
                } else {
                    FinalizeChannel(accUb, sumUb, squareSumUb, j);
                }
            }

            outQueueSum_.EnQue<T_SUM>(sumLocal);
            outQueueSquareSum_.EnQue<T_SUM>(squareSumLocal);
            CopyOutSumSquareSum(cBase, curRound);
        }
    }

private:
    // ---------- R0 全载：一次 DataCopyPad 搬 cnt 个 n 行 ----------
    // 同一通道跨 N 的步长是 numC * numR0，用 srcStride 跳过其他通道，避免逐行发搬运。
    //
    // 两个参数的正确性都不显然，各记一条：
    //
    // ① srcStride 窄化为 uint32 是安全的。Host 侧在 (numC-1)*numR0*sizeof(T_X) 超出
    //    UINT32_MAX 时会把 nTile 压到 1（见 tiling 的 srcStrideBytes 判断），而
    //    blockCount == 1 时硬件根本不使用 srcStride，被截断的值不参与寻址。
    //
    // ② dstStride 的整除不会丢精度。DataCopyPad 的目的地按 CeilAlign(blockLen, 32B)
    //    递进，故正确值应为 (r0Align*s - CeilAlign(numR0*s, 32)) / 32；这里写的是
    //    (r0Align - numR0)*s / 32。两者相等，依据是恒等式
    //        floor((A - B) / 32) == A/32 - ceil(B/32)      当 A ≡ 0 (mod 32)
    //    而 r0Align 是 VL_FP32(64) 的整数倍、s ∈ {2,4}，故 A = r0Align*s 恒为 128 的
    //    整数倍，前提成立。R0 非对齐的用例由 tiling UT 019 与泛化档看护。
    __aicore__ inline void CopyInRows(uint64_t c, uint64_t nBase, uint16_t cnt)
    {
        LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
        const uint64_t offset = nBase * numC_ * numR0_ + c * numR0_;
        DataCopyExtParams extParams{
            cnt,                                                                        // blockCount
            static_cast<uint32_t>(numR0_ * sizeof(T_X)),                                // blockLen
            static_cast<uint32_t>((numC_ - 1) * numR0_ * sizeof(T_X)),                  // srcStride
            static_cast<uint32_t>((r0Align_ - numR0_) * sizeof(T_X) / ALIGN_32_FACTOR), // dstStride
            0                                                                           // rsv
        };
        DataCopyPadExtParams<T_X> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                            static_cast<T_X>(0)};
        DataCopyPad(xLocal, xGm_[offset], extParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void AccumulateChannelFull(uint64_t c, __local_mem__ float* accUb)
    {
        uint32_t slot = 0;
        for (uint64_t nBase = 0; nBase < numN_; nBase += nTile_) {
            const uint16_t cnt = static_cast<uint16_t>(((nBase + nTile_) > numN_) ? (numN_ - nBase) : nTile_);
            CopyInRows(c, nBase, cnt);
            LocalTensor<T_X> xLocal = inQueueX_.DeQue<T_X>();
            __local_mem__ T_X* xUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
            // 槽轮转在标量侧完成，AccumulateRows 的 __VEC_SCOPE__ 内部一行不改。
            AccumulateRows(xUb, AccSlot(accUb, slot), CompSlot(accUb, slot), cnt, static_cast<uint32_t>(r0Align_),
                           static_cast<uint32_t>(numR0_));
            inQueueX_.FreeTensor(xLocal);
            slot = NextSlot(slot);
        }
    }

    // ---------- sub-R：单行 R0 超单次 UB 容量，按 r0Factor 分块搬入并累加 ----------
    __aicore__ inline void AccumulateChannelSubR(uint64_t c, __local_mem__ float* accUb)
    {
        uint32_t slot = 0;
        for (uint64_t n = 0; n < numN_; ++n) {
            const uint64_t rowOffset = n * numC_ * numR0_ + c * numR0_;
            for (uint32_t k = 0; k < numChunks_; ++k) {
                const uint32_t curLen = (k == numChunks_ - 1) ? tailLen_ : r0Factor_;
                LocalTensor<T_X> xLocal = inQueueX_.AllocTensor<T_X>();
                DataCopyExtParams extParams{
                    1,                                           // blockCount
                    static_cast<uint32_t>(curLen * sizeof(T_X)), // blockLen
                    0,                                           // srcStride
                    0,                                           // dstStride
                    0                                            // rsv
                };
                DataCopyPadExtParams<T_X> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                    static_cast<T_X>(0)};
                DataCopyPad(xLocal, xGm_[rowOffset + static_cast<uint64_t>(k) * r0Factor_], extParams, padParams);
                inQueueX_.EnQue(xLocal);

                LocalTensor<T_X> xIn = inQueueX_.DeQue<T_X>();
                __local_mem__ T_X* xUb = (__local_mem__ T_X*)xIn.GetPhyAddr();
                AccumulateRows(xUb, AccSlot(accUb, slot), CompSlot(accUb, slot), 1, 0, curLen);
                inQueueX_.FreeTensor(xIn);
                slot = NextSlot(slot);
            }
        }
    }

    // ---------- Kahan 分块参数：把 chunks 切成 numBlocks 块，块 0 吃掉不整除的余数 ----------
    // 三个量全在 __VEC_SCOPE__ **外**算好，scope 内只剩定长循环，不引入任何分支
    // （★VF 约束 A★：scope 内的数据无关分支会让 codegen 出错，见 02_design.md §4.5）。
    //
    //   numBlocks = max(1, chunks / K)
    //   period    = chunks / numBlocks            后续块的 chunk 数
    //   period0   = chunks - period * (numBlocks-1)   块 0 的 chunk 数，>= period
    //
    // 为什么把余数并进块 0 而不是另起一个尾块：尾块要么 trip count 与其他块不同
    // （= 分支），要么用空转 iteration 补齐（会越界读到下一行、且读未初始化 UB）。
    // 块 0 因 chunks >= 1 恒被执行，可以直接从循环里剥出来写，三者皆免。
    //
    // 验算：chunks=956 → (119, 8, 12)；chunks=128 → (16, 8, 8)；chunks=7 → (1, 7, 7)；
    //       chunks=1 → (1, 1, 1)，退化为"每行合并一次"，正是小 R0 场景想要的。
    struct KahanSplit {
        uint16_t numBlocks;
        uint16_t period;
        uint16_t period0;
    };

    // ---------- ★补偿量的非有限值清零：守住 DFX 的 IEEE 754 传播契约★ ----------
    // Kahan 的补偿项 c = (t - acc) - y 在输入含 ±inf 时必然退化为 nan：
    //   blk = +inf → y = inf - 0 = inf → t = 0 + inf = inf
    //             → c = (inf - 0) - inf = inf - inf = nan
    // 下一块 y = blk - c = inf - nan = nan，acc 随即被污染成 nan。
    // 实测代价：DFX 的 16 条 inf/nan 用例挂掉 12 条 —— all_pinf 应得 (+inf, +inf)
    // 却得到 (nan, nan)，all_ninf / mix_inf 同理。本算子的行为契约是 IEEE 754 默认
    // 传播（内核不做 isnan/isinf 分支），这属于真回归，不是判据问题。
    //
    // 修法：c 非有限时置 0。语义上成立 —— 累加器已是 ±inf 或 nan 时，"上一次加法丢掉的
    // 低位"本就没有意义，补 0 即"不补偿"，而 acc 自身不动，inf/nan 照常沿 acc 传播。
    //
    // 为什么自比较 (c == c) 就够、不必额外判 ±inf：穷举 c = (t - acc) - y 的所有非有限
    // 组合（acc 或 y 为 ±inf / nan）末步恒是 inf-inf 或 nan-x，结果**恒为 nan**，
    // c 取不到 ±inf。故一次 EQ 自比较即可精确识别，省掉 Abs + 阈值比较那一套。
    //
    // 这是向量 select（逐 lane 生效），不是控制流分支，不触碰 ★VF 约束 A★。
    // 成本：每块每个累加器 +2 条（Compare + Select），K=8 时摊到每 chunk 约 +0.5 条。
    __aicore__ inline void SanitizeComp(RegTensor<float>& comp, RegTensor<float>& zeroReg, MaskReg& pregFull)
    {
        MaskReg isFinite;
        Compare<float, CMPMODE::EQ>(isFinite, comp, comp, pregFull);
        Select(comp, comp, zeroReg, isFinite);
    }

    __aicore__ inline KahanSplit MakeKahanSplit(uint16_t chunks) const
    {
        uint16_t numBlocks = static_cast<uint16_t>(chunks / KAHAN_BLOCK_CHUNKS);
        if (numBlocks == 0) {
            numBlocks = 1;
        }
        const uint16_t period = static_cast<uint16_t>(chunks / numBlocks);
        const uint16_t period0 = static_cast<uint16_t>(chunks - period * (numBlocks - 1));
        return KahanSplit{numBlocks, period, period0};
    }

    // ---------- VF：把 UB 中 rows 行、每行 validLen 个有效元素累进 UB 累加器 ----------
    // ★分块 Kahan 补偿求和：块内朴素累加，块尾一次补偿合并★
    //
    // 朴素累加的相对误差是 O(N·ε)（N = 累加链长）。本算子唯一剩下的精度瓶颈就是这条链：
    // 实测仍失败的用例 100% 由单次调用内部的 rows*chunks 链主导（中位数 490），
    // 多累加槽只能切跨调用的那一段、切不到它。
    //
    // 逐 chunk 做 Kahan 能把误差压到 O(ε)，但每 chunk 要多 10 条向量指令（内层由 ~6 条
    // 涨到 ~16 条），实测大用例慢 2.4~3.1×、性能腿最小 folded G/N 由 0.224 掉到 0.076
    // 破了 0.1 门限。"归约是 memory-bound，多出来的向量指令会被访存掩盖"这个预期只
    // 对小用例成立；大用例本来就是向量发射受限。
    //
    // 故改为分块：块内 K 个 chunk 走朴素 Add（2 条/chunk），块尾把块和 blk 用 Kahan
    // 合并进 acc（12 条/块）。摊到每 chunk 是 2 + 12/K，K=8 时补偿开销只剩 1/8。
    // 误差 = O(K·ε)（块内链长 K，块间 Kahan 与块数无关），与总链长无关这一点不变。
    //
    //   y = blk - c;  t = acc + y;  c = (t - acc) - y;  acc = t
    //
    // c 是"上一次加法丢掉的低位"，下一次先补回去。c 必须跨 tile 存活，故与累加器
    // 一样常驻 UB（RegTensor 出了 __VEC_SCOPE__ 即失效）。
    // (t - acc) 算完先落回 c 自身当临时量再减 y —— 此时旧 c 已被 y 消费掉，省一个
    // RegTensor，10 个寄存器就够，降低寄存器 spill 风险。
    //
    // `acc = t` 这一步用 Adds(acc, t, 0.0f) 实现寄存器搬移：加 0 对有限值逐位精确，
    // 对 inf / nan 也按 IEEE 原样保留。
    // ⚠ 但**仅审这一步不足以保住 DFX 契约** —— 真正的破坏点在补偿量 c 上：
    // c = (t - acc) - y 遇到 ±inf 会退化成 nan，回灌下一块把 acc 也污染成 nan，
    // 实测挂掉 16 条 inf/nan 用例里的 12 条。故每次合并后必须调 SanitizeComp() 把
    // 非有限的 c 清零，见该函数的说明。
    __aicore__ inline void AccumulateRows(__local_mem__ T_X* xUb, __local_mem__ float* accUb,
                                          __local_mem__ float* compUb, uint16_t rows, uint32_t rowStride,
                                          uint32_t validLen)
    {
        const uint16_t chunks = static_cast<uint16_t>((validLen + VL_FP32 - 1) / VL_FP32);
        const KahanSplit sp = MakeKahanSplit(chunks);
        const uint16_t numBlocks = sp.numBlocks;
        const uint16_t period = sp.period;
        const uint16_t period0 = sp.period0;
        __VEC_SCOPE__
        {
            RegTensor<float> accSum;
            RegTensor<float> accSq;
            RegTensor<float> cSum;   // Σx  的补偿量
            RegTensor<float> cSq;    // Σx² 的补偿量
            RegTensor<float> blkSum; // 块内 Σx  的朴素累加器
            RegTensor<float> blkSq;  // 块内 Σx² 的朴素累加器
            RegTensor<float> x;
            RegTensor<float> xSquare;
            RegTensor<float> y;       // blk - c
            RegTensor<float> t;       // acc + y
            RegTensor<float> zeroReg; // SanitizeComp 的常量 0
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();

            Duplicate(zeroReg, static_cast<float>(0.0), pregFull);
            DataCopy<float, LoadDist::DIST_NORM>(accSum, accUb);
            DataCopy<float, LoadDist::DIST_NORM>(accSq, accUb + VL_FP32);
            DataCopy<float, LoadDist::DIST_NORM>(cSum, compUb);
            DataCopy<float, LoadDist::DIST_NORM>(cSq, compUb + VL_FP32);

            for (uint16_t i = 0; i < rows; ++i) {
                const uint32_t baseOffset = i * rowStride;
                uint32_t width = validLen; // UpdateMask 按引用逐次消耗，故每行重置
                uint16_t chunkBase = 0;    // 本行已消费的 chunk 数

                // ── 块 0：period0 个 chunk（含余数）。chunks >= 1 故必然执行，无需判空 ──
                Duplicate(blkSum, static_cast<float>(0.0), pregFull);
                Duplicate(blkSq, static_cast<float>(0.0), pregFull);
                for (uint16_t k = 0; k < period0; ++k) {
                    MaskReg preg = UpdateMask<float>(width);
                    LoadTensorForDtypeTIn<T_X>(xUb, x, preg, baseOffset + (chunkBase + k) * VL_FP32);
                    // 尾块无效 lane 先清零，再以全掩码累加。
                    // 不能直接用 preg 累加：ZEROING 语义会把累加器在无效 lane 上清掉。
                    ShiftLefts((RegTensor<uint32_t>&)x, (RegTensor<uint32_t>&)x, static_cast<int16_t>(0), preg);
                    Mul(xSquare, x, x, pregFull); // 已清零的 lane 平方后仍为 0，不污染 Σx²
                    Add(blkSum, blkSum, x, pregFull);
                    Add(blkSq, blkSq, xSquare, pregFull);
                }
                chunkBase = static_cast<uint16_t>(chunkBase + period0);
                // Σx：把块和 Kahan 合并进累加器
                Sub(y, blkSum, cSum, pregFull);
                Add(t, accSum, y, pregFull);
                Sub(cSum, t, accSum, pregFull);
                Sub(cSum, cSum, y, pregFull);
                SanitizeComp(cSum, zeroReg, pregFull);
                Adds(accSum, t, static_cast<float>(0.0), pregFull);
                // Σx²：独立累加器、独立补偿量
                Sub(y, blkSq, cSq, pregFull);
                Add(t, accSq, y, pregFull);
                Sub(cSq, t, accSq, pregFull);
                Sub(cSq, cSq, y, pregFull);
                SanitizeComp(cSq, zeroReg, pregFull);
                Adds(accSq, t, static_cast<float>(0.0), pregFull);

                // ── 块 1 .. numBlocks-1：每块 period 个 chunk。numBlocks == 1 时空转 ──
                for (uint16_t b = 1; b < numBlocks; ++b) {
                    Duplicate(blkSum, static_cast<float>(0.0), pregFull);
                    Duplicate(blkSq, static_cast<float>(0.0), pregFull);
                    for (uint16_t k = 0; k < period; ++k) {
                        MaskReg preg = UpdateMask<float>(width);
                        LoadTensorForDtypeTIn<T_X>(xUb, x, preg, baseOffset + (chunkBase + k) * VL_FP32);
                        ShiftLefts((RegTensor<uint32_t>&)x, (RegTensor<uint32_t>&)x, static_cast<int16_t>(0), preg);
                        Mul(xSquare, x, x, pregFull);
                        Add(blkSum, blkSum, x, pregFull);
                        Add(blkSq, blkSq, xSquare, pregFull);
                    }
                    chunkBase = static_cast<uint16_t>(chunkBase + period);
                    Sub(y, blkSum, cSum, pregFull);
                    Add(t, accSum, y, pregFull);
                    Sub(cSum, t, accSum, pregFull);
                    Sub(cSum, cSum, y, pregFull);
                    SanitizeComp(cSum, zeroReg, pregFull);
                    Adds(accSum, t, static_cast<float>(0.0), pregFull);

                    Sub(y, blkSq, cSq, pregFull);
                    Add(t, accSq, y, pregFull);
                    Sub(cSq, t, accSq, pregFull);
                    Sub(cSq, cSq, y, pregFull);
                    SanitizeComp(cSq, zeroReg, pregFull);
                    Adds(accSq, t, static_cast<float>(0.0), pregFull);
                }
            }

            DataCopy<float, StoreDist::DIST_NORM>(accUb, accSum, pregFull);
            DataCopy<float, StoreDist::DIST_NORM>(accUb + VL_FP32, accSq, pregFull);
            DataCopy<float, StoreDist::DIST_NORM>(compUb, cSum, pregFull);
            DataCopy<float, StoreDist::DIST_NORM>(compUb + VL_FP32, cSq, pregFull);
        }
    }

    // ---------- 单 tile 融合快路（仅 channel-first）----------
    // 整个通道一次 DataCopyPad 就搬完时，把 reset / 累加 / 收尾合并进**一个** __VEC_SCOPE__，
    // 累加器全程留在寄存器里，不经 UB 往返。
    //
    // 动因是 scalar 瓶颈而非算力：常规路径每通道要进 3 次 __VEC_SCOPE__（ResetAcc /
    // AccumulateRows / FinalizeChannel），每次进出都是一次流水同步。通道很小时这些固定
    // 开销完全盖过实际计算——msprof 实测 (2,512,4,8)/fp32 上 aiv_scalar_ratio 高达 0.551、
    // aiv_vec_ratio 仅 0.303（健康用例分别是 0.065 / 0.900），80.9us 跑 32K 个元素。
    // 融合后省掉 2 次 scope 进出、3 次整宽 store（ResetAcc）与 4 次 UB 读写（累加器往返）。
    //
    // 只对 C0_PACKED == false 启用：C0 折叠要从 accUb + VL_FP32 + k * numC0 处按 numC0
    // 递进地整宽载入，依赖累加器落在 UB 上，寄存器版做不到。
    __aicore__ inline void ProcessChannelFused(uint64_t c, __local_mem__ float* sumUb, __local_mem__ float* squareSumUb,
                                               uint32_t slot)
    {
        const uint16_t rows = static_cast<uint16_t>(numN_);
        CopyInRows(c, 0, rows);
        LocalTensor<T_X> xLocal = inQueueX_.DeQue<T_X>();
        __local_mem__ T_X* xUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();

        const uint32_t rowStride = static_cast<uint32_t>(r0Align_);
        const uint32_t validLen = static_cast<uint32_t>(numR0_);
        const uint16_t chunks = static_cast<uint16_t>((validLen + VL_FP32 - 1) / VL_FP32);
        const KahanSplit sp = MakeKahanSplit(chunks);
        const uint16_t numBlocks = sp.numBlocks;
        const uint16_t period = sp.period;
        const uint16_t period0 = sp.period0;
        __VEC_SCOPE__
        {
            RegTensor<float> accSum;
            RegTensor<float> accSq;
            RegTensor<float> cSum;   // Kahan 补偿量：本路整通道在一个 scope 内跑完，
            RegTensor<float> cSq;    // 故补偿量全程留在寄存器里，不占 UB
            RegTensor<float> blkSum; // 块内朴素累加器，口径与 AccumulateRows 完全一致
            RegTensor<float> blkSq;
            RegTensor<float> x;
            RegTensor<float> xSquare;
            RegTensor<float> y;
            RegTensor<float> t;
            RegTensor<float> vSum;
            RegTensor<float> vSquare;
            RegTensor<float> zeroReg; // SanitizeComp 的常量 0
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            Duplicate(accSum, static_cast<float>(0.0), pregFull);
            Duplicate(accSq, static_cast<float>(0.0), pregFull);
            Duplicate(cSum, static_cast<float>(0.0), pregFull);
            Duplicate(cSq, static_cast<float>(0.0), pregFull);
            Duplicate(zeroReg, static_cast<float>(0.0), pregFull);

            for (uint16_t i = 0; i < rows; ++i) {
                const uint32_t baseOffset = i * rowStride;
                uint32_t width = validLen; // UpdateMask 按引用逐次消耗，故每行重置
                uint16_t chunkBase = 0;

                // 块 0：period0 个 chunk（含余数），chunks >= 1 故必然执行
                Duplicate(blkSum, static_cast<float>(0.0), pregFull);
                Duplicate(blkSq, static_cast<float>(0.0), pregFull);
                for (uint16_t k = 0; k < period0; ++k) {
                    MaskReg preg = UpdateMask<float>(width);
                    LoadTensorForDtypeTIn<T_X>(xUb, x, preg, baseOffset + (chunkBase + k) * VL_FP32);
                    // 尾块无效 lane 先清零，再以全掩码累加（理由同 AccumulateRows）。
                    ShiftLefts((RegTensor<uint32_t>&)x, (RegTensor<uint32_t>&)x, static_cast<int16_t>(0), preg);
                    Mul(xSquare, x, x, pregFull);
                    Add(blkSum, blkSum, x, pregFull);
                    Add(blkSq, blkSq, xSquare, pregFull);
                }
                chunkBase = static_cast<uint16_t>(chunkBase + period0);
                Sub(y, blkSum, cSum, pregFull);
                Add(t, accSum, y, pregFull);
                Sub(cSum, t, accSum, pregFull);
                Sub(cSum, cSum, y, pregFull);
                SanitizeComp(cSum, zeroReg, pregFull);
                Adds(accSum, t, static_cast<float>(0.0), pregFull);

                Sub(y, blkSq, cSq, pregFull);
                Add(t, accSq, y, pregFull);
                Sub(cSq, t, accSq, pregFull);
                Sub(cSq, cSq, y, pregFull);
                SanitizeComp(cSq, zeroReg, pregFull);
                Adds(accSq, t, static_cast<float>(0.0), pregFull);

                // 块 1 .. numBlocks-1：每块 period 个 chunk
                for (uint16_t b = 1; b < numBlocks; ++b) {
                    Duplicate(blkSum, static_cast<float>(0.0), pregFull);
                    Duplicate(blkSq, static_cast<float>(0.0), pregFull);
                    for (uint16_t k = 0; k < period; ++k) {
                        MaskReg preg = UpdateMask<float>(width);
                        LoadTensorForDtypeTIn<T_X>(xUb, x, preg, baseOffset + (chunkBase + k) * VL_FP32);
                        ShiftLefts((RegTensor<uint32_t>&)x, (RegTensor<uint32_t>&)x, static_cast<int16_t>(0), preg);
                        Mul(xSquare, x, x, pregFull);
                        Add(blkSum, blkSum, x, pregFull);
                        Add(blkSq, blkSq, xSquare, pregFull);
                    }
                    chunkBase = static_cast<uint16_t>(chunkBase + period);
                    Sub(y, blkSum, cSum, pregFull);
                    Add(t, accSum, y, pregFull);
                    Sub(cSum, t, accSum, pregFull);
                    Sub(cSum, cSum, y, pregFull);
                    SanitizeComp(cSum, zeroReg, pregFull);
                    Adds(accSum, t, static_cast<float>(0.0), pregFull);

                    Sub(y, blkSq, cSq, pregFull);
                    Add(t, accSq, y, pregFull);
                    Sub(cSq, t, accSq, pregFull);
                    Sub(cSq, cSq, y, pregFull);
                    SanitizeComp(cSq, zeroReg, pregFull);
                    Adds(accSq, t, static_cast<float>(0.0), pregFull);
                }
            }

            ReduceSum(vSum, accSum, pregFull);
            ReduceSum(vSquare, accSq, pregFull);
            // 补偿量先横向归约再扣除，口径与 FinalizeChannel 完全一致（理由见那里）。
            // y / t 在循环结束后已无用，直接复用以免增加寄存器压力。
            ReduceSum(y, cSum, pregFull);
            ReduceSum(t, cSq, pregFull);
            Sub(vSum, vSum, y, pregFull);
            Sub(vSquare, vSquare, t, pregFull);
            StoreOneFp32(sumUb, vSum, pregOne, slot);
            StoreOneFp32(squareSumUb, vSquare, pregOne, slot);
        }
        inQueueX_.FreeTensor(xLocal);
    }

    // ---------- 多累加槽 + Kahan 补偿：寻址、轮转、归并 ----------
    // accBuf_ 布局（VL_FP32 为单位）——累加器区在前、补偿区在后：
    //   [0]                     Σx  槽 0      [1]                     Σx² 槽 0
    //   [2]                     Σx  槽 1      [3]                     Σx² 槽 1
    //   ...
    //   [2*S + 0]               Σx  补偿 0    [2*S + 1]               Σx² 补偿 0
    //   ...
    //   [4*S]                   保护槽（C0 折叠整宽载入的越界余量，见 FinalizeChannelC0）
    //
    // 之所以分两段而不是每槽 4 个 VL 交错：这样槽 0 的 Σx / Σx² 仍落在 [0] 与 [1]，
    // FinalizeChannel / FinalizeChannelC0 一行都不用改。
    __aicore__ inline uint32_t AccBufSlotNum() const
    {
        return ACC_SLOT_NUM * numAccSlots_ * NUM_ACC_AND_COMP + ACC_GUARD_SLOT_NUM;
    }

    __aicore__ inline __local_mem__ float* AccSlot(__local_mem__ float* accUb, uint32_t slot) const
    {
        return accUb + static_cast<uint64_t>(slot) * ACC_SLOT_NUM * VL_FP32;
    }

    // 槽 slot 的补偿量：跳过整个累加器区（2 * numAccSlots_ 个 VL）再按槽定位。
    __aicore__ inline __local_mem__ float* CompSlot(__local_mem__ float* accUb, uint32_t slot) const
    {
        return accUb + (static_cast<uint64_t>(numAccSlots_) + slot) * ACC_SLOT_NUM * VL_FP32;
    }

    // numAccSlots_ 恒为 2 的幂（Host 保证），故取模退化成一次与运算。
    __aicore__ inline uint32_t NextSlot(uint32_t slot) const { return (slot + 1U) & (numAccSlots_ - 1U); }

    __aicore__ inline void ResetAcc(__local_mem__ float* accUb)
    {
        const uint16_t slotNum = static_cast<uint16_t>(AccBufSlotNum());
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            Duplicate(zero, static_cast<float>(0.0), pregFull);
            // 每个累加槽的 Σx / Σx² 两段都要清；最后一段是保护槽——C0 折叠会从
            // accUb + VL_FP32 + k * numC0 处整宽载入，最远读到 2 * VL_FP32 - numC0
            // 之后的位置。那些高 lane 在 Add 时被 pregC0 掩掉、不参与结果，但
            //"读未初始化 UB"本身会被内存检查工具判为非法读，故一并清零。
            // 归纳变量必须是 uint16_t（VF 约束），槽数远小于 uint16 上限。
            for (uint16_t s = 0; s < slotNum; ++s) {
                DataCopy<float, StoreDist::DIST_NORM>(accUb + static_cast<uint64_t>(s) * VL_FP32, zero, pregFull);
            }
        }
    }

    // 把 numAccSlots_ 个槽两两折叠归并回槽 0：槽 g += 槽 g + half，half 每趟减半。
    // 趟数由 Host 算好（foldPasses_ = log2(numAccSlots_)）传进来，__VEC_SCOPE__ 内
    // 没有任何数据依赖分支；half 逐趟减半是纯标量运算，同族 BinaryAddVF 同样写法。
    // 单槽时 foldPasses_ == 0，整个 scope 空转，行为与改动前逐位一致。
    __aicore__ inline void FoldAccSlots(__local_mem__ float* accUb)
    {
        if (numAccSlots_ <= 1) {
            return;
        }
        const uint16_t passes = foldPasses_;
        uint16_t half = static_cast<uint16_t>(numAccSlots_);
        // 补偿区相对累加器区的固定偏移（槽数不随折叠变化，故 base 是常量）。
        const uint64_t compBase = static_cast<uint64_t>(numAccSlots_) * ACC_SLOT_NUM * VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> lhsSum;
            RegTensor<float> rhsSum;
            RegTensor<float> lhsSq;
            RegTensor<float> rhsSq;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();

            for (uint16_t p = 0; p < passes; ++p) {
                half = static_cast<uint16_t>(half / 2U);
                for (uint16_t g = 0; g < half; ++g) {
                    const uint64_t lo = static_cast<uint64_t>(g) * ACC_SLOT_NUM * VL_FP32;
                    const uint64_t hi = static_cast<uint64_t>(g + half) * ACC_SLOT_NUM * VL_FP32;
                    DataCopy<float, LoadDist::DIST_NORM>(lhsSum, accUb + lo);
                    DataCopy<float, LoadDist::DIST_NORM>(rhsSum, accUb + hi);
                    Add(lhsSum, lhsSum, rhsSum, pregFull);
                    DataCopy<float, StoreDist::DIST_NORM>(accUb + lo, lhsSum, pregFull);

                    DataCopy<float, LoadDist::DIST_NORM>(lhsSq, accUb + lo + VL_FP32);
                    DataCopy<float, LoadDist::DIST_NORM>(rhsSq, accUb + hi + VL_FP32);
                    Add(lhsSq, lhsSq, rhsSq, pregFull);
                    DataCopy<float, StoreDist::DIST_NORM>(accUb + lo + VL_FP32, lhsSq, pregFull);

                    // ★补偿区同步折叠★ 收尾要用 Σacc - Σc，各槽的补偿量必须一并归并到槽 0，
                    // 否则只有槽 0 的补偿被计入，其余 S-1 个槽的丢失位直接作废。
                    // 补偿量彼此量级相近（都是各自累加器的末位丢失），朴素相加即可，
                    // 不必再套一层 Kahan。
                    DataCopy<float, LoadDist::DIST_NORM>(lhsSum, accUb + compBase + lo);
                    DataCopy<float, LoadDist::DIST_NORM>(rhsSum, accUb + compBase + hi);
                    Add(lhsSum, lhsSum, rhsSum, pregFull);
                    DataCopy<float, StoreDist::DIST_NORM>(accUb + compBase + lo, lhsSum, pregFull);

                    DataCopy<float, LoadDist::DIST_NORM>(lhsSq, accUb + compBase + lo + VL_FP32);
                    DataCopy<float, LoadDist::DIST_NORM>(rhsSq, accUb + compBase + hi + VL_FP32);
                    Add(lhsSq, lhsSq, rhsSq, pregFull);
                    DataCopy<float, StoreDist::DIST_NORM>(accUb + compBase + lo + VL_FP32, lhsSq, pregFull);
                }
                // 本趟的 store 必须先于下一趟的 load 生效。
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
        }
    }

    // 整个通道累加完毕后才做的唯一一次水平归约。
    //
    // ★补偿量必须先横向归约再扣除，不能逐 lane 扣★
    // Kahan 结束时 acc 里少掉的部分记在 c 里，更准的和是 acc - c。但逐 lane 做
    // acc_i - c_i 是错的：c_i 的量级恰好就是 ulp(acc_i)，fp32 下这个减法只能把结果
    // 挪 0 或 1 个格点，c_i 的大部分信息在舍入里没了（实测见 02_design.md §4.8.4，
    // 那次"逐 lane 扣"的尝试全量通过率 91.50% -> 86.00%，被回退）。
    // 先把 64 个 c_i 横向归约：它们彼此量级相近，求和几乎无损，得到的总补偿 C 是
    // 最终结果尺度上的有效修正量（~1 ulp(S)），再做一次 S - C 即可。
    __aicore__ inline void FinalizeChannel(__local_mem__ float* accUb, __local_mem__ float* sumUb,
                                           __local_mem__ float* squareSumUb, uint32_t slot)
    {
        // 补偿区槽 0（FoldAccSlots 已把各槽补偿归并到此）。
        __local_mem__ float* compUb = CompSlot(accUb, 0);
        __VEC_SCOPE__
        {
            RegTensor<float> accSum;
            RegTensor<float> accSq;
            RegTensor<float> vSum;
            RegTensor<float> vSquare;
            RegTensor<float> vCompSum;
            RegTensor<float> vCompSq;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

            DataCopy<float, LoadDist::DIST_NORM>(accSum, accUb);
            DataCopy<float, LoadDist::DIST_NORM>(accSq, accUb + VL_FP32);
            ReduceSum(vSum, accSum, pregFull);
            ReduceSum(vSquare, accSq, pregFull);

            DataCopy<float, LoadDist::DIST_NORM>(accSum, compUb);
            DataCopy<float, LoadDist::DIST_NORM>(accSq, compUb + VL_FP32);
            ReduceSum(vCompSum, accSum, pregFull);
            ReduceSum(vCompSq, accSq, pregFull);

            // 非有限值下 SanitizeComp 已把 c 置 0，此处减 0，inf/nan 传播契约不变。
            Sub(vSum, vSum, vCompSum, pregFull);
            Sub(vSquare, vSquare, vCompSq, pregFull);

            StoreOneFp32(sumUb, vSum, pregOne, slot);
            StoreOneFp32(squareSumUb, vSquare, pregOne, slot);
        }
    }

    // C0 打包收尾：累加器 lane L 对应 c0 = L % numC0，故按 numC0 折叠而非水平全归约。
    // out[c0] = Σ_{k} acc[k * numC0 + c0]，k ∈ [0, VL_FP32 / numC0)。
    __aicore__ inline void FinalizeChannelC0(__local_mem__ float* accUb, __local_mem__ float* sumUb,
                                             __local_mem__ float* squareSumUb, uint32_t slot)
    {
        const uint32_t c0 = outPerChannel_;
        // __VEC_SCOPE__ 内的循环归纳变量必须是 uint16_t，folds 因此也取 uint16_t。
        const uint16_t folds = static_cast<uint16_t>(VL_FP32 / c0); // numC0 整除 VL_FP32 由 Host 保证
        const uint32_t dstOffset = slot * c0;
        // 补偿区槽 0（FoldAccSlots 已把各槽补偿归并到此）。
        __local_mem__ float* compUb = CompSlot(accUb, 0);
        __VEC_SCOPE__
        {
            RegTensor<float> accSum;
            RegTensor<float> accSq;
            RegTensor<float> partSum;
            RegTensor<float> partSq;
            RegTensor<float> compSum;
            RegTensor<float> compSq;
            // 取前 c0 个 lane 的掩码。UpdateMask 按引用消耗 width，此处只取一次。
            uint32_t width = c0;
            MaskReg pregC0 = UpdateMask<float>(width);

            DataCopy<float, LoadDist::DIST_NORM>(accSum, accUb);
            DataCopy<float, LoadDist::DIST_NORM>(accSq, accUb + VL_FP32);
            // 每次全宽载入偏移 k * c0 处的数据，只把前 c0 个 lane 累进结果；
            // 掩码外的 lane 被 ZEROING 清零，不影响结果（后续只写回前 c0 个 lane）。
            for (uint16_t k = 1; k < folds; ++k) {
                const uint32_t foldOffset = static_cast<uint32_t>(k) * c0;
                DataCopy<float, LoadDist::DIST_NORM>(partSum, accUb + foldOffset);
                Add(accSum, accSum, partSum, pregC0);
                DataCopy<float, LoadDist::DIST_NORM>(partSq, accUb + VL_FP32 + foldOffset);
                Add(accSq, accSq, partSq, pregC0);
            }
            // 补偿量按同一套 C0 折叠（lane L 对应 c0 = L % numC0 的映射对 acc 与 c 一致），
            // 折完再逐 c0 位置扣除。口径与 FinalizeChannel 相同：先归约、后扣除。
            DataCopy<float, LoadDist::DIST_NORM>(compSum, compUb);
            DataCopy<float, LoadDist::DIST_NORM>(compSq, compUb + VL_FP32);
            for (uint16_t k = 1; k < folds; ++k) {
                const uint32_t foldOffset = static_cast<uint32_t>(k) * c0;
                DataCopy<float, LoadDist::DIST_NORM>(partSum, compUb + foldOffset);
                Add(compSum, compSum, partSum, pregC0);
                DataCopy<float, LoadDist::DIST_NORM>(partSq, compUb + VL_FP32 + foldOffset);
                Add(compSq, compSq, partSq, pregC0);
            }
            Sub(accSum, accSum, compSum, pregC0);
            Sub(accSq, accSq, compSq, pregC0);
            DataCopy<float, StoreDist::DIST_NORM>(sumUb + dstOffset, accSum, pregC0);
            DataCopy<float, StoreDist::DIST_NORM>(squareSumUb + dstOffset, accSq, pregC0);
        }
    }

    __aicore__ inline void CopyOutSumSquareSum(uint64_t cBase, uint32_t cnt)
    {
        LocalTensor<T_SUM> sumLocal = outQueueSum_.DeQue<T_SUM>();
        LocalTensor<T_SUM> squareSumLocal = outQueueSquareSum_.DeQue<T_SUM>();
        // C0 打包时每通道占 outPerChannel_ 个 fp32，且 GM 上 c1 连着 c0 排布
        // （[1,1,C1,1,1,C0] 的线性下标 = c1 * C0 + c0），故偏移与长度同比放大即可保持连续。
        const uint64_t gmOffset = cBase * outPerChannel_;
        const uint32_t elemCnt = cnt * outPerChannel_;
        DataCopyExtParams copyParams{
            1,                                              // blockCount
            static_cast<uint32_t>(elemCnt * sizeof(T_SUM)), // blockLen
            0,                                              // srcStride
            0,                                              // dstStride
            0                                               // rsv
        };
        // 两个输出地址完全独立，先后写回互不覆盖。
        DataCopyPad(sumGm_[gmOffset], sumLocal, copyParams);
        DataCopyPad(squareSumGm_[gmOffset], squareSumLocal, copyParams);
        outQueueSum_.FreeTensor(sumLocal);
        outQueueSquareSum_.FreeTensor(squareSumLocal);
    }

private:
    TPipe pipe_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_SUM> sumGm_;
    GlobalTensor<T_SUM> squareSumGm_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueSum_;
    TQue<QuePosition::VECOUT, 1> outQueueSquareSum_;
    TBuf<TPosition::VECCALC> accBuf_;

    int64_t blockIdx_{0};
    uint64_t numN_{0};
    uint64_t numC_{0};
    uint64_t numR0_{0};
    uint64_t r0Align_{0};
    uint64_t usedCoreNum_{0};
    uint64_t cPerCore_{0};
    uint64_t cRound_{0};
    uint32_t nTile_{0};
    uint64_t isSubR_{0};
    uint32_t r0Factor_{0};
    uint32_t numChunks_{0};
    uint32_t tailLen_{0};
    uint32_t outPerChannel_{1};
    bool fusedSingleTile_{false}; // 整通道一次 tile 搬完 → 可走融合快路
    uint32_t numAccSlots_{1};     // 累加槽数，恒为 2 的幂；1 = 退化为单槽（原行为）
    uint16_t foldPasses_{0};      // 归并趟数 = log2(numAccSlots_)
};
} // namespace BN3DTrainingReduceOps
#endif // BN3D_TRAINING_REDUCE_DENSE_CHANNEL_H_
