/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_kernel/arch35/lp_norm_reduce_empty.h
// =============================================================================
//
// ROLE: LpNormReduce Empty 模板（tilingKey=2）kernel 类实现。
//
//   据 docs/LpNormReduce/design/Kernel.md §9.7（空 tensor kernel 规格 +
//   跨分支共享 DuplicateEmptyROutputVf 完整源码）与
//   docs/LpNormReduce/design/branches/DESIGN-BRANCH-2.md §3–§5（流水线 /
//   内存规划 / Kernel 计算链）落码，范式来源 reduction 空 tensor 模板参考实现
//   euclidean_norm_empty.h（占位替换：sqrt(空和)=0 → 空和常量
//   EMPTY_R_OUTPUT_VALUE=0；p=±inf 哨兵 + 空 R 轴已在 host 前置拒绝，Empty
//   TilingData 无 pOrder——单一编译期常量覆盖全部放行分支）。
//
//   两态数据流（DESIGN-BRANCH-2.md §3）：
//   - EMPTY_A（usedCoreNum=0、SetBlockDim(1)）：Init 直接返回（不绑 GM、
//     不分配 buffer），Process 入口 blockIdx >= usedCoreNum 全核早退——
//     零计算、零 IO、零同步（Kernel.md §9.7「EMPTY_A 数据流」；该判定亦
//     覆盖 EMPTY_R 的多余核，为第二道防线）；
//   - EMPTY_R（usedCoreNum>0，按 aTotal 切核）每核两段式：
//       Compute（V 队列）：Duplicate 一次 postBuf ← EMPTY_R_OUTPUT_VALUE=0
//         （标量常数与 aOff 无关，整个 kernel 只执行一次、CopyOut 循环复用）
//       → SetFlag/WaitFlag V_MTE3（§5.5 唯一同步对——postBuf 被 V 队列
//         Duplicate 写、被 MTE3 队列 DataCopyPad 读的 RAW 依赖）
//       → CopyOut（MTE3 队列）循环：for aOff ∈ [aStart, aEnd) 步进 aUbFactor，
//         每轮 DataCopyPad 搬 aLen = min(aUbFactor, aEnd−aOff) 个 0 到 gmY
//         （尾块 valid 现算，无 padding 写出）。
//     无 CopyIn（empty 不做 reduce，只 Duplicate 固化值、不搬入输入数据，
//     §5.2）、无 MTE2 相关同步、无跨核 SyncAll（各核输出区间互不重叠）。
//
//   常量 0 直写 D_T 域（has_post_elewise=false：0.0f→fp16=0x0000 与
//   「fp32 常量 + 缩位 Cast」位级等价，Kernel.md §9.7 结论）——无 fp32 中间量、
//   无 Cast、无 PostElewise VF 链。
//
//   UB 内存规划（§4）：仅 1 个 TBuf——postBuf（VECCALC 位置、D_T、深度 1、
//   不开 double buffer；postBufSize = CeilAlign(max(aUbFactor×maxDtypeSize,
//   blockSize), blockSize)，封顶 64KB）；无 TQue 队列对（无 CopyIn）。
//
// =============================================================================

#ifndef LP_NORM_REDUCE_EMPTY_H_
#define LP_NORM_REDUCE_EMPTY_H_

#include "kernel_operator.h"            // Ascend C core framework (AscendC:: namespace)
#include "lp_norm_reduce_tiling_data.h" // LpNormReduceEmptyTilingData（empty 专用独立 struct）

namespace NsLpNormReduce {

using namespace AscendC;

// =============================================================================
// §9.7 DuplicateEmptyROutputVf —— EMPTY_R 输出固化值填充 VF
// （Kernel.md §9.7 完整源码，跨分支共享；对应 §5 逐步算子链唯一一条 VF 步 C1：
//  常量 0 → postBuf（D_T），即「数学公式」空集合 Σ|x|^p = 0）
//
// 寄存器 vlElems 按元素数分 repeat（fp32=64 / b16=128）、UpdateMask 精确覆盖
// count 个元素（引用语义自动递减 remaining），非对齐 aUbFactor 无需额外尾处理；
// postBuf 中 [aLen, aUbFactor) 区间的多余 0 不被 CopyOut 读出（blockLen 按
// valid aLen 传，§5.4），无越界写 GM 风险。
// =============================================================================
template <typename T>
__simd_vf__ inline void DuplicateEmptyROutputVfImpl(__ubuf__ T* dst, T value, uint32_t count, uint16_t repU16)
{
    constexpr uint32_t vlElems = 256 / sizeof(T); // 寄存器 vlElems（元素数）：fp32=64，b16=128
    AscendC::Reg::RegTensor<T> vReg;
    AscendC::Reg::Duplicate(vReg, value);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repU16; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(vlElems);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::StoreAlign(dst + off, vReg, mask);
    }
}

// 调用侧 wrapper（Kernel.md §9.7）：repU16 由 count 现算后经 asc_vf_call 传参
template <typename T>
__aicore__ inline void DuplicateEmptyROutputVf(__ubuf__ T* dst, T value, uint32_t count)
{
    constexpr uint32_t vlElems = 256 / sizeof(T); // 与 impl 一致：fp32=64，b16=128
    uint16_t repU16 = static_cast<uint16_t>((count + vlElems - 1) / vlElems);
    asc_vf_call<DuplicateEmptyROutputVfImpl<T>>(dst, value, count, repU16);
}

// ===========================================================================
// LpNormReduceEmptyKernel<DType> —— Empty 模板 kernel 类（tilingKey=2）
//
// EMPTY_A / EMPTY_R 共用同一 binary（DTYPE_X 编译期实例化 fp16 / fp32 / bf16 三档），
// kernel 内按 usedCoreNum 区分（§0）。不继承 / 不复用 Base / Group（Empty
// 独立 TilingData struct，无 CopyIn / PreElewise / Reduce / PostElewise 链，
// Kernel.md §1 入口三形态之 op.Init(y, &tilingData, &pipe)——不绑输入 x）。
// ===========================================================================
template <typename DType>
class LpNormReduceEmptyKernel {
public:
    using DT = DType;

    __aicore__ inline LpNormReduceEmptyKernel() {}

    // Empty 初始化（§5.1）：EMPTY_A 短路（usedCoreNum==0 直接返回——不绑 GM、
    // 不分配 buffer，零计算零 IO 零同步）；EMPTY_R 绑定输出 y（不绑输入 x——
    // 空 tensor 不读输入）、按 postBufSize 分配唯一 postBuf。
    __aicore__ inline void Init(GM_ADDR y, const LpNormReduceEmptyTilingData* td, TPipe* pipe);

    // Empty 主流程（§5.3）：blockIdx >= usedCoreNum 早退（EMPTY_A 全核 +
    // EMPTY_R 多余核第二道防线）→ blockIdx → [aStart, aEnd) 大小核区间映射
    // → Duplicate 一次 → V_MTE3 事件对（§5.5 唯一同步）→ CopyOut 循环。
    __aicore__ inline void Process();

private:
    // EMPTY_R 写回（§5.4）：DataCopyPad 单路径 dense（UB→GM）——blockLen 按
    // valid aLen 计（不要求 32B 对齐）、srcStride=0（UB gap=0）/ dstStride=0
    // （GM dense）；UB→GM 方向不需要 DataCopyPadExtParams（不支持也不需
    // padParams）。EMPTY_A 无 CopyOut（早退于任何搬运之前）。
    __aicore__ inline void CopyOut(int64_t aOff, int64_t aLen);

    const LpNormReduceEmptyTilingData* td_ = nullptr;
    TPipe* pipe_ = nullptr;
    GlobalTensor<DT> gmY_;            // 输出 y（不绑 x——empty 不读输入，§5.1）
    TBuf<TPosition::VECCALC> outBuf_; // postBuf（§4 唯一 buffer，VECCALC、深度 1）
    // 空和常量（Kernel.md §9.7 取值表：sum 族空和 = 0；p=±inf 哨兵 + 空 R 轴
    // 已被 host 前置拒绝，Empty TilingData 无 pOrder——单一编译期常量覆盖
    // 全部放行分支 p=0/1/≥2）
    static constexpr DT EMPTY_R_OUTPUT_VALUE = static_cast<DT>(0);
    // 事件 ID 分配表（§5.5：本分支仅用 1 个）——V_MTE3：postBuf 被 V 队列
    // Duplicate 写、被 MTE3 队列 DataCopyPad 读的唯一同步对
    static constexpr uint8_t EVENT_ID = 0;
};

// ---------------------------------------------------------------------------
// Init（DESIGN-BRANCH-2.md §5.1）
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceEmptyKernel<DType>::Init(GM_ADDR y, const LpNormReduceEmptyTilingData* td,
                                                            TPipe* pipe)
{
    td_ = td;
    pipe_ = pipe;
    if (td_->usedCoreNum == 0) {
        return; // ★ EMPTY_A：全核早退（零计算、零 IO、零 buffer）
    }
    // GM 绑定：仅输出 y（empty 不读 x）
    gmY_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(y));
    // TBuf 分配（§4 Buffer 划分表：唯一 postBuf，VECCALC 位置、深度 1）
    pipe_->InitBuffer(outBuf_, static_cast<uint32_t>(td_->postBufSize));
    // 无 DMA / pattern 参数预计算：本分支无 CopyIn、无合轴 pattern、无二分树派生量
}

// ---------------------------------------------------------------------------
// Process（DESIGN-BRANCH-2.md §5.3 / Kernel.md §9.7 骨架 + §5.5 同步对）
//
// 持有法则 trace（§5.3 注）：执行前持有=[]；Duplicate 后持有=[postBuf]（P=1
// 峰值，b16/fp32 同为 1，无 dtype 分档）；postBuf 写一次、CopyOut 循环读多次。
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceEmptyKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
        return; // ★ EMPTY_A：usedCoreNum=0 → 所有核早退；
                //   亦覆盖 EMPTY_R 的 blockIdx ≥ usedCoreNum 多余核
    }

    // ── blockIdx → [aStart, aEnd) 按元素数切（§2 大小核协议，非 aLoop 多维解码）──
    int64_t aStart = 0;
    int64_t aEnd = 0;
    if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
        aStart = blockIdx * td_->aBigCoreLoopCnt * td_->aUbFactor;
        aEnd = aStart + td_->aBigCoreLoopCnt * td_->aUbFactor;
    } else {
        aStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt * td_->aUbFactor +
                 (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt * td_->aUbFactor;
        aEnd = aStart + td_->aSmallCoreLoopCnt * td_->aUbFactor;
    }
    aEnd = (aEnd > td_->aTotal) ? td_->aTotal : aEnd; // 防越界
    if (aStart >= aEnd) {
        return;
    }

    // ── Compute（V 队列）：Duplicate 一次（标量常数与 aOff 无关；
    //    has_post_elewise=false → 常量直写 D_T 域，无 fp32 中间量、无 Cast）──
    __ubuf__ DT* outPtr = reinterpret_cast<__ubuf__ DT*>(outBuf_.Get<DT>().GetPhyAddr());
    DuplicateEmptyROutputVf<DT>(outPtr, EMPTY_R_OUTPUT_VALUE, static_cast<uint32_t>(td_->aUbFactor));

    SetFlag<HardEvent::V_MTE3>(EVENT_ID);  // §5.5：postBuf V→MTE3 RAW 依赖（唯一同步对）
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID); // MTE3 队列等 Duplicate 完成；MTE3 同队列保序，
                                           // 后续 CopyOut 不再重复等待，无尾部同步必要

    // ── CopyOut（MTE3 队列）循环：仅搬出（无 CopyIn——empty 不搬输入数据）──
    for (int64_t aOff = aStart; aOff < aEnd; aOff += td_->aUbFactor) {
        const int64_t aLen = (td_->aUbFactor < aEnd - aOff) ? td_->aUbFactor : (aEnd - aOff);
        CopyOut(aOff, aLen);
    }
}

// ---------------------------------------------------------------------------
// CopyOut（DESIGN-BRANCH-2.md §5.4）—— DataCopyPad 单路径 dense（UB→GM）
//
// 路径决策表（本分支仅 1 行）：EMPTY_R（唯一路径）| blockLen = aLen×sizeof(D_T)
// （valid，尾块 aEnd−aOff 兜底）| blockCount=1 | srcStride/dstStride = 0/0。
// 非 inplace（outputs.y.aliasing: none）直接写 gmY[aOff]，输出偏移即 GM 上
// A 轴 dense 元素偏移（无 stride 解码）；keepdim 的 shape 差异由调用方按
// InferShape 分配的 y 表达，kernel 无感知。
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceEmptyKernel<DType>::CopyOut(int64_t aOff, int64_t aLen)
{
    auto outDeq = outBuf_.Get<DT>();
    DataCopyExtParams outParams;
    outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT))); // valid 字节
    outParams.blockCount = 1;
    outParams.srcStride = 0; // UB 侧 gap=0
    outParams.dstStride = 0; // GM dense 写出
    outParams.rsv = 0;       // 必须显式填 0（datacopypad-rules）
    DataCopyPad(gmY_[aOff], outDeq, outParams);
}

} // namespace NsLpNormReduce

#endif // LP_NORM_REDUCE_EMPTY_H_
