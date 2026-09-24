/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GruBlockCell — Ascend 950PR (dav_3510) MIX kernel（1 AIC : 2 AIV，fp32-only）。
// reset_before 变体：
//   ru_bar = [x, h_prev]·wRu + bRu;  r = σ(ru_bar[:,0:H]);  u = σ(ru_bar[:,H:2H])
//   c = tanh([x, h_prev⊙r]·wC + bC);  h = c + u·(h_prev − c)
//
// 循环嵌套（DATAFLOW NOTES #7 的核心）：**列片 s 最外、split-K 组 g 最内**。
//   AIC: for s { for g { [A 现搬] r 门 Mmad→Fixpipe→C2V ; u 门 Mmad→Fixpipe→C2V } }
//        → CubeWaitVec（全部列片的 h⊙r 已落 GM）
//        → for s { for g { [A 现搬] c 门 Mmad→Fixpipe→C2V } }
//   AIV: for s { for g { VecWaitCube→σ 前的 Neumaier 归并→V2C }
//                → finalize→σ(r/u)→CopyOut r/u→hr=hp⊙r(VF)→CopyOut hr }
//        → for s { for g { VecWaitCube→归并→V2C }
//                → finalize→tanh(c)→CopyOut c→h=c+u·(hp−c)(VF)→CopyOut h }
//
// 为什么列片必须外提（性能契约，勿回退）：
//   UB 的 8 个平面若按**全幅 Hp** 分配，则 rowsMax=⌈mChunk/2⌉ 被 8·Hp·4B 钉死
//   （H=6144 → rowsMax=1 → mChunk=2），而权重 wRu+wC 每个 m-chunk 都要从 HBM 全量
//   重灌一遍 ⇒ 流量 = ⌈B/mChunk⌉·3(I+H)H·4。实测 B=16384/I=620/H=6144 时
//   mChunk=2 → 4.09TB → 4.1s（有效带宽 ~0.95TB/s，纯 HBM 墙）。列片外提后平面宽
//   降为 nL0c，mChunk 只受 UB×L0C 联合预算约束（与 Hp 无关），host 侧按流量模型
//   择优 ⇒ 权重重灌次数从 B/2 降到 B/mChunk。
//
// 关键约束（由 I/H 任意 ≥1、非 8/16 对齐的 shape 契约强制）：
// 1. N 向 pad 到 Hp=CeilAlign(H,8)：Fixpipe(isToUB) 要求 drain 宽度 32B 对齐。
//    pad 列承载尾块垃圾（Mmad k 按 H 精确截断，不贡献），CopyOut 按每行 tileCols
//    截断。列片宽 w=min(nL0c, Hp−s) 恒 8 对齐（nL0c 16 对齐、Hp 8 对齐、s 16 对齐）
//    ⇒ UB 平面行距取 w 即紧凑，AIV 侧全部逐元素算子可按扁平 work=rows·w 处理。
// 2. r/u 拆两次独立 N=w 的 GEMM（H%8≠0 时合并门的 GatherPlane blockLen 不可表达）。
// 3. K 按 I/H 边界分段、再按组宽 kg 分组；pass0 两门与 pass1 的 x 段，A 切片
//    [mChunk, kg] 每 (列片, 组) 从 GM 现搬进 aK 槽（x / h_prev 两源共用），足迹与
//    I、H 均无关。
// 4. h⊙r 仍走 **L1 全幅回灌槽 aH [mChunk, Hp] + UB→L1 FeedbackToL1**（不过 GM，
//    本算子无 workspace）。pass1 的 h 段 Mmad 需要整幅 A，故 aH 不可 K 分片——它是
//    mChunk 的上界（CeilAlign(mChunk,16)×Hp×4 须与 aK/b/bias 共存于 512KB L1，
//    H=6144 → mChunk ≤ 16），也是支持域 Hp ≤ 8152 的来源。
//    与列片外提共存的关键：**pass0 的 h 段不读 aH**（改按 K 组从 GM 现搬 hPrev 进
//    aK），故 pass0 各列片把 h⊙r 回灌进 aH[:, s] 与 pass0 自身的 A 读互不冲突；
//    pass1 由块级 CubeWaitVec（V2C 为 FIFO barrier，消费末列片末组即蕴含全部前序
//    回灌已完成）保证 aH 全幅写满后才开始读。
//    （曾评估过 h⊙r 改走 GM workspace 以彻底解除 mChunk 的 L1 上界：GE 侧确实按
//    tiling 申请额分配并下发地址，但 kernel 任一侧访问都触发 EZ9999 errcode 95
//    "DDR address of the MTE instruction is out of range"，且阈值与 shape 无关地
//    落在 16~64MB 量级（B=4/H=8 实需 128B 也要 ≥17MB，B=5120/H=1024 给 24MB 仍炸、
//    64MB 才过），无法稳健工程化，故维持片上回灌。）
// 5. AIV 工作集 8 个 [rowsMax, nL0c] 列片平面；别名写读先序见 Layout UB 段——
//    改别名必须重核。
// 6. M 向按 mChunk 分块，每块独立走完 pass0（全列片）→ pass1（全列片）。
// 7. 列片 ↔ GM 的搬运逐行下发（CopyTileToGm/CopyTileFromGm，blockCount=1）：
//    DataCopyPad 在 stride=0 时 GM 侧行距取 blockLen 本身，而列片的
//    blockLen=tileCols×4 < H×4，整块搬会让第 1 行起落到错误行偏移（实测多列片
//    shape 只有每个 AIV stripe 的第 0 行正确）；改用 dstStride 表达 H×4 也不通用
//    ——它以 32B 为单位，需要 (H−tileCols)%8==0，H 非 8 对齐时（H=4097 末列片
//    tileCols=161）不成立。逐行搬对任意 H 成立，代价是每列片 stripe.count 条指令。
//
// tiling 决策全在 host（ComputeLayoutDecision）；本文件 Layout 只消费 TilingData +
// 纯算术推导 + Bump 偏移。
//
// VF 主链：GruBlockCellResetMulVF(hr=hp⊙r)、GruBlockCellBlendVF(h=c+u·(hp−c))。
// σ/tanh 走 HL 链（regbase VF 无 Sigmoid/Tanh）；tanh 为分段补偿实现（INTRINSIC
// 近零区 1.67e7 ULP）；饱和 clamp 用 Select 而非 Mins——Mins 的 minNum 语义吞 NaN，
// 破坏 NaN 逐元素传播契约。
//
// cube 通路全走 AscendC::Te 原子层；唯一裸路 gbc::vector::FeedbackToL1（Te ub_to_l1
// 无 ND→NZ 路由，底层 DataCopyUB2L1ND2NZImpl 存在但未暴露）。
//
// 文件组织（单文件交付，按数据通路分节；节间只有向下的依赖）：
//   §1 gbc::layout  布局算术与片上分配器（常量 / CeilDiv·CeilAlign·MinU·MaxU /
//                   Bump / NzLayout / RowStripe·DrainStripe / FinFinalize）
//   §2 gbc::sync    同步事件族（核内定向 hazard + 跨核 C2V/V2C）
//   §3 gbc::cube    AIC 侧数据通路（GM→L1→L0A/L0B/BT→Mmad→L0C→UB，全走 Te 原子层）
//   §4 gbc::vector  AIV 侧激活与回灌（σ/tanh / UB→L1 FeedbackToL1）+ VF 主链
//   §5 GruBlockCellKernel  算子主类（Layout 片上偏移 + CubeChunk/VectorChunk 流水）
// 另两个头文件不可并入本文件：gru_block_cell_tiling_struct.h 是 host—kernel 共享
// POD（op_host 侧 tiling 直接 include），gru_block_cell_struct.h 是 ASCENDC_TPL
// 模板参数声明（编译期分发表，须独立成头）。

#ifndef GRU_BLOCK_CELL_KERNEL_H
#define GRU_BLOCK_CELL_KERNEL_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "gru_block_cell_tiling_struct.h" // GruBlockCellTilingData（host—kernel 共享 POD）

// ===========================================================================
// §1 gbc::layout — 布局算术与片上分配器
// ===========================================================================
namespace gbc {
namespace layout {

constexpr uint32_t CUBE_BLOCK = 16; // M/N 侧分形边长（元素，与 dtype 无关）
constexpr uint32_t C0_BYTES = 32;   // ⚠ 字节；fp32 的 C0 = 8 元素（分形 16x8）
constexpr uint32_t C0F = C0_BYTES / sizeof(float);
constexpr uint32_t BT_ALIGN = 64;      // BT 表突发粒度（字节）
constexpr uint32_t BITS_PER_BYTE = 8;  // msk 位图换算（1 bit/元素；host tiling 同名常量同值）
constexpr uint32_t DRAIN_CHUNK = 2048; // L0C→UB drain 列宽分块（fixpipe nSize 实测安全界；
                                       // Te 内部 main_loop_n_size=512 分段在 SPLIT_M 大宽度
                                       // 下静默丢数据——实测 H=4097 主片 4096 列全零交付）

// 片上容量与预算常量在 host 侧（决策唯一真值：tiling_arch35.cpp 的
// ComputeLayoutDecision；常量 GRU_TIL_* 在共享头 gru_block_cell_tiling_struct.h）。
// kernel 不做容量/策略判定，只消费 TilingData。
// 布局算术三件套。与 common/inc/op_kernel/kernel_utils.h 的 ops:: 版语义重复，
// 但该公共头不在本算子 kernel 的 include 路径上（裸 include 会被 CANN 内部同名头
// 抢先解析），故保留本地实现。无 b==0 守卫：全部调用点除数为编译期非零常量或
// host gate 保证的正值。
__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
__aicore__ inline uint32_t MinU(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
__aicore__ inline uint32_t MaxU(uint32_t a, uint32_t b) { return (a > b) ? a : b; }

// ---------------------------------------------------------------------------
// Bump — 单一 on-chip buffer 的运行期布局器（每笔分配对齐 32B；DataCopy 族以
// 32B 块表达长度/步长，不对齐的起始偏移会整块漂移）。
// ---------------------------------------------------------------------------
struct Bump {
    uint32_t cur;
    __aicore__ inline explicit Bump(uint32_t base = 0) : cur(base) {}
    __aicore__ inline uint32_t Take(uint32_t bytes)
    {
        uint32_t o = cur;
        cur += gbc::layout::CeilAlign(bytes, C0_BYTES);
        return o;
    }
    template <typename T>
    __aicore__ inline uint32_t TakeT(uint32_t elems)
    {
        return Take(elems * sizeof(T));
    }
};

// ---------------------------------------------------------------------------
// NzLayout — NZ 分形布局（(m,k) → (k/c0)*rowsAligned*c0 + m*c0 + k%c0）。
// rowsAligned 是**父 tile**的对齐行数（外层步长），与 rows 分立：紧凑 tile 与
// 大 tile 的列块切片同形不同步长（门平面抽取 stride 陷阱同源）。
// ---------------------------------------------------------------------------
struct NzLayout {
    uint32_t rows;
    uint32_t cols;
    uint32_t c0;
    uint32_t rowsAligned;

    __aicore__ inline NzLayout(uint32_t r, uint32_t c, uint32_t c0In, uint32_t parentRows)
        : rows(r), cols(c), c0(c0In), rowsAligned(CeilAlign(parentRows, CUBE_BLOCK))
    {}
    __aicore__ inline uint32_t ColBlock(uint32_t kb) const { return kb * rowsAligned * c0; }
};

// ---------------------------------------------------------------------------
// RowStripe — 一个 AIV 独占的行区间。L1 无写仲裁，双 AIV 回灌安全完全依赖
// 「写回分区 == drain 交付分区」这一构造不变量（与 RowStripe 构造同源）。
// ---------------------------------------------------------------------------
struct RowStripe {
    uint32_t base;
    uint32_t count;
};

// Fixpipe(isToUB) 交付给单 AIV 的行数（ALLOCATION 上界；odd m 下 AIV1 实际
// 有效行数少一，pad 行不携带数据）。
__aicore__ inline uint32_t DrainRowsMax(uint32_t m, bool splitM) { return splitM ? CeilDiv(m, 2) : m; }

// ---------------------------------------------------------------------------
// FinFinalize — Neumaier 累加的收尾：acc += comp，附有限性守卫（2-scratch 形态：
// t1 先承载 |acc| 供比较，消费后复用作 acc+comp——UB 平面数从 14 缩到
// 8 时去掉了独立 t3 平面，A/B 端收尾与激活共用 t1/t2 两 scratch）。
// 部分和含 ±Inf/NaN 时 TwoSum 的补偿项无定义（Inf−Inf=NaN），按
// 「Inf 经激活饱和 / NaN 逐元素传播」契约直接采用主和：
// msk = (|acc| ≤ FLT_MAX)（NaN 比较恒 false）；acc = msk ? acc+comp : acc。
// ---------------------------------------------------------------------------
constexpr float FLT_MAX_LIMIT = 3.402823466385288598e+38F; // FLT_MAX

__aicore__ inline void FinFinalize(const AscendC::LocalTensor<float>& acc, const AscendC::LocalTensor<float>& comp,
                                   const AscendC::LocalTensor<float>& t1, const AscendC::LocalTensor<float>& t2,
                                   const AscendC::LocalTensor<uint8_t>& msk, uint32_t n)
{
    AscendC::Abs(t1, acc, n);
    AscendC::Duplicate(t2, FLT_MAX_LIMIT, n);
    AscendC::Compares(msk, t1, t2, AscendC::CMPMODE::LE, n); // finite 判定（NaN 恒 false）
    AscendC::Add(t1, acc, comp, n);                          // t1 = acc+comp（|acc| 已被消费，scratch 复用）
    AscendC::Select(acc, msk, t1, acc, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
}

// DrainToUB(splitM=true) 交给本 AIV 的 stripe：每 AIV 得 CeilDiv(m,2) 行、
// AIV k 从 k*CeilDiv(m,2) 起（硬件实测 CeilDiv 而非 m/2；m=1 时 AIV1 为 0 行）。
__aicore__ inline RowStripe DrainStripe(uint32_t m, bool splitM, uint32_t subBlockIdx)
{
    RowStripe s;
    if (splitM) {
        const uint32_t half = CeilDiv(m, 2);
        s.base = subBlockIdx * half;
        s.count = (s.base >= m) ? 0 : (MinU(m - s.base, half));
    } else {
        s.base = 0;
        s.count = (subBlockIdx == 0) ? m : 0;
    }
    return s;
}

} // namespace layout
} // namespace gbc

// ===========================================================================
// §2 gbc::sync — 同步事件族（一个握手的两端强制同处相邻）
// ===========================================================================
namespace gbc {
namespace sync {

// 核内：Set+Wait 同一事件 id 的全停（功能参考正确性优先；软件流水化留作优化）。
// ⚠ EVENT_ID0 为 cce_aicore_intrinsics.h 的全局枚举；SetFlag<E>/WaitFlag<E> 形参
// 为 int32_t（kernel_event.h SetFlagImpl(int32_t)）。
template <AscendC::HardEvent E>
__aicore__ inline void PipeWait(int32_t id = EVENT_ID0)
{
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}

__aicore__ inline void WaitMte2ToMte1() { PipeWait<AscendC::HardEvent::MTE2_MTE1>(); } // GM→L1 后
__aicore__ inline void WaitMte1ToM() { PipeWait<AscendC::HardEvent::MTE1_M>(); }       // L1→L0/BT 后
__aicore__ inline void WaitMToFix() { PipeWait<AscendC::HardEvent::M_FIX>(); }         // Mmad 后
__aicore__ inline void WaitFixToM() { PipeWait<AscendC::HardEvent::FIX_M>(); }         // drain 后复用 L0C

// ---- 定向化细分（范式 patterns.md 误区3错误3：热循环屏障换命名 hazard 的
// 定向事件，实测同类改造快 13~17% 且输出逐位相同；只排真正相关的那对流水）----
__aicore__ inline void WaitMToMte1() { PipeWait<AscendC::HardEvent::M_MTE1>(); } // Mmad 读 L0A/L0B/BT 后，MTE1 复用装载
__aicore__ inline void WaitMte1ToMte2()
{
    PipeWait<AscendC::HardEvent::MTE1_MTE2>();
} // SplitB 读 L1 b1 后，MTE2 复写 b1
__aicore__ inline void WaitVToMte3() { PipeWait<AscendC::HardEvent::V_MTE3>(); } // V 读消费完 → MTE3（置位/搬出）
__aicore__ inline void WaitVToMte2() { PipeWait<AscendC::HardEvent::V_MTE2>(); } // V 写消费完 → MTE2 复写
// （MTE3→V / VF 相邻界面不设定向事件：VF 异步完成不被命名事件闭合，见文件头）

// ---- 跨核（AIC ↔ 2 AIV）----
constexpr uint16_t FLAG_C2V = 8;
constexpr uint16_t FLAG_V2C = 9;

// CUBE → VECTOR：drain 落 UB 后由 FIX 管置位；两个 AIV 各自 wait。
__aicore__ inline void CubeSignalVec() { AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_C2V); }
__aicore__ inline void VecWaitCube() { AscendC::CrossCoreWaitFlag(FLAG_C2V); }

// VECTOR → CUBE：⚠ barrier 而非计数信号量——两个 AIV 都要 set，cube 恰好消费一次
// （1:1 与 2:2 配对均死锁，实测）。cube 的 wait 在 M 管，但对回灌数据的后续读在
// MTE2/MTE1，故附带全栅栏。
__aicore__ inline void VecSignalCube() { AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_V2C); }
__aicore__ inline void CubeWaitVec()
{
    AscendC::CrossCoreWaitFlag<0x2, PIPE_M>(FLAG_V2C);
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace sync
} // namespace gbc

// ===========================================================================
// §3 gbc::cube — AIC 侧数据通路（AscendC::Te 原子层）
// ===========================================================================
namespace gbc {
namespace cube {

using gbc::layout::C0F;
using gbc::layout::CeilAlign;
using gbc::layout::CUBE_BLOCK;

// BT 表足迹（元素）：按 64B 突发粒度向上取整（Bump 槽容量/host 同源）。
__aicore__ inline uint32_t BtElems(uint32_t n)
{
    return gbc::layout::CeilAlign(n * sizeof(float), gbc::layout::BT_ALIGN) / sizeof(float);
}

// Mmad 的 m 有下限 2（m=1 实测静默错数；多余输出行落在 L0C pad 行，drain 按真实
// m 交付、不读 pad 行）。
__aicore__ inline uint32_t MmadM(uint32_t m) { return (m < 2) ? 2 : m; }

// ---------------------------------------------------------------------------
// 布局构造捷径
// ---------------------------------------------------------------------------

// 带 pitch 的 ND 布局（Blaze per_token_scale 同款公开形态）：rowPitch=行元素距。
// 跨步 GM 源（权重列切片）与 UB 平面列片落点（行距=全平面宽）共用此构造。
__aicore__ inline auto NdPitchLayout(uint32_t rows, uint32_t cols, uint32_t rowPitch)
{
    auto shape = AscendC::Te::MakeShape(AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows),
                                        AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, cols));
    auto stride = AscendC::Te::MakeStride(AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, rowPitch),
                                          AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
    return AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(shape,
                                                                                                               stride);
}

// fp32 NZ 帧布局（c0=8；rowsAligned=CeilAlign(rows,16) 由布局代数承载——
// (k/c0)*rowsAligned*c0 + m*c0 + k%c0 与 Te NZ 帧逐项一致）
__aicore__ inline auto NzLayoutF(uint32_t rows, uint32_t cols)
{
    return AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(rows, cols);
}

// L0C 帧布局（NZ + c0=16：L0C 原生 16 元素 C0；Blaze bl1_full_load 同款）
__aicore__ inline auto L0cLayoutF(uint32_t rows, uint32_t cols)
{
    return AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<16>>{}(rows, cols);
}

// ---------------------------------------------------------------------------
// Mmad 三态（Te::Mmad；MmadParams{m,n,k,unitFlag,cmatrixInitVal} 与
// 裸 MmadParams 逐字段同名同义——cmatrixInitVal→init_with_zero 直映，bias 形式
// 的 init 硬编码 false 与原「cmatrixInitVal 必须 false——true 丢弃 BT 种子」一致）
// seedMode：0=Bias（BT 种子，段/门首块）/ 1=Plain（覆写从零起，组 g≥1 首块）/
// 2=Accum（续累加，同链后续段）。
// ---------------------------------------------------------------------------
__aicore__ inline void MmadGate(uint32_t mNow, uint32_t sliceN, uint32_t n0, uint32_t ns, uint32_t kcNow,
                                uint32_t seedMode, bool firstChunk)
{
    constexpr auto atom = AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{});
    const uint32_t nAl = gbc::layout::CeilAlign(ns, gbc::layout::CUBE_BLOCK);
    auto cParent = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0u),
                                           L0cLayoutF(mNow, sliceN));
    auto cTile = cParent.Slice(AscendC::Te::MakeCoord(0u, n0), AscendC::Te::MakeShape(mNow, ns));
    auto aTile = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(0u),
                                         NzLayoutF(mNow, kcNow));
    auto bTile = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(0u),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(kcNow, ns));
    if (seedMode == 0 && firstChunk) {
        auto bias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(0u),
                                            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1u, ns));
        AscendC::Te::Mmad(
            atom.with(AscendC::Te::MmadParams{static_cast<uint16_t>(MmadM(mNow)), static_cast<uint16_t>(nAl),
                                              static_cast<uint16_t>(kcNow), 0, false}),
            cTile, aTile, bTile, bias);
    } else if (seedMode == 1 && firstChunk) {
        AscendC::Te::Mmad(
            atom.with(AscendC::Te::MmadParams{static_cast<uint16_t>(MmadM(mNow)), static_cast<uint16_t>(nAl),
                                              static_cast<uint16_t>(kcNow), 0, true}),
            cTile, aTile, bTile);
    } else {
        AscendC::Te::Mmad(
            atom.with(AscendC::Te::MmadParams{static_cast<uint16_t>(MmadM(mNow)), static_cast<uint16_t>(nAl),
                                              static_cast<uint16_t>(kcNow), 0, false}),
            cTile, aTile, bTile);
    }
}

// ---------------------------------------------------------------------------
// 搬运原语（Te 原子；签名从 LocalTensor 视图改为 Layout 槽偏移 + 窗口标量）
// ---------------------------------------------------------------------------

// GM(ND 跨步窗口) → L1 NZ tile（CopyGM2L1 的 nd2nz 路由；行距由 pitch 布局
// 承载——原 srcDValue/gw 防截断防线与 dstNzC0Stride 推导全部内化于布局代数。
// parentRows==rows 于全部调用点成立，NZ 帧布局按自身 rows 推导 rowsAligned）。
__aicore__ inline void CopyInNd2Nz(const AscendC::GlobalTensor<float>& gm, uint32_t srcRow, uint32_t srcCol,
                                   uint32_t rows, uint32_t cols, uint32_t rowPitch, uint32_t l1Off)
{
    auto src = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                           gm.GetPhyAddr(static_cast<uint64_t>(srcRow) * rowPitch + srcCol)),
                                       NdPitchLayout(rows, cols, rowPitch));
    auto dst = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(l1Off),
                                       NzLayoutF(rows, cols));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{}), dst, src);
}

// GM(ND 一行窗口，字节精确) → L1 bias 表（nd2nd 路由；blockLen=elems*4 字节
// 精确——范式不变量 #7 由 Te 的 align_v2 字节块承载）
__aicore__ inline void CopyInBiasPad(const AscendC::GlobalTensor<float>& gm, uint32_t srcCol, uint32_t gmCols,
                                     uint32_t elems, uint32_t l1Off)
{
    auto src = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(gm.GetPhyAddr(srcCol)),
                                       NdPitchLayout(1u, elems, gmCols));
    auto dst = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(l1Off),
                                       NdPitchLayout(1u, elems, elems));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{}), dst, src);
}

// L1 NZ 父 tile 列块切片 → L0A NZ（CopyL12L0A 的 nz←nz 路由。「parent
// extent == SplitA m == Mmad m」不变量由父布局+Slice 承载——原 LoadData2DParamsV2
// 四参数同步推导内化；k0 恒 16 对齐（kg/kgH 构造保证））。
__aicore__ inline void SplitA(uint32_t a1Off, uint32_t a1Cols, uint32_t k0, uint32_t kcNow, uint32_t mNow)
{
    auto src = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(a1Off),
                                       NzLayoutF(mNow, a1Cols));
    auto tile = src.Slice(AscendC::Te::MakeCoord(0u, k0), AscendC::Te::MakeShape(mNow, kcNow));
    auto dst = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(0u),
                                       NzLayoutF(mNow, kcNow));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}), dst, tile);
}

// L1 NZ tile → L0B ZN（CopyL12L0B 的 zn←nz 路由。转置分形参数
// （srcStride=2×kFrac/dstFracGap=nRep−1/srcFracGap=kFrac−1）由 NZ/ZN 布局模式
// 推导——V1/V2 参数陷阱与 n0 列片寻址全部消失）。
__aicore__ inline void SplitB(uint32_t b1Off, uint32_t kcNow, uint32_t ns)
{
    auto src = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(b1Off),
                                       NzLayoutF(kcNow, ns));
    auto dst = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(0u),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>(kcNow, ns));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}), dst, src);
}

// L1 bias 列段 → BT（CopyL12BT 路由；64B 突发对齐由 BtElems 槽容量 + Te 承载）
__aicore__ inline void LoadBiasToBT(uint32_t biasL1Off, uint32_t n0, uint32_t ns)
{
    auto src = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(biasL1Off + n0 * sizeof(float)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1u, ns));
    auto dst = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(0u),
                                       AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1u, ns));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{}), dst, src);
}

// ---------------------------------------------------------------------------
// L0C → UB drain（CopyL0C2UB + SplitM trait。裸路三件套全部由 Te 内化：
// ① CFG_ROW_MAJOR_UB/isToUB——nz→nd_ext 路由即 fixpipe 落 UB 形态；
// ② DRAIN_CHUNK=2048 分段——Te 内部 main_loop_n_size=512 自动分段；
// ③ splitM mSize=CeilAlign(m,2)/dualDstCtl=0b01——trait + 两侧布局行数承载。
// dst 为 UB 全宽平面的列片：行距=planePitch（全平面宽）由 pitch 布局承载。
// ---------------------------------------------------------------------------
constexpr AscendC::Te::CopyL0C2UBTrait GRU_L0C2UB_SPLITM_TRAIT = AscendC::Te::CopyL0C2UBTrait{
    AscendC::Te::RoundMode::DEFAULT, false, false, AscendC::Te::DUAL_DST_SPLIT_M};

struct GruL0c2UbSplitMTrait {
    using TraitType = AscendC::Te::CopyL0C2UBTrait;
    static constexpr const TraitType value = GRU_L0C2UB_SPLITM_TRAIT;
};

// ubOff：UB 平面基偏移；colOff/width：列片；mNow：行数。⚠ dst 张量行数必须取
// 逻辑全高 mEven=CeilAlign(mNow,2)（fixpipe 的 m_size=min(src,dst) 行数——若 dst
// 呈现每 AIV 半高 rowsMax 会被钳到半数，splitM 半区投递残缺：实测 AIV0 行对、
// AIV1 行全错）。物理平面每 AIV 仅 m/2 行，硬件按 dualDstCtl 对半投递，逻辑全高
// 不越界；行距=planePitch（全平面宽）由 pitch 布局承载。
__aicore__ inline void DrainToUB(uint32_t ubOff, uint32_t colOff, uint32_t width, uint32_t mNow, uint32_t nL0c,
                                 uint32_t planePitch)
{
    const uint32_t mEven = gbc::layout::CeilAlign(mNow, 2);
    // 源恒在 L0C 帧基 [0, width)——mmad 列片写帧内 [0, sliceN)（MmadGate 的
    // cTile 以帧内 n0 为偏移，outBase 不进帧坐标）；目的在 UB 平面列段
    // [colOff, colOff+width)。sliced 尾片（colOff≥nL0c）若把源也切在 colOff
    // 会越出 [mEven, nL0c] 帧 → CCU instruction address check error（实测
    // H=4097 error 263 subErrType 0x4）。
    auto src = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0u),
                                       L0cLayoutF(mEven, nL0c));
    auto dst = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubOff),
                                       NdPitchLayout(mEven, planePitch, planePitch));
    for (uint32_t off = 0; off < width; off += gbc::layout::DRAIN_CHUNK) {
        const uint32_t cn = gbc::layout::MinU(gbc::layout::DRAIN_CHUNK, width - off);
        auto srcTile = src.Slice(AscendC::Te::MakeCoord(0u, off), AscendC::Te::MakeShape(mEven, cn));
        auto dstTile = dst.Slice(AscendC::Te::MakeCoord(0u, colOff + off), AscendC::Te::MakeShape(mEven, cn));
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{}, GruL0c2UbSplitMTrait{}), dstTile, srcTile);
    }
}

// ---------------------------------------------------------------------------
// cube 通路全部走 AscendC::Te 张量原子层（CopyInNd2Nz / CopyInBiasPad / SplitA /
// SplitB / LoadBiasToBT / Mmad / DrainToUB）。gbc::vector::FeedbackToL1 为唯一手摆循环——
// Te ub_to_l1 无 ND→NZ 路由（底层 DataCopyUB2L1ND2NZImpl 存在但未暴露）。

} // namespace cube
} // namespace gbc

// ===========================================================================
// §4 gbc::vector — AIV 侧激活与回灌 + VF 主链
// ===========================================================================
namespace gbc {
namespace vector {

using gbc::layout::C0F;

// sigmoid(x) = 1/(1+exp(-x))，in-place on dst[0:n]；t1/t2 为 scratch。
// Div 而非 Reciprocal：vrec 是 ~4e-3 快速近似。NaN 逐元素传播、±Inf 饱和
// （Exp(∓Inf)→0/Inf → 1/(1+0)=1 / 1/Inf=0），与 特殊值契约一致。
__aicore__ inline void SigmoidVec(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& t1,
                                  const AscendC::LocalTensor<float>& t2, uint32_t n)
{
    AscendC::Muls(t1, dst, -1.0f, n);
    AscendC::Exp(t1, t1, n);
    AscendC::Adds(t1, t1, 1.0f, n);
    AscendC::Duplicate(t2, 1.0f, n);
    AscendC::Div(dst, t2, t1, n);
}

// tanh(x)，分段补偿（精度要求：默认 (e^{2x}−1)/(e^{2x}+1) 在近零区
// 有 1.67e7 ULP 抵消误差——|x|<6e-8 时 e^{2x} 舍入为 1.0f，tanh 硬归零）。
// 近零区用奇多项式（x·(1+O(x²))，相对精度存活），远离零用 exp 形式（答案不在
// 零附近，无抵消），Select 连接。系数为 devkit 最小极大集（0.55 切分）。
// ⚠ 饱和 clamp 用 Select 而非 Mins：Mins 的 minNum 语义 min(NaN,20)=20 会吞
// NaN；Select(GT ? 20 : x) 对 NaN 恒走 x 支路 → NaN 传播。
__aicore__ inline void TanhVec(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                               const AscendC::LocalTensor<float>& t1, const AscendC::LocalTensor<float>& t2,
                               const AscendC::LocalTensor<float>& t3, const AscendC::LocalTensor<uint8_t>& msk,
                               uint32_t n)
{
    // --- 近零：tanh(x) = x + x³·P(x²) ---
    AscendC::Mul(t1, src, src, n); // t1 = x²
    AscendC::Muls(t2, t1, 0.0157296831f, n);
    AscendC::Adds(t2, t2, -0.0523029624f, n);
    AscendC::Mul(t2, t2, t1, n);
    AscendC::Adds(t2, t2, 0.133152977f, n);
    AscendC::Mul(t2, t2, t1, n);
    AscendC::Adds(t2, t2, -0.333327681f, n);
    AscendC::Mul(t2, t2, t1, n);  // t2 = x²·P
    AscendC::Mul(t2, t2, src, n); // t2 = x³·P
    AscendC::Add(t2, t2, src, n); // t2 = x + x³·P

    // --- 远离零：(e^{2x}−1)/(e^{2x}+1)，x>20 处 Select 饱和（e^{2x} fp32 溢出） ---
    AscendC::Muls(t3, src, 2.0f, n);
    AscendC::Duplicate(t1, 20.0f, n); // t1 此处空闲（x² 已被消费）
    AscendC::Compares(msk, t3, 20.0f, AscendC::CMPMODE::GT, n);
    AscendC::Select(t3, msk, t1, t3, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    AscendC::Exp(t3, t3, n);
    AscendC::Adds(t1, t3, 1.0f, n);  // t1 = e+1
    AscendC::Adds(t3, t3, -1.0f, n); // t3 = e−1
    AscendC::Div(t3, t3, t1, n);

    // --- 连接：|x| < 0.55 走多项式（NaN 比较恒 false → 走 exp 支路 → NaN）---
    AscendC::Abs(t1, src, n);
    AscendC::Compares(msk, t1, 0.55f, AscendC::CMPMODE::LT, n);
    AscendC::Select(dst, msk, t2, t3, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
}

// ---------------------------------------------------------------------------
// 回灌 UB → L1：按列块手摆 NZ（⚠ 不能用 DataCopy(l1, ub, Nd2NzParams)——该重载
// 先经 stack buffer 中转再单次连续 burst，dstNzC0Stride 不进寻址，只能产紧凑
// tile，写不进大 tile 的列块切片；扁平重载直落 CopyUbufToCbuf，布局自述）。
// srcCols 为本列片宽 w（必须 8 对齐——srcStride 以 32B 块计，行距非整块不可表达）；
// UB 源平面行距即 w（列片紧凑排布，见 kernel.h DATAFLOW NOTES #1），故
// srcStride = w/c0 − 1 恰好跳过本行其余列块。
// dstColBlock0 = s/c0：列片外提后回灌只写 aH 的 [s, s+w) 列块区间。
// ---------------------------------------------------------------------------
__aicore__ inline void FeedbackToL1(const AscendC::LocalTensor<float>& l1, const AscendC::LocalTensor<float>& src,
                                    const gbc::layout::NzLayout& dstTile, uint32_t srcCols, uint32_t dstColBlock0,
                                    const gbc::layout::RowStripe& stripe)
{
    const uint32_t c0 = dstTile.c0;
    const uint32_t srcBlocks = srcCols / c0;
    for (uint32_t j = 0; j < srcBlocks; ++j) {
        AscendC::DataCopyParams cp;
        cp.blockCount = static_cast<uint16_t>(stripe.count); // 每行一个 32B 块
        cp.blockLen = 1;                                     // c0 元素恰 32B
        cp.srcStride = static_cast<uint16_t>(srcBlocks - 1); // 跳过本行其余源列块
        cp.dstStride = 0;                                    // NZ 块内行连续
        AscendC::DataCopy(l1[dstTile.ColBlock(dstColBlock0 + j) + stripe.base * c0], src[j * c0], cp);
    }
}

} // namespace vector
} // namespace gbc

// ===========================================================================
// VF 函数 — 逐元素主链（__simd_vf__ + asc_vf_call，工作流硬约束）
// 两处均按 的代数形态：reset_before 乘 / Sub-Mul-Add 三步 blend。
// in-place 安全性：每个 repeat 内 load→compute→store 触及不相交元素段。
// ===========================================================================

// hr = hp ⊙ r（★ reset_before：r 先与 h_prev 逐元素相乘再进候选 GEMM）
template <typename T>
__simd_vf__ inline void GruBlockCellResetMulVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
                                               uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        uint32_t remain = count - static_cast<uint32_t>(i) * oneRepeatSize; // UpdateMask 收非 const 引用
        mask = AscendC::Reg::UpdateMask<T>(remain);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::Mul(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// h = c + u·(hp − c)（Sub/Mul/Add 三步形态，避免显式构造 1−u；hAddr==cAddr 时
// in-place：c 的载入先于同段写回）
template <typename T>
__simd_vf__ inline void GruBlockCellBlendVF(__ubuf__ T* hAddr, __ubuf__ T* cAddr, __ubuf__ T* uAddr, __ubuf__ T* hpAddr,
                                            uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> cReg;
    AscendC::Reg::RegTensor<T> uReg;
    AscendC::Reg::RegTensor<T> hpReg;
    AscendC::Reg::RegTensor<T> tReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        uint32_t remain = count - static_cast<uint32_t>(i) * oneRepeatSize; // UpdateMask 收非 const 引用
        mask = AscendC::Reg::UpdateMask<T>(remain);
        AscendC::Reg::LoadAlign(hpReg, hpAddr, aReg);
        AscendC::Reg::LoadAlign(cReg, cAddr, aReg);
        AscendC::Reg::LoadAlign(uReg, uAddr, aReg);
        AscendC::Reg::Sub(tReg, hpReg, cReg, mask); // hp − c
        AscendC::Reg::Mul(tReg, uReg, tReg, mask);  // u·(hp − c)
        AscendC::Reg::Add(hReg, cReg, tReg, mask);  // c + u·(hp − c)
        AscendC::Reg::StoreAlign(hAddr, hReg, aReg, mask);
    }
}

// ===========================================================================

// ===========================================================================
// §5 GruBlockCellKernel — 算子主类
// ===========================================================================
// ===========================================================================
class GruBlockCellKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hPrev, GM_ADDR wRu, GM_ADDR wC, GM_ADDR bRu, GM_ADDR bC, GM_ADDR r,
                                GM_ADDR u, GM_ADDR c, GM_ADDR h, const GruBlockCellTilingData* td);
    __aicore__ inline void Process();

private:
    // ---- Layout：两侧（AIC/AIV）从同一组标量构造全部片上偏移——「两核同名偏移」
    // 是 AIC 的 FIX 与 AIV 的 VEC 会合的机制（无 TPipe）。
    struct Layout {
        // RowDispatch — 满行切/商余分核双模型的行分派（CubeHalf/VectorHalf 共用）。
        // 编码由 host TilingFunc 的 splitMode 字段**显式承载**（T1/K3：原算术判别式
        // `(coresUsed−1)×rowsPerCore+rowsTail==batchSize` 解码已删——隐式协议消除）：
        //   splitMode=0 满行切：主核 rowsPerCore(=sM) 行、尾核 rowsTail 行，
        //                        行基 = cluster×rowsPerCore
        //   splitMode=1 商余分核：前 rowsTail(=rem) 核 rowsPerCore(=q)+1 行、
        //                        其余 q 行，行基 = cluster×q+min(cluster,rem)
        //                        （全核启用、核间差 ≤1 行）
        __aicore__ inline void RowDispatch(uint32_t cluster, uint32_t& rowBase, uint32_t& rows) const
        {
            if (splitMode == 0) {
                rows = (cluster + 1 == coresUsed) ? rowsTail : rowsPerCore;
                rowBase = cluster * rowsPerCore;
            } else {
                rows = rowsPerCore + ((cluster < rowsTail) ? 1u : 0u);
                rowBase = cluster * rowsPerCore + gbc::layout::MinU(cluster, rowsTail);
            }
        }

        uint32_t batchSize;   // 输入维度 batch（单步 cell，无时间轴字段；B）
        uint32_t inputSize;   // 输入维度 inputSize
        uint32_t hiddenSize;  // 输入维度 hiddenSize
        uint32_t padHidden;   // host 决策：drain/平面宽（32B 对齐，记法 Hp）
        uint32_t nAl;         // host 决策：L0C / B-tile 的 N 分形范围
        uint32_t nSlice;      // host 决策：L0B 单 tile N 宽上限
        uint32_t nL0c;        // host 决策：列片宽（= L0C 片宽 = UB 平面宽）
        bool sliced;          // host 决策：nAl > nL0c（多列片）
        uint32_t rowsPerCore; // host 决策：满行切=sM / 商余分核=商 q
        uint32_t rowsTail;    // host 决策：满行切=尾核行数 / 商余=余 rem
        uint32_t coresUsed;   // host 决策：实际启用核数（blockDim）
        uint32_t splitMode;   // host 决策：0=满行切 1=商余分核（RowDispatch 显式编码）
        uint32_t mChunk;      // host 决策：单块行数（≤ rowsPerCore）
        uint32_t kc;          // host 决策：K 分块行数（16 倍数）
        uint32_t cGroups;     // host 决策：split-K 总组数
        uint32_t cGroupsX;    // host 决策：x 段组数
        uint32_t kgX;         // host 决策：x 段组宽（16 对齐）
        uint32_t kgH;         // host 决策：h 段组宽（16 对齐）
        uint32_t rowsMax;     // 派生：CeilDiv(mChunk, 2)，单 AIV drain 行数上界
        uint32_t planeElems;  // 派生：rowsMax × nL0c（列片平面容量）

        // L1（偏移，字节）：A 的 K 组流式槽 + h⊙r 全幅回灌槽 + 单 B slot + bias 共享槽
        uint32_t aKOff;       // A: K 组切片槽 [mChunk, ≤kgMax]（x / hPrev 两源共用）
        uint32_t aHOff;       // A: h⊙r 全幅回灌槽 [mChunk, Hp]（pass1 的 h 段 A 操作数）
        uint32_t bOff;        // B: 权重行块 [kc, nSlice]
        uint32_t biasSlotOff; // BT: bias 列片共享槽（r/u/c 三门按 (门, 列片) 轮用）
        uint32_t aKElems;     // aK 槽容量（元素）
        uint32_t aHElems;     // aH 槽容量（元素）
        uint32_t bElems;      // b 槽容量（元素）
        // L0（单 slot，偏移 0）：L0A/L0B/L0C/BT
        // UB（per AIV；VECOUT=drain 会合区，VECCALC=激活平面，同一 Bump 同一基址）
        uint32_t rBarOff;  // VECOUT: pass0 r drain
        uint32_t uBarOff;  // VECOUT: pass0 u drain
        uint32_t cBarOff;  // VECOUT: pass1 c drain
        uint32_t rAccOff;  // VECCALC: r 累加器
        uint32_t rCompOff; // VECCALC: r Neumaier 补偿项
        uint32_t uAccOff;  // VECCALC: u 累加器
        uint32_t uCompOff; // VECCALC: u 补偿项
        uint32_t hpOff;    // VECCALC: h_prev 现拷槽（≡t1）
        uint32_t hrOff;    // VECCALC: hr 平面（≡rBar）
        uint32_t cAccOff;  // VECCALC: c 累加器（≡uAcc）
        uint32_t cCompOff; // VECCALC: c 补偿项（≡uComp）
        uint32_t t1Off;    // VECCALC: scratch（≡hp 现拷）
        uint32_t t2Off;    // VECCALC: scratch（≡u 回读）
        uint32_t t3Off;    // VECCALC: scratch（宽路≡rAcc）
        uint32_t mskOff;   // VECCALC: uint8 mask（tanh 连接）

        // 默认构造（kernel 类需可默认构造；Init 前不消费布局值）
        __aicore__ inline Layout() : Layout(nullptr) {}

        __aicore__ inline explicit Layout(const GruBlockCellTilingData* tdIn)
            : batchSize(1),
              inputSize(1),
              hiddenSize(1),
              padHidden(gbc::layout::C0F),
              nAl(gbc::layout::CUBE_BLOCK),
              nSlice(gbc::layout::CUBE_BLOCK),
              nL0c(gbc::layout::CUBE_BLOCK),
              sliced(false),
              rowsPerCore(1),
              rowsTail(1),
              coresUsed(1),
              splitMode(0),
              mChunk(1),
              kc(gbc::layout::CUBE_BLOCK),
              cGroups(2),
              cGroupsX(1),
              kgX(gbc::layout::CUBE_BLOCK),
              kgH(gbc::layout::CUBE_BLOCK),
              rowsMax(1),
              planeElems(gbc::layout::C0F),
              aKOff(0),
              aHOff(0),
              bOff(0),
              biasSlotOff(0),
              aKElems(0),
              aHElems(0),
              bElems(0),
              rBarOff(0),
              uBarOff(0),
              cBarOff(0),
              rAccOff(0),
              rCompOff(0),
              uAccOff(0),
              uCompOff(0),
              hpOff(0),
              hrOff(0),
              cAccOff(0),
              cCompOff(0),
              t1Off(0),
              t2Off(0),
              t3Off(0),
              mskOff(0)
        {
            if (tdIn == nullptr) {
                return; // 默认构造占位（Init 前不消费）
            }
            // 决策字段全部来自 host TilingFunc（ComputeLayoutDecision）。kernel 侧
            // 只做纯算术推导（rowsMax/planeElems）与 Bump 片上偏移布局，不做任何
            // 容量/精度策略判定。决策依据见 host 侧函数注释与 tiling_struct.h。
            batchSize = static_cast<uint32_t>(tdIn->batchSize);
            inputSize = static_cast<uint32_t>(tdIn->inputSize);
            hiddenSize = static_cast<uint32_t>(tdIn->hiddenSize);
            rowsPerCore = static_cast<uint32_t>(tdIn->rowsPerCore);
            rowsTail = static_cast<uint32_t>(tdIn->rowsTail);
            coresUsed = static_cast<uint32_t>(tdIn->coreNumUsed);
            splitMode = static_cast<uint32_t>(tdIn->splitMode);
            padHidden = static_cast<uint32_t>(tdIn->padHidden);
            nAl = static_cast<uint32_t>(tdIn->nAl);
            nSlice = static_cast<uint32_t>(tdIn->nSlice);
            kc = static_cast<uint32_t>(tdIn->kc);
            nL0c = static_cast<uint32_t>(tdIn->nL0c);
            sliced = (tdIn->sliced != 0);
            mChunk = static_cast<uint32_t>(tdIn->mChunk);
            cGroups = static_cast<uint32_t>(tdIn->cGroups);
            cGroupsX = static_cast<uint32_t>(tdIn->cGroupsX);
            kgX = static_cast<uint32_t>(tdIn->kgX);
            kgH = static_cast<uint32_t>(tdIn->kgH);
            // ---- 纯算术推导（决策的确定函数）----
            rowsMax = gbc::layout::DrainRowsMax(mChunk, true);
            planeElems = rowsMax * nL0c;

            // L1 足迹（元素）：k 尾块按各自 chunk 的 parent extent 圆整，分配取最大。
            // aK 为 K 组流式槽：容量按 kgMax=max(kgX,kgH) 列——每 (列片, 组) 现搬该组
            // 的 A 切片 [mChunk, kg]，足迹与 I、H 均解耦。
            // aH 为 h⊙r 的**全幅**回灌槽 [mChunk, Hp]：pass1 的 h 段 Mmad 需要整幅 A，
            // 故 aH 不可 K 分片 —— 它是 mChunk 的上界（CeilAlign(mChunk,16)×Hp×4 ≤ L1
            // 余量），H=6144 时 mChunk ≤ 16。pass0 的 h 段**不读 aH**（改按 K 组从 GM
            // 现搬 hPrev 进 aK），因此 pass0 各列片的回灌写 aH[:, s] 与 pass0 自身的
            // A 读互不冲突（这是列片外提能与全幅回灌共存的关键）。
            const uint32_t kgMax = gbc::layout::MaxU(kgX, kgH);
            aKElems = gbc::layout::CeilDiv(kgMax, gbc::layout::C0F) *
                      gbc::layout::CeilAlign(mChunk, gbc::layout::CUBE_BLOCK) * gbc::layout::C0F;
            aHElems = gbc::layout::CeilDiv(padHidden, gbc::layout::C0F) *
                      gbc::layout::CeilAlign(mChunk, gbc::layout::CUBE_BLOCK) * gbc::layout::C0F;
            bElems = (nSlice / gbc::layout::C0F) * gbc::layout::CeilAlign(kc, gbc::layout::CUBE_BLOCK) *
                     gbc::layout::C0F;
            gbc::layout::Bump l1;
            aKOff = l1.TakeT<float>(aKElems); // A: K 组切片槽 [mChunk, ≤kgMax]
            aHOff = l1.TakeT<float>(aHElems); // A: h⊙r 全幅回灌槽 [mChunk, Hp]
            bOff = l1.TakeT<float>(bElems);   // B: 权重行块 [kc, nSlice]
            // bias 列片共享槽：容量按 nL0c（LoadBiasToBT 读 [0, sliceN)，sliceN ≤ nL0c；
            // 末片读越 Hp 界的部分落 L0C pad 列、不进 GM，与 SplitB 超读同界）。
            // r/u/c 三门按 (门, 列片) 从 GM 现搬该列段 bias（宽路指令序不变）。
            biasSlotOff = l1.TakeT<float>(gbc::cube::BtElems(nL0c));
            // L1 合计足迹 = l1.cur（Bump 游标；host 侧 CheckLayoutCapacity 同公式复刻
            // 做前置拒绝，无需在 Layout 登记冗余字段）。

            // L0（单 slot；k-chunk 间 PIPE_ALL 栅栏 + 门间 WaitFixToM 序贯复用）。
            // L0A/L0B/L0C/BT 容量不在此登记：上界由 host 决策公式钉死（nL0c 由
            // L0C×UB 联合反推、nSlice/kc 由 L0B 反推、mChunk 受 L0A 行界钳位）并由
            // tests/ut 属性测试断言；kernel 侧存一份不校验的容量字段属误导代码。

            // UB（两 AIV 同名偏移；AIC 的 FIX 以 VECOUT 偏移写入各 AIV 的 UB）：
            // 8 个 [rowsMax, nL0c] **列片**平面（6 独立 + t1/t2，t3≡rAcc）+ msk。
            // 平面行距取当前列片宽 w（≤ nL0c，恒 8 对齐）⇒ 片内紧凑，AIV 逐元素算子
            // 可按扁平 work=stripe.count×w 处理（见 DATAFLOW NOTES #1）。
            // ⚠ 别名复用的写读先序（均经既有 V2C/C2V/PIPE_ALL 栅栏闭合；改别名必须重核）：
            //   别名            末读                          首写 / 闭合屏障
            //   rBar ≡ hr       列片末轮 VF 写 hr + MTE3 读    AIC 下一列片 rBar drain——被该轮
            //                                                 V2C（MTE3 排空才置位）+ 片顶 CubeWaitVec 双重后序
            //   uBar ≡ cBar     末组 u-merge                  pass1 首片 g0 drain——被 pass0 末轮 V2C + CubeWaitVec 后序
            //   uAcc ≡ cAcc     CopyOut u（MTE3，其后有 PIPE_ALL）  pass1 首片 g0 DataCopy
            //   uComp ≡ cComp   FinFinalize(u)（片末轮）       pass1 首片 g1 Neumaier（g0 的 Duplicate 清零亦在其后）
            //   t1 ≡ hp 副本    Sigmoid/Tanh 的 V 读           hp 两次 MTE2 现拷（pass0 片末 VF hr 前 / pass1 blend
            //   前），
            //                                                 读写间有 PIPE_ALL；pass1 各列片 blend 复写 t1 故每片重拷
            //   t2 ≡ u 回读     blend 的 V 读                  pass1 片末 CopyOut c 之后的 MTE2 现拷
            //   rAcc ≡ Tanh s3  VF hr（pass0 片末轮，其后有 PIPE_ALL） pass1 的 B 端 Tanh（pass1 全程不触及 rAcc）
            //   msk             FinFinalize/Tanh 共用按需 scratch，无跨相位保持需求
            gbc::layout::Bump ub;
            rBarOff = ub.TakeT<float>(planeElems);  // VECOUT: pass0 r drain；≡ hr
            uBarOff = ub.TakeT<float>(planeElems);  // VECOUT: pass0 u drain；≡ cBar
            rAccOff = ub.TakeT<float>(planeElems);  // VECCALC: r 累加器；B 端 ≡ Tanh s3（宽路）
            rCompOff = ub.TakeT<float>(planeElems); // VECCALC: r Neumaier 补偿项
            uAccOff = ub.TakeT<float>(planeElems);  // VECCALC: u 累加器；≡ cAcc
            uCompOff = ub.TakeT<float>(planeElems); // VECCALC: u 补偿项；≡ cComp
            t1Off = ub.TakeT<float>(planeElems);    // VECCALC: scratch；≡ hp 两现拷
            t2Off = ub.TakeT<float>(planeElems);    // VECCALC: scratch；≡ u 回读
            // ---- 生命周期别名（见上表；字段名保留，Tensor 构造零改动）----
            cBarOff = uBarOff;
            cAccOff = uAccOff;
            cCompOff = uCompOff;
            hrOff = rBarOff;
            hpOff = t1Off;
            t3Off = rAccOff; // Tanh 第 3 scratch ≡ rAcc 平面（既有别名）
            mskOff = ub.Take(gbc::layout::CeilAlign(planeElems / gbc::layout::BITS_PER_BYTE,
                                                    gbc::layout::C0F)); // uint8 mask（tanh 连接）
            // UB 合计足迹 = ub.cur（Bump 游标；host 侧 CheckLayoutCapacity 同公式复刻
            // 做前置拒绝，无需在 Layout 登记冗余字段）。
        }
    };

    // ---- 单门 K 段 GEMM：A=aK 槽的 [kLo,kHi) 列块 × wGM 行块 [wRowOff+kLo, +kHi-kLo)
    // 列块 [wColOff+outBase,+sliceN)；k 按 kc 分块。outBase 为本列片的输出列基。
    // 首块种子模式 seedMode：
    // 0 = MmadBias（BT 偏置种子，覆盖 L0C——段/门的首块）
    // 1 = MmadPlain（乘积覆写种子——split-K 组 g≥1 的首块，从零起）
    // 2 = MmadAccum（在 L0C 现值上续累加——同一累加链的后续段）
    // 其余块一律 MmadAccum。
    __aicore__ inline void GateGemmRange(uint32_t a1Off, uint32_t a1Cols, uint32_t kLo, uint32_t kHi,
                                         const AscendC::GlobalTensor<float>& wGM, uint32_t wRowOff, uint32_t wColOff,
                                         uint32_t gw, uint32_t seedMode, uint32_t biasL1Off, uint32_t outBase,
                                         uint32_t mNow);

    // ---- 一个 m-chunk 的完整两段（cube 侧 / vector 侧）
    __aicore__ inline void CubeChunk(uint32_t rowBase, uint32_t mNow, bool firstChunk);
    __aicore__ inline void VectorChunk(uint32_t rowBase, uint32_t mNow);

    __aicore__ inline void CubeHalf();
    __aicore__ inline void VectorHalf();

    const GruBlockCellTilingData* td_ = nullptr;
    Layout layout_;
    AscendC::GlobalTensor<float> gmX_;     // x [B,I]
    AscendC::GlobalTensor<float> gmHPrev_; // hPrev [B,H]
    AscendC::GlobalTensor<float> gmWRu_;   // wRu [I+H,2H]
    AscendC::GlobalTensor<float> gmWC_;    // wC [I+H,H]
    AscendC::GlobalTensor<float> gmBRu_;   // bRu [2H]（[1,2H] 广播视图当前由 tiling 拒绝，见 [V3] 注释）
    AscendC::GlobalTensor<float> gmBC_;    // bC [H]（[1,H] 广播视图同上）
    AscendC::GlobalTensor<float> gmR_;     // r [B,H]
    AscendC::GlobalTensor<float> gmU_;     // u [B,H]
    AscendC::GlobalTensor<float> gmC_;     // c [B,H]
    AscendC::GlobalTensor<float> gmH_;     // h [B,H]

    static constexpr uint32_t VL_F32 = AscendC::GetVecLen() / sizeof(float);
};

// ---------------------------------------------------------------------------
// Init — GM 绑定 + Layout 构造（两侧同源标量 → 全部片上偏移一致）
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::Init(GM_ADDR x, GM_ADDR hPrev, GM_ADDR wRu, GM_ADDR wC, GM_ADDR bRu,
                                                GM_ADDR bC, GM_ADDR r, GM_ADDR u, GM_ADDR c, GM_ADDR h,
                                                const GruBlockCellTilingData* td)
{
    td_ = td;
    layout_ = Layout(td);
    gmX_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    gmHPrev_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hPrev));
    gmWRu_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wRu));
    gmWC_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wC));
    gmBRu_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bRu));
    gmBC_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bC));
    gmR_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(r));
    gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(u));
    gmC_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    gmH_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h));
}

// ---------------------------------------------------------------------------
// GateGemmRange — 一个门在 K 区间 [kLo, kHi) 上的 GEMM（L0C 列片 [outBase, +nL0c)）
// B 源：wGM[(wRowOff+k)*gw + wColOff + outBase]（k ∈ [kLo,kHi)），行距 gw（wRu=2H /
// wC=H），列宽 H。A 源：aK 槽列块 k/8 起（SplitA 内 kStep 按 kcNow）。k 按 kc 分块。
// 首块种子模式 seedMode（0=Bias / 1=Plain / 2=Accum，见声明处注释）。
// 本调用只算输出列片 [outBase, min(outBase+nL0c, nAl))（L0B n0 为片内局部列偏移、
// c0 片内列偏移、bias 槽片内索引）。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::GateGemmRange(uint32_t a1Off, uint32_t a1Cols, uint32_t kLo, uint32_t kHi,
                                                         const AscendC::GlobalTensor<float>& wGM, uint32_t wRowOff,
                                                         uint32_t wColOff, uint32_t gw, uint32_t seedMode,
                                                         uint32_t biasL1Off, uint32_t outBase, uint32_t mNow)
{
    const Layout& lyt = layout_;

    // 本 L0C 列片的列域（片内局部坐标 [0, sliceN)）
    const uint32_t sliceN = gbc::layout::MinU(lyt.nL0c, lyt.nAl - outBase);

    for (uint32_t k0 = kLo; k0 < kHi; k0 += lyt.kc) {
        const uint32_t kcNow = gbc::layout::MinU(lyt.kc, kHi - k0);
        const bool firstChunk = (k0 == kLo);
        // N 分片循环（L0B 上界）：b1 每片现搬 [kcNow, ns] 进 [kc, nSlice] 槽；
        // rowAvail 钳位防 GM 越行读。nSlice ≥ sliceN 时单迭代。
        for (uint32_t n0 = 0; n0 < sliceN; n0 += lyt.nSlice) {
            const uint32_t ns = gbc::layout::MinU(lyt.nSlice, sliceN - n0);
            const uint32_t colG = wColOff + outBase + n0;
            const uint32_t rowAvail = (gw > colG) ? (gw - colG) : 0;
            const uint32_t copyCols = gbc::layout::MinU(ns, rowAvail);
            if (copyCols > 0) {
                gbc::cube::CopyInNd2Nz(wGM, wRowOff + k0, colG, kcNow, copyCols, gw, lyt.bOff);
            }
            gbc::sync::WaitMte2ToMte1();
            if (n0 == 0) {
                gbc::cube::SplitA(a1Off, a1Cols, k0, kcNow, mNow);
            }
            gbc::cube::SplitB(lyt.bOff, kcNow, ns);
            if (seedMode == 0 && firstChunk) {
                gbc::cube::LoadBiasToBT(biasL1Off, n0, ns); // BT 每片重装该列段 bias
            }
            gbc::sync::WaitMte1ToM();
            gbc::cube::MmadGate(mNow, sliceN, n0, ns, kcNow, seedMode, firstChunk);
            // 下一片 SplitB 复用 L0B/BT、下一 k 块 SplitA 复用 L0A——定向事件对
            // （范式误区3错误3）：M→MTE1 排 L0A/L0B/BT 复用，MTE1→MTE2 排 b1/aK/bias 复写。
            gbc::sync::WaitMToMte1();
            gbc::sync::WaitMte1ToMte2();
        }
    }
}

// ---------------------------------------------------------------------------
// 列片 ↔ GM 的行跨步搬运。
// ⚠ 不能用单条 blockCount=stripe.count 的 DataCopyPad：stride=0 时 GM 侧行距取
// blockLen 本身，而列片的 blockLen = tileCols*4 < hiddenSize*4，会让第 1 行起写到
// 错误的行偏移（实测多列片 shape 只有每个 AIV stripe 的第 0 行正确）。改用
// dstStride 表达 GM 行距也不通用——dstStride 以 32B 为单位，需要
// (hiddenSize − tileCols) % 8 == 0，而 H 非 8 对齐时（如 H=4097、末列片
// tileCols=161）不成立。故逐行搬（blockCount=1，无需行距），对任意 H 成立。
// ubPitch = 本列片宽 w（UB 平面行距）；gmPitch = hiddenSize。
// ---------------------------------------------------------------------------
__aicore__ inline void CopyTileToGm(const AscendC::GlobalTensor<float>& gm, uint64_t gmElemOff,
                                    const AscendC::LocalTensor<float>& ub, uint32_t rows, uint32_t ubPitch,
                                    uint32_t gmPitch, uint32_t tileCols)
{
    AscendC::DataCopyExtParams e;
    e.blockCount = 1;
    e.blockLen = tileCols * sizeof(float);
    e.srcStride = 0;
    e.dstStride = 0;
    for (uint32_t rr = 0; rr < rows; ++rr) {
        AscendC::DataCopyPad(gm[gmElemOff + static_cast<uint64_t>(rr) * gmPitch], ub[rr * ubPitch], e);
    }
}

__aicore__ inline void CopyTileFromGm(const AscendC::GlobalTensor<float>& gm, uint64_t gmElemOff,
                                      const AscendC::LocalTensor<float>& ub, uint32_t rows, uint32_t ubPitch,
                                      uint32_t gmPitch, uint32_t tileCols)
{
    AscendC::DataCopyExtParams e;
    e.blockCount = 1;
    e.blockLen = tileCols * sizeof(float);
    e.srcStride = 0;
    e.dstStride = 0;
    const AscendC::DataCopyPadExtParams<float> pad{};
    for (uint32_t rr = 0; rr < rows; ++rr) {
        AscendC::DataCopyPad(ub[rr * ubPitch], gm[gmElemOff + static_cast<uint64_t>(rr) * gmPitch], e, pad);
    }
}

// ---------------------------------------------------------------------------
// CubeChunk — 一个 m-chunk 的 cube 侧：
//   pass0（全列片 × 全组，r/u 两门）→ CubeWaitVec（h⊙r 已全部落 GM）
//   → pass1（全列片 × 全组，c 门）。
// 前一块的收尾 V2C 由循环顶部的 CubeWaitVec 消费（WAR：UB drain tile 的复用排在
// 其后）。每列片、每组一轮 C2V/V2C 握手；轮次配平见文件头 DATAFLOW NOTES 与
// gbc::sync 注释（每块 AIV 置位 2·nTiles·cGroups 次，cube 消费同数）。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::CubeChunk(uint32_t rowBase, uint32_t mNow, bool firstChunk)
{
    const Layout& lyt = layout_;
    // 片上张量由 gbc::cube:: Te 助手按 Layout 槽偏移现构，不预建 LocalTensor 视图。

    if (!firstChunk) {
        gbc::sync::CubeWaitVec(); // 前一块 epilogue 完成（rBar/uBar 可覆写）
    }

    // ---- pass0 · r/u 两门：[x|h_prev] @ wRu + bRu，列片外提 × split-K 分组内嵌。
    // 每组：A 切片现搬进 aK → r 门 Mmad → drain rBar → u 门 Mmad → drain uBar →
    // C2V（同轮交付 r/u 两组部分和，轮数不翻倍）。bias 仅首组（g==0，必为 x 段首组）
    // 种子，按列片从 GM 现搬进共享槽。
    // （r_bar/u_bar 的 GEMM 噪声经 σ 压缩后仍以 2.4e-7 级残差经 h⊙r 回流进 c——
    // 三门全部走 AIV Neumaier 补偿合并。）
    for (uint32_t s = 0; s < lyt.padHidden; s += lyt.nL0c) {
        const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
        const uint32_t frameN = gbc::layout::MinU(lyt.nL0c, lyt.nAl - s); // L0C 帧宽（Mmad cParent 同源）
        for (uint32_t g = 0; g < lyt.cGroups; ++g) {
            if (s > 0 || g > 0) {
                gbc::sync::CubeWaitVec(); // AIV 已消费上一轮 rBar/uBar
                gbc::sync::WaitFixToM();  // 上一轮 drain 排空后方可覆写 L0C
            }
            const bool isX = (g < lyt.cGroupsX);
            const uint32_t kg = isX ? lyt.kgX : lyt.kgH;
            const uint32_t kLim = isX ? lyt.inputSize : lyt.hiddenSize;
            const uint32_t k0 = isX ? (g * lyt.kgX) : ((g - lyt.cGroupsX) * lyt.kgH);
            const uint32_t kw = gbc::layout::MinU(kLim - k0, kg);
            // A 切片现搬（x / h_prev 均按 K 组从 GM 进 aK；aH 全程只被回灌写）
            gbc::cube::CopyInNd2Nz(isX ? gmX_ : gmHPrev_, rowBase, k0, mNow, kw, kLim, lyt.aKOff);
            if (g == 0 && s < lyt.hiddenSize) {
                gbc::cube::CopyInBiasPad(gmBRu_, s, static_cast<uint32_t>(GRU_GATE_NUM) * lyt.hiddenSize,
                                         gbc::layout::MinU(lyt.hiddenSize - s, w), lyt.biasSlotOff);
            }
            gbc::sync::WaitMte2ToMte1();
            // r 门（wRu 列块 [0,H)，bRu[0:H]）
            GateGemmRange(lyt.aKOff, kw, 0, kw, gmWRu_, isX ? k0 : (lyt.inputSize + k0), 0,
                          static_cast<uint32_t>(GRU_GATE_NUM) * lyt.hiddenSize, (g == 0) ? 0 : 1,
                          (g == 0) ? lyt.biasSlotOff : 0u, s, mNow);
            gbc::sync::WaitMToFix();
            gbc::cube::DrainToUB(lyt.rBarOff, 0, w, mNow, frameN, w);
            gbc::sync::WaitFixToM(); // L0C 复用于 u 门
            if (g == 0 && s < lyt.hiddenSize) {
                gbc::cube::CopyInBiasPad(gmBRu_, lyt.hiddenSize + s,
                                         static_cast<uint32_t>(GRU_GATE_NUM) * lyt.hiddenSize,
                                         gbc::layout::MinU(lyt.hiddenSize - s, w), lyt.biasSlotOff);
                gbc::sync::WaitMte2ToMte1();
            }
            // u 门（wRu 列块 [H,2H)，bRu[H:2H]）；A 切片与 r 门同槽复用（同 k 区间）
            GateGemmRange(lyt.aKOff, kw, 0, kw, gmWRu_, isX ? k0 : (lyt.inputSize + k0), lyt.hiddenSize,
                          static_cast<uint32_t>(GRU_GATE_NUM) * lyt.hiddenSize, (g == 0) ? 0 : 1,
                          (g == 0) ? lyt.biasSlotOff : 0u, s, mNow);
            gbc::sync::WaitMToFix();
            gbc::cube::DrainToUB(lyt.uBarOff, 0, w, mNow, frameN, w);
            gbc::sync::CubeSignalVec(); // 本轮 r/u 部分和已落 UB（列片 s）
        }
    }

    // ---- 等待 AIV 把全部列片的 h⊙r 回灌进 L1 aH ----
    // V2C 为 FIFO barrier：消费末列片末组的那一轮即蕴含前面全部轮次（含各列片的
    // FeedbackToL1）已完成；CubeWaitVec 尾随的 PIPE_ALL 使 aH 对后续 MTE1 可见。
    gbc::sync::CubeWaitVec(); // 内含 PIPE_ALL：pass0 末轮 drain 的 FIX 亦已排空

    // ---- pass1 · c 门：[x | h⊙r] @ wC + bC，同款列片外提 × split-K 分组。
    // x 段按 K 组从 GM 现搬进 aK；h 段读 **aH 全幅回灌槽**（pass0 各列片已把
    // h⊙r 写满 [0,Hp) 列），故 h 段的 kLo/kHi 是 aH 内的绝对列区间。
    // bias 仅首组种子。每组一段 Mmad → drain cBar → C2V，轮间 CubeWaitVec 为 cBar
    // 的 WAR 屏障。----
    for (uint32_t s = 0; s < lyt.padHidden; s += lyt.nL0c) {
        const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
        const uint32_t frameN = gbc::layout::MinU(lyt.nL0c, lyt.nAl - s);
        for (uint32_t g = 0; g < lyt.cGroups; ++g) {
            if (s > 0 || g > 0) {
                gbc::sync::CubeWaitVec(); // AIV 已消费上一轮 cBar
                gbc::sync::WaitFixToM();  // 上一轮 drain 排空后方可覆写 L0C
            }
            const bool isX = (g < lyt.cGroupsX);
            const uint32_t kg = isX ? lyt.kgX : lyt.kgH;
            const uint32_t kLim = isX ? lyt.inputSize : lyt.hiddenSize;
            const uint32_t k0 = isX ? (g * lyt.kgX) : ((g - lyt.cGroupsX) * lyt.kgH);
            const uint32_t kw = gbc::layout::MinU(kLim - k0, kg);
            if (isX) {
                gbc::cube::CopyInNd2Nz(gmX_, rowBase, k0, mNow, kw, kLim, lyt.aKOff);
                if (g == 0 && s < lyt.hiddenSize) {
                    gbc::cube::CopyInBiasPad(gmBC_, s, lyt.hiddenSize, gbc::layout::MinU(lyt.hiddenSize - s, w),
                                             lyt.biasSlotOff);
                }
                gbc::sync::WaitMte2ToMte1();
                GateGemmRange(lyt.aKOff, kw, 0, kw, gmWC_, k0, 0, lyt.hiddenSize, (g == 0) ? 0 : 1,
                              (g == 0) ? lyt.biasSlotOff : 0u, s, mNow);
            } else {
                // h 段：A 为 aH **全幅**回灌槽，故 kLo/kHi 用 aH 内的绝对列区间
                // [k0, k0+kw)，而 wRowOff 只给段基 inputSize（行号 = wRowOff + k0i，
                // k0i 已含组偏移）。⚠ 与 pass0 的 h 段约定不同——那边 A 是 aK 组内
                // 切片（kLo=0），故 wRowOff 必须是 inputSize+k0。两处混用会使权重
                // 行号变成 inputSize+2·k0 而越界读 wC（H=8 时 k0≡0 侥幸不炸）。
                // bias 只可能落在 g==0，而 cGroupsX ≥ 1 保证 g==0 恒为 x 段组，
                // 故本支无 bias 装载（seedMode 恒 1 = Plain 覆写，组间由 AIV 合并）。
                GateGemmRange(lyt.aHOff, lyt.hiddenSize, k0, k0 + kw, gmWC_, lyt.inputSize, 0, lyt.hiddenSize, 1, 0u, s,
                              mNow);
            }
            gbc::sync::WaitMToFix();
            gbc::cube::DrainToUB(lyt.cBarOff, 0, w, mNow, frameN, w);
            gbc::sync::CubeSignalVec(); // 本轮部分和已落 UB（列片 s）
        }
    }
}

// ---------------------------------------------------------------------------
// CubeHalf — 本 cluster 的全部 m-chunk（batch 行切，无跨核栅栏）
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::CubeHalf()
{
    const Layout& lyt = layout_;
    // ⚠ 950 核号约定：AIC 的 GetBlockIdx() 即 cluster 号
    const uint32_t cluster = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t blockBase;
    uint32_t rowsThisCore;
    lyt.RowDispatch(cluster, blockBase, rowsThisCore); // 满行切/商余分核双模型行分派
    if (blockBase >= lyt.batchSize) {
        return; // 无行的 cluster：与伴 AIV 一同早退，不参与任何 flag 轮
    }
    const uint32_t nChunk = gbc::layout::CeilDiv(rowsThisCore, lyt.mChunk);
    for (uint32_t ci = 0; ci < nChunk; ++ci) {
        const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, rowsThisCore - ci * lyt.mChunk);
        CubeChunk(blockBase + ci * lyt.mChunk, mNow, ci == 0);
    }
    // 收尾等待：消费末块末轮的 V2C，使整个 launch 的 V2C 轮次零残留（跨 launch
    // 的 flag 残留可能满足下一次 launch 的早期等待——防患于未然）。
    gbc::sync::CubeWaitVec();
    AscendC::PipeBarrier<PIPE_ALL>;
}

// ---------------------------------------------------------------------------
// VectorChunk — 一个 m-chunk 的 AIV 侧：pass0 全列片 → pass1 全列片
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::VectorChunk(uint32_t rowBase, uint32_t mNow)
{
    const Layout& lyt = layout_;
    AscendC::LocalTensor<float> rBar(AscendC::TPosition::VECOUT, lyt.rBarOff, lyt.planeElems);
    AscendC::LocalTensor<float> uBar(AscendC::TPosition::VECOUT, lyt.uBarOff, lyt.planeElems);
    AscendC::LocalTensor<float> cBar(AscendC::TPosition::VECOUT, lyt.cBarOff, lyt.planeElems);
    AscendC::LocalTensor<float> rAcc(AscendC::TPosition::VECCALC, lyt.rAccOff, lyt.planeElems);
    AscendC::LocalTensor<float> rComp(AscendC::TPosition::VECCALC, lyt.rCompOff, lyt.planeElems);
    AscendC::LocalTensor<float> uAcc(AscendC::TPosition::VECCALC, lyt.uAccOff, lyt.planeElems);
    AscendC::LocalTensor<float> uComp(AscendC::TPosition::VECCALC, lyt.uCompOff, lyt.planeElems);
    AscendC::LocalTensor<float> hp(AscendC::TPosition::VECCALC, lyt.hpOff, lyt.planeElems);
    AscendC::LocalTensor<float> hr(AscendC::TPosition::VECCALC, lyt.hrOff, lyt.planeElems);
    AscendC::LocalTensor<float> cAcc(AscendC::TPosition::VECCALC, lyt.cAccOff, lyt.planeElems);
    AscendC::LocalTensor<float> cComp(AscendC::TPosition::VECCALC, lyt.cCompOff, lyt.planeElems);
    AscendC::LocalTensor<float> t1(AscendC::TPosition::VECCALC, lyt.t1Off, lyt.planeElems);
    AscendC::LocalTensor<float> t2(AscendC::TPosition::VECCALC, lyt.t2Off, lyt.planeElems);
    AscendC::LocalTensor<float> t3(AscendC::TPosition::VECCALC, lyt.t3Off, lyt.planeElems);
    AscendC::LocalTensor<uint8_t> msk(
        AscendC::TPosition::VECCALC, lyt.mskOff,
        gbc::layout::CeilAlign(lyt.planeElems / gbc::layout::BITS_PER_BYTE, gbc::layout::C0F));
    AscendC::LocalTensor<float> aH(AscendC::TPosition::A1, lyt.aHOff, lyt.aHElems);

    // 与 drain 相同的 stripe（本 AIV 独占的行区间）
    const gbc::layout::RowStripe stripe = gbc::layout::DrainStripe(mNow, true,
                                                                   static_cast<uint32_t>(AscendC::GetSubBlockIdx()));
    const bool hasRows = stripe.count > 0;
    const uint32_t gRow = rowBase + stripe.base; // 本 AIV 起始的全局行

    // ---- pass0：逐列片完成 r/u 的补偿累加 → 门激活 → reset_before 乘 → 落 GM ----
    // 每组：VecWaitCube → rBar/uBar 并入 (rAcc,rComp)/(uAcc,uComp) → VecSignalCube
    // （下一轮 drain 的 WAR 屏障）。g=0 直接以首组部分和初始化累加器。
    // ⚠ 列片收尾（finalize/sigmoid/copyout/hp/VF/hr 落 GM）必须并入**末组轮内**再
    // VecSignalCube：V2C 按 FIFO 计数配对，且 cube 的 pass0→pass1 CubeWaitVec 消费的
    // 正是末列片末组的 V2C——收尾若单独成轮，cube 会在 hr 尚未落 GM 时就开始 pass1
    // 读 aH（静默错数）。并入后每块 V2C 严格 2·nTiles·cGroups 设 = 同数等，轮次配平。
    for (uint32_t s = 0; s < lyt.padHidden; s += lyt.nL0c) {
        const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
        const uint32_t tileCols = (s < lyt.hiddenSize) ? gbc::layout::MinU(w, lyt.hiddenSize - s) : 0;
        const uint32_t work = stripe.count * w;
        const uint64_t gmOff = static_cast<uint64_t>(gRow) * lyt.hiddenSize + s;
        const uint16_t rep = static_cast<uint16_t>(AscendC::CeilDivision(work, VL_F32));

        if (hasRows) {
            AscendC::Duplicate(rComp, 0.0f, work);
            AscendC::Duplicate(uComp, 0.0f, work);
        }
        for (uint32_t g = 0; g < lyt.cGroups; ++g) {
            gbc::sync::VecWaitCube();
            AscendC::PipeBarrier<PIPE_ALL>();
            if (hasRows) {
                if (g == 0) {
                    AscendC::DataCopy(rAcc, rBar, work);
                    AscendC::DataCopy(uAcc, uBar, work);
                    // 轮尾定向事件（范式误区3错误3）：rBar/uBar 的 V 读须先于本轮
                    // VecSignalCube 的 MTE3 跨核置位完成——命名 V→MTE3 hazard。
                    gbc::sync::WaitVToMte3();
                } else {
                    // Neumaier（Knuth TwoSum，无分支形态）：t = a + p 的舍入损失进补偿项。
                    // ⚠ accumulator 回拷（UB→UB DataCopy，V 管指令）与下一链对 t1 的复写
                    // 同属 V 管，程序序即保序；此处 PIPE_V 为条款口径的同管防御标记。
                    AscendC::Add(t1, rAcc, rBar, work);
                    AscendC::Sub(t2, rAcc, t1, work);
                    AscendC::Add(t2, t2, rBar, work);
                    AscendC::Add(rComp, rComp, t2, work);
                    AscendC::DataCopy(rAcc, t1, work);
                    AscendC::PipeBarrier<PIPE_V>(); // t1 拷贝读完成前不得复写（同管保序）
                    AscendC::Add(t1, uAcc, uBar, work);
                    AscendC::Sub(t2, uAcc, t1, work);
                    AscendC::Add(t2, t2, uBar, work);
                    AscendC::Add(uComp, uComp, t2, work);
                    AscendC::DataCopy(uAcc, t1, work);
                    // 轮尾定向事件（非末组轮）：uBar/rBar 的 V 读须先于 VecSignalCube 的
                    // MTE3 跨核置位——命名 V→MTE3 hazard。
                    gbc::sync::WaitVToMte3();
                }
                if (g == lyt.cGroups - 1) {
                    // 列片收尾（并入末轮，见上 ⚠）。hp 现拷进 t1（≡hpOff；Sigmoid 的
                    // t1 V 读与 MTE2 写之间有上一行 PIPE_ALL）
                    gbc::layout::FinFinalize(rAcc, rComp, t1, t2, msk, work); // r_bar 就绪
                    gbc::layout::FinFinalize(uAcc, uComp, t1, t2, msk, work); // u_bar 就绪
                    AscendC::PipeBarrier<PIPE_V>(); // FinFinalize 与 SigmoidVec 同为 V 管指令，同管保序
                    gbc::vector::SigmoidVec(rAcc, t1, t2, work); // r = σ(r_bar)（就地）
                    gbc::vector::SigmoidVec(uAcc, t1, t2, work); // u = σ(u_bar)（就地）
                    // 定向事件对（原 PIPE_ALL）：V 写 rAcc/uAcc → MTE3（CopyOut r/u/hr 读）
                    // 与 → MTE2（hp 现拷写 t1 槽）双 hazard，分别命名闭合。
                    gbc::sync::WaitVToMte3();
                    gbc::sync::WaitVToMte2();
                    // 输出 r（fp32 直出，字节精确）/ u（pass1 blend 经 t2 回读）/ h_prev 副本（t1 槽）
                    CopyTileToGm(gmR_, gmOff, rAcc, stripe.count, w, lyt.hiddenSize, tileCols);
                    CopyTileToGm(gmU_, gmOff, uAcc, stripe.count, w, lyt.hiddenSize, tileCols);
                    CopyTileFromGm(gmHPrev_, gmOff, hp, stripe.count, w, lyt.hiddenSize, tileCols);
                    AscendC::PipeBarrier<PIPE_ALL>(); // MTE2 写 hp → VF 读（跨管必须全栅栏，见文件头）
                    // ★ reset_before：hr = h_prev ⊙ r（VF 主链；hr 落 rBar 平面——
                    // 末组 r-merge 已消费完该 drain，同 AIV 程序序）
                    asc_vf_call<GruBlockCellResetMulVF<float>>(reinterpret_cast<__ubuf__ float*>(hr.GetPhyAddr()),
                                                               reinterpret_cast<__ubuf__ float*>(hp.GetPhyAddr()),
                                                               reinterpret_cast<__ubuf__ float*>(rAcc.GetPhyAddr()),
                                                               work, VL_F32, rep);
                    AscendC::PipeBarrier<PIPE_ALL>(); // VF 异步读闭合 → 回灌（必须全栅栏：VF 异步单元
                                                      // 完成不被命名 V 管事件闭合，定向化实测 pass1 全错）
                    // 回灌 UB→L1：只写 aH 的 [s, s+w) 列块区间、本 stripe 的行区间
                    // （950 独有，不过 GM）。pass0 的 h 段不读 aH，故与本片/后续片的
                    // pass0 GEMM 无竞争；pass1 由块级 CubeWaitVec 保证全幅已写满。
                    gbc::vector::FeedbackToL1(aH, hr,
                                              gbc::layout::NzLayout(mNow, lyt.padHidden, gbc::layout::C0F, mNow), w,
                                              s / gbc::layout::C0F, stripe);
                }
            }
            gbc::sync::VecSignalCube(); // ⚠ count==0 的 AIV 也必须 set——barrier
        }
    }

    // ---- pass1：逐列片完成 c_bar 的补偿累加 + 候选激活 + 状态更新 ----
    // 每组：VecWaitCube → cBar 并入 (cAcc,cComp) → VecSignalCube（下一轮 drain 的
    // WAR 屏障）。g=0 直接以首组部分和初始化累加器；g≥1 用 Neumaier：
    // t = cAcc + p; cComp += (cAcc − t) + p; cAcc = t（组间合并近精确）。
    for (uint32_t s = 0; s < lyt.padHidden; s += lyt.nL0c) {
        const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
        const uint32_t tileCols = (s < lyt.hiddenSize) ? gbc::layout::MinU(w, lyt.hiddenSize - s) : 0;
        const uint32_t work = stripe.count * w;
        const uint64_t gmOff = static_cast<uint64_t>(gRow) * lyt.hiddenSize + s;
        const uint16_t rep = static_cast<uint16_t>(AscendC::CeilDivision(work, VL_F32));

        if (hasRows) {
            AscendC::Duplicate(cComp, 0.0f, work); // 补偿项清零（首组前）
        }
        for (uint32_t g = 0; g < lyt.cGroups; ++g) {
            gbc::sync::VecWaitCube();
            AscendC::PipeBarrier<PIPE_ALL>();
            if (hasRows) {
                if (g == 0) {
                    AscendC::DataCopy(cAcc, cBar, work); // 首组部分和即累加器初值
                    // 轮尾定向事件（原 PIPE_ALL）：cBar 的 V 读先于本轮 VecSignalCube 的
                    // MTE3 跨核置位——命名 V→MTE3 hazard。
                    gbc::sync::WaitVToMte3();
                } else {
                    // Neumaier：大数吃小数的舍入损失进补偿项
                    AscendC::Add(t1, cAcc, cBar, work);   // t = cAcc + p
                    AscendC::Sub(t2, cAcc, t1, work);     // cAcc − t
                    AscendC::Add(t2, t2, cBar, work);     // + p
                    AscendC::Add(cComp, cComp, t2, work); // cComp += (cAcc − t) + p
                    AscendC::DataCopy(cAcc, t1, work);    // cAcc = t
                    // 轮尾定向事件（非末组轮，原 PIPE_ALL）：cBar 的 V 读先于 VecSignalCube
                    // 的 MTE3 跨核置位——命名 V→MTE3 hazard。
                    gbc::sync::WaitVToMte3();
                }
            }
            gbc::sync::VecSignalCube(); // ⚠ count==0 的 AIV 也必须 set——barrier
        }
        if (hasRows) {
            gbc::layout::FinFinalize(cAcc, cComp, t1, t2, msk, work); // c_bar 就绪（含有限性守卫）
            AscendC::PipeBarrier<PIPE_V>(); // FinFinalize 与 TanhVec 同为 V 管指令，同管保序
            // Tanh 的第 3 scratch 落 rAcc 平面（≡t3Off；rAcc 自 pass0 末列片 VF hr 后
            // 无人触及，pass1 各组亦只用 cAcc/cComp/t1/t2）
            gbc::vector::TanhVec(cAcc, cAcc, t1, t2, t3, msk, work); // c = tanh(c_bar)（分段补偿）
            gbc::sync::WaitVToMte3(); // V 写 cAcc → MTE3 读（原 PIPE_ALL；命名 V→MTE3）
            CopyTileToGm(gmC_, gmOff, cAcc, stripe.count, w, lyt.hiddenSize, tileCols); // 输出 c
            AscendC::PipeBarrier<PIPE_ALL>(); // c 的 MTE3 读完成前不得就地改写（全栅栏：随后消费者是 VF，
                                              // MTE3→VF 界面不适用命名事件，保守 PIPE_ALL）
            // blend 双源现拷（Layout 别名表：uAcc≡cAcc 已被 pass1 复用、hp 副本在 t1 槽
            // 被本片各组 merge 覆写——两源均须重取；gmU_ 于 pass0 已落盘，回读值逐位
            // 相同，数学不变）
            CopyTileFromGm(gmU_, gmOff, t2, stripe.count, w, lyt.hiddenSize, tileCols);     // u 回读副本
            CopyTileFromGm(gmHPrev_, gmOff, t1, stripe.count, w, lyt.hiddenSize, tileCols); // h_prev 副本
            AscendC::PipeBarrier<PIPE_ALL>(); // MTE2 写 t1/t2 → VF 读（跨管必须全栅栏，见文件头）
            // h = c + u·(hp − c)（VF 主链，Sub/Mul/Add 三步形态；就地写回 cAcc 平面）
            asc_vf_call<GruBlockCellBlendVF<float>>(reinterpret_cast<__ubuf__ float*>(cAcc.GetPhyAddr()),
                                                    reinterpret_cast<__ubuf__ float*>(cAcc.GetPhyAddr()),
                                                    reinterpret_cast<__ubuf__ float*>(t2.GetPhyAddr()),
                                                    reinterpret_cast<__ubuf__ float*>(t1.GetPhyAddr()), work, VL_F32,
                                                    rep);
            AscendC::PipeBarrier<PIPE_ALL>(); // VF 异步读闭合 → MTE3 输出 h（必须全栅栏，同上 VF 规则）
            CopyTileToGm(gmH_, gmOff, cAcc, stripe.count, w, lyt.hiddenSize, tileCols); // 输出 h（cAcc 就地复用）
            // 下一列片首组的 DataCopy(cAcc/cComp) 为 V 管写，与本片末尾的 MTE3 读跨管：
            // 由下一片首轮的 VecWaitCube 后 PipeBarrier<PIPE_ALL> 闭合（与既有块间口径一致）。
        }
        // 末列片末组的 VecSignalCube 即本块的 V2C 收尾（下一块 pass0 drain 的 WAR
        // 屏障；本片后续 tanh/blend/copyout 只读 AIV 私有平面与 GM，与 cube 无竞争）。
    }
}

// ---------------------------------------------------------------------------
// VectorHalf — 本 cluster 的全部 m-chunk（与 CubeHalf 同一套行切/块切标量）
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::VectorHalf()
{
    const Layout& lyt = layout_;
    // ⚠ 950 核号约定：AIV 的 GetBlockIdx() 是扁平 subcore 号，÷ 核比得 cluster
    const uint32_t cluster = static_cast<uint32_t>(AscendC::GetBlockIdx()) /
                             static_cast<uint32_t>(AscendC::GetTaskRatio());
    uint32_t blockBase;
    uint32_t rowsThisCore;
    lyt.RowDispatch(cluster, blockBase, rowsThisCore); // 满行切/商余分核双模型行分派
    if (blockBase >= lyt.batchSize) {
        return;
    }
    const uint32_t nChunk = gbc::layout::CeilDiv(rowsThisCore, lyt.mChunk);
    for (uint32_t ci = 0; ci < nChunk; ++ci) {
        const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, rowsThisCore - ci * lyt.mChunk);
        VectorChunk(blockBase + ci * lyt.mChunk, mNow);
    }
    AscendC::PipeBarrier<PIPE_ALL>;
}

// ---------------------------------------------------------------------------
// Process — MIX 分派（cube 半 / vector 半）
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::Process()
{
    if ASCEND_IS_AIC {
        CubeHalf();
    }
    if ASCEND_IS_AIV {
        VectorHalf();
    }
    AscendC::PipeBarrier<PIPE_ALL>;
}

#endif // GRU_BLOCK_CELL_KERNEL_H
