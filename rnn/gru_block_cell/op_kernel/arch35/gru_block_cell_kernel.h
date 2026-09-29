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
// reset_before：ru_bar=[x,h_prev]·wRu+bRu；r=σ(ru_bar[:,0:H])；u=σ(ru_bar[:,H:2H])；
//               c=tanh([x,h_prev⊙r]·wC+bC)；h=c+u·(h_prev−c)
// 结构：M 按 mChunk 分块，每块 pass0（r/u 门）→ pass1（c 门）；列片 s 最外、split-K 组
// g 最内（kg=16 精度契约）；AIC/AIV 以 C2V/V2C 单 flag 对按 (片,组) 轮握手。
// 设计依据、失败史与判别证据：log/gru_block_cell_perf_session_archive_20260928.md。
// ⚠ 勿回退（均有实测翻车记录，细节见存档）：① h 输出经 t3 staging + WaitMte3ToV
// （Fix B）；② pass1 奇偶 need 分级（纯深度-2 触 L1 无写仲裁）；③ aHR 三槽+前瞻-3；
// ④ 批量搬运 stride 单位非对称（GM=1 字节/UB=32B）；⑤ A1 splitMode=2 相位拆分每核
// 恰一次 SyncAll（死锁契约）；⑥ 列片外提（全幅平面钉死 mChunk → HBM 墙）。
// 文件组织：§1 gbc::layout 布局算术 §2 gbc::sync 同步事件 §3 gbc::cube AIC 数据通路
// §4 gbc::vector AIV 激活/回灌+VF 主链 §5 GruBlockCellKernel 主类。
// tiling_struct.h 为 host—kernel 共享 POD、gru_block_cell_struct.h 为 ASCENDC_TPL
// 编译期分发表（均不可并入本文件）。

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

// 片上容量与预算常量在 host 侧（决策唯一真值：gru_block_cell_tiling.cpp 的
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
__aicore__ inline void WaitMte2ToV() { PipeWait<AscendC::HardEvent::MTE2_V>(); } // MTE2 落 UB 完 → V 读
__aicore__ inline void WaitMte3ToV() { PipeWait<AscendC::HardEvent::MTE3_V>(); } // MTE3 读 UB 完 → V 复写
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
        // RowDispatch — 行分派（host splitMode 显式编码，kernel 不做算术判别）：
        //   0=满行切（主核 rowsPerCore、尾核 rowsTail）  1=商余分核（前 rowsTail 核 +1 行）
        //   2=A1 2D 分派：行块商余同 1，但作用于 rowChunk=cluster/sliceCores
        __aicore__ inline void RowDispatch(uint32_t rowChunk, uint32_t& rowBase, uint32_t& rows) const
        {
            if (splitMode == 0) {
                rows = (rowChunk + 1 == coresUsed) ? rowsTail : rowsPerCore;
                rowBase = rowChunk * rowsPerCore;
            } else {
                rows = rowsPerCore + ((rowChunk < rowsTail) ? 1u : 0u);
                rowBase = rowChunk * rowsPerCore + gbc::layout::MinU(rowChunk, rowsTail);
            }
        }

        // SliceDispatch — A1 列片组分派：nTiles 商余分给 sliceCores 组，输出本核列域
        // [sliceLoCol, sliceHiCol)。旧路 sliceCores==1 退化为全幅 [0, padHidden)。
        __aicore__ inline void SliceDispatch(uint32_t sliceGrp, uint32_t& sliceLoCol, uint32_t& sliceHiCol) const
        {
            const uint32_t nTiles = gbc::layout::CeilDiv(padHidden, nL0c);
            const uint32_t tilesPer = nTiles / sliceCores;
            const uint32_t remT = nTiles % sliceCores;
            const uint32_t lo = sliceGrp * tilesPer + gbc::layout::MinU(sliceGrp, remT);
            const uint32_t cnt = tilesPer + ((sliceGrp < remT) ? 1u : 0u);
            sliceLoCol = lo * nL0c;
            sliceHiCol = gbc::layout::MinU(padHidden, (lo + cnt) * nL0c);
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
        uint32_t splitMode;   // host 决策：0=满行切 1=商余分核 2=行列 2D 分派（A1）
        uint32_t sliceCores;  // host 决策：A1 列片组数 nsc（旧路恒 1）
        // ---- per-core 分派（cluster 的确定函数；片上偏移仍全核同名）----
        uint32_t rowBase;      // 本核行块起始行
        uint32_t rowsThisCore; // 本核行块行数（splitMode=2 防御域可 0——仅参与屏障）
        uint32_t sliceLoCol;   // 本核列片域起始列（含；旧路恒 0）
        uint32_t sliceHiCol;   // 本核列片域结束列（不含；旧路恒 padHidden）
        uint32_t mChunk;       // host 决策：单块行数（≤ rowsPerCore）
        uint32_t kc;           // host 决策：K 分块行数（16 倍数）
        uint32_t cGroups;      // host 决策：split-K 总组数
        uint32_t cGroupsX;     // host 决策：x 段组数
        uint32_t kgX;          // host 决策：x 段组宽（16 对齐）
        uint32_t kgH;          // host 决策：h 段组宽（16 对齐）
        uint32_t rowsMax;      // 派生：CeilDiv(mChunk, 2)，单 AIV drain 行数上界
        uint32_t planeElems;   // 派生：rowsMax × nL0c（列片平面容量）

        // L1（偏移，字节）：A 的 K 组流式槽（x/hPrev 双槽 + hr 双槽）+ B 双槽 + bias 共享槽
        uint32_t aKOff;       // A: x/hPrev 切片双槽 [mChunk, ≤kgMax]×2（S2 预取）
        uint32_t aHROff;      // A: pass1 hr 切片三槽 [mChunk, ≤kgMax]×3（S3' 前瞻-3；AIV 写/AIC 读）
        uint32_t bOff;        // B: 权重行块双槽 [kc, nSlice]×2（S2 预取）
        uint32_t biasSlotOff; // BT: bias 列片共享槽（r/u/c 三门按 (门, 列片) 轮用）
        uint32_t aKElems;     // aK/aHR 单槽容量（元素）
        uint32_t bElems;      // b 单槽容量（元素）
        // L0（单 slot，偏移 0）：L0A/L0B/L0C/BT
        // UB（per AIV；VECOUT=drain 会合区，VECCALC=激活平面，同一 Bump 同一基址）
        uint32_t rBarOff;  // VECOUT: pass0 r drain（偶数轮；≡ pass1 cBar1）
        uint32_t uBarOff;  // VECOUT: pass0 u drain（偶数轮；≡ pass1 cBar0）
        uint32_t rBar1Off; // VECOUT: pass0 r drain（奇数轮——S3'' 深度-2 双缓冲）
        uint32_t uBar1Off; // VECOUT: pass0 u drain（奇数轮）
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
        __aicore__ inline Layout() : Layout(nullptr, 0) {}

        __aicore__ inline explicit Layout(const GruBlockCellTilingData* tdIn, uint32_t cluster = 0)
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
              sliceCores(1),
              rowBase(0),
              rowsThisCore(0),
              sliceLoCol(0),
              sliceHiCol(gbc::layout::C0F),
              mChunk(1),
              kc(gbc::layout::CUBE_BLOCK),
              cGroups(2),
              cGroupsX(1),
              kgX(gbc::layout::CUBE_BLOCK),
              kgH(gbc::layout::CUBE_BLOCK),
              rowsMax(1),
              planeElems(gbc::layout::C0F),
              aKOff(0),
              aHROff(0),
              bOff(0),
              biasSlotOff(0),
              aKElems(0),
              bElems(0),
              rBarOff(0),
              uBarOff(0),
              rBar1Off(0),
              uBar1Off(0),
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
            sliceCores = static_cast<uint32_t>(tdIn->sliceCores);
            if (sliceCores < 1u) {
                sliceCores = 1u; // 防御：旧 tiling 数据无该字段语义时退化为全列片域
            }
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
            // ---- per-core 分派（A1 splitMode=2：cluster → (rowChunk, sliceGrp) 行列 2D；
            // 旧路 rowChunk=cluster、列片域全幅——指令序与 A1 前逐位同域）----
            uint32_t rowChunk = cluster;
            if (splitMode == 2) {
                rowChunk = cluster / sliceCores;
                const uint32_t sliceGrp = cluster % sliceCores;
                SliceDispatch(sliceGrp, sliceLoCol, sliceHiCol);
            } else {
                sliceLoCol = 0;
                sliceHiCol = padHidden;
            }
            RowDispatch(rowChunk, rowBase, rowsThisCore);
            if (splitMode == 2 && rowBase >= batchSize) {
                rowsThisCore = 0; // 防御（host 契约 rowChunks≤B 下不可达）：空行块仅参与屏障
            }
            // ---- 纯算术推导（决策的确定函数）----
            rowsMax = gbc::layout::DrainRowsMax(mChunk, true);
            planeElems = rowsMax * nL0c;

            // L1 足迹（元素）：aK = K 组流式槽 [mChunk, kgMax]，每 (列片,组) 现搬 A
            // 切片，足迹与 I/H 解耦（B1 已删全幅 aH 回灌槽，mChunk 不再被 Hp 钉死）。
            const uint32_t kgMax = gbc::layout::MaxU(kgX, kgH);
            aKElems = gbc::layout::CeilDiv(kgMax, gbc::layout::C0F) *
                      gbc::layout::CeilAlign(mChunk, gbc::layout::CUBE_BLOCK) * gbc::layout::C0F;
            bElems = (nSlice / gbc::layout::C0F) * gbc::layout::CeilAlign(kc, gbc::layout::CUBE_BLOCK) *
                     gbc::layout::C0F;
            gbc::layout::Bump l1;
            // S2 门级预取：aK/b 各双槽（门 k 尾发射门 k+1 装载进异槽，1-ahead）。
            // S3' aHR 三槽独立于 aK：AIV 前瞻-3 写 hr[it+3]→aHR[it%3]，AIC 门 it 读
            // aHR[it%3]。⚠ 三槽+前瞻-3 为实测收敛结果，勿减为前瞻-2（跨核 L1 写
            // 可见性裕量不足，实测 c/h 大面积错；细节见存档）。
            aKOff = l1.TakeT<float>(aKElems * 2);  // A: x/hPrev 切片双槽 [mChunk, ≤kgMax]×2
            aHROff = l1.TakeT<float>(aKElems * 3); // A: pass1 hr 切片三槽（AIV 写 / AIC 读）
            bOff = l1.TakeT<float>(bElems * 2);    // B: 权重行块双槽 [kc, nSlice]×2
            // bias 列片共享槽：容量按 nL0c（末片超读落 L0C pad 列、不进 GM）。
            biasSlotOff = l1.TakeT<float>(gbc::cube::BtElems(nL0c));
            // L1 合计足迹 = l1.cur（host CheckLayoutCapacity 同公式复刻做前置拒绝）。

            // L0（单 slot，PIPE_ALL/WaitFixToM 序贯复用）：容量上界由 host 决策公式
            // 钉死 + UT 断言，kernel 侧不登记不校验的容量字段。

            // UB（两 AIV 同名偏移）：10 个 [rowsMax, nL0c] 列片平面（8 独立 + t1/t2，
            // t3≡rAcc）+ msk；平面行距 = 当前片宽 w（恒 8 对齐），AIV 扁平 work=stripe.count×w。
            // ⚠ 别名写读先序（均经既有 V2C/C2V/PIPE_ALL 栅栏闭合；改别名必须重核）：
            //   rBar0≡cBar1、uBar0≡cBar0：pass0 偶轮 merge ↔ pass1 c-merge（深度差 ≥2 轮
            //     + 块界 CubeWaitVec 后序）
            //   rBar1/uBar1：仅 pass0 奇轮（深度-2 的 V2C[gIdx−2] 闭合 WAR）
            //   uAcc≡cAcc：CopyOut u（MTE3+PIPE_ALL）→ pass1 首片 g0 DataCopy
            //   uComp≡cComp：FinFinalize(u) → pass1 首片 g1 Neumaier
            //   t1≡hp 副本、t2≡u 回读：V 读 → blend 前 MTE2 现拷（PIPE_ALL 间隔；t1 每片重拷）
            //   rAcc≡t3：pass0 σ(r)/r CopyOut → pass1 片末 Tanh s3/VF h 落点/h CopyOut
            //     （WaitMte3ToV 闭合）；⚠ h 不得改回 cAcc 就地（Fix B 零窗竞态）
            //   msk：FinFinalize/Tanh 按需 scratch，无跨相位保持
            gbc::layout::Bump ub;
            rBarOff = ub.TakeT<float>(planeElems);  // VECOUT: pass0 r drain 偶轮；≡ pass1 cBar1
            uBarOff = ub.TakeT<float>(planeElems);  // VECOUT: pass0 u drain 偶轮；≡ pass1 cBar0
            rBar1Off = ub.TakeT<float>(planeElems); // VECOUT: pass0 r drain 奇轮（S3'' 深度-2）
            uBar1Off = ub.TakeT<float>(planeElems); // VECOUT: pass0 u drain 奇轮（pass1 空闲）
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

    // ---- 一个 m-chunk 的两段（cube 侧 / vector 侧）。phase：PHASE_BOTH=pass0+pass1
    // （旧路，指令序与 A1 前逐位一致）；P0/P1=A1 splitMode=2 相位拆分单段（P0 末枚 V2C
    // 留 pending，P1 块首 CubeWaitVec 消费——与旧路 pass0→pass1 界同 flag 同 WAR 语义）。
    // SyncAll 全局屏障在 Half 层，每核恰一次（死锁契约见 CubeHalf ⚠）。
    static constexpr uint32_t PHASE_BOTH = 0;
    static constexpr uint32_t PHASE_P0 = 1;
    static constexpr uint32_t PHASE_P1 = 2;
    __aicore__ inline void CubeChunk(uint32_t rowBase, uint32_t mNow, bool firstChunk, uint32_t phase);
    __aicore__ inline void VectorChunk(uint32_t rowBase, uint32_t mNow, uint32_t phase);

    // ---- S2 门级装载预取通路（nSlice ≥ nL0c ⇒ 每门单 n0 片时启用；否则走
    // GateGemmRange 串行旧路，两路 Mmad 序列逐位一致）----
    // 组参数包（一个 (列片 s, 组 g) 的全部派生量；PlanGroup 纯算术，无副作用）
    struct GrpPlan {
        uint32_t s;      // 列片基列
        uint32_t w;      // drain/平面宽 = min(nL0c, Hp−s)
        uint32_t frameN; // L0C 帧宽 = min(nL0c, nAl−s)（= 旧路 sliceN）
        uint32_t k0;     // K 组起点（段内）
        uint32_t kw;     // K 组宽
        bool isX;        // x 段（否则 h 段）
        bool valid;      // s/g 越界 ⇒ false（末组无 next）
    };
    __aicore__ inline GrpPlan PlanGroup(uint32_t s, uint32_t g) const;
    __aicore__ inline GrpPlan PlanNext(uint32_t s, uint32_t g) const;
    // 异步发射一门的重载（MTE2，无等待；槽 = 线性门序奇偶）
    __aicore__ inline void IssueWeightTile(const AscendC::GlobalTensor<float>& wGM, uint32_t wRow, uint32_t wCol,
                                           uint32_t gw, uint32_t kcNow, uint32_t ns, uint32_t bSlot) const;
    __aicore__ inline void IssueATile(const AscendC::GlobalTensor<float>& aGM, uint32_t rowBase, uint32_t k0,
                                      uint32_t mNow, uint32_t kw, uint32_t kLim, uint32_t aSlot) const;
    // 消费一门的前半：等装载落 L1 → 切 L0（A 可选，aL1Off 为 aK/aHR 槽绝对偏移）→
    // BT（可选）→ MTE1 排空（返回后槽对 MTE2 自由，调用点随即发射下一门装载）
    __aicore__ inline void GatePrepare(uint32_t aL1Off, bool doSplitA, uint32_t a1Cols, uint32_t kcNow, uint32_t sliceN,
                                       bool withBias, uint32_t bL1Off, uint32_t mNow) const;
    __aicore__ inline void CubeChunkPf(uint32_t rowBase, uint32_t mNow, bool firstChunk, uint32_t phase);
    // AIV：从 GM 的 hPrev+r 现场重算 hr[gN] 切片（t1/t2 为 scratch，hr 落 t1）→
    // FeedbackToL1 进 aHR 槽（aHRSlotOff）。S3' 前瞻-2 / !pf 前瞻-1 共用。
    __aicore__ inline void RecomputeHr(uint32_t gN, uint32_t aHRSlotOff, const gbc::layout::RowStripe& stripe,
                                       uint32_t gRow, uint32_t mNow, const AscendC::LocalTensor<float>& t1,
                                       const AscendC::LocalTensor<float>& t2) const;
    // S3'-T2（pf2 轮首回灌）：RecomputeHr 拆两段——StageHr 算 hr[gN] 暂存 t3
    // （t1/t2 为 scratch）；FeedbackHr 自 t3 回灌 aHR 槽。两段跨相邻轮（轮 r 尾
    // stage → 轮 r+1 首 feedback），使 feedback 落在 AIC 的 L1 静默窗（drain[r−1]
    // → 下一门 L1 访问之间）——L1 无写仲裁，见 pass1 AIC 段 ⚠。
    __aicore__ inline void StageHr(uint32_t gN, const gbc::layout::RowStripe& stripe, uint32_t gRow,
                                   const AscendC::LocalTensor<float>& t1, const AscendC::LocalTensor<float>& t2,
                                   const AscendC::LocalTensor<float>& t3) const;
    __aicore__ inline void FeedbackHr(uint32_t gN, uint32_t aHRSlotOff, const gbc::layout::RowStripe& stripe,
                                      uint32_t mNow, const AscendC::LocalTensor<float>& t3) const;

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
    // ⚠ 950 核号约定：AIC 的 GetBlockIdx() 即 cluster 号；AIV 的是扁平 subcore 号
    // （÷ 核比得 cluster）。per-core 分派（A1）在 Layout 构造内完成。
    uint32_t cluster;
    if ASCEND_IS_AIC {
        cluster = static_cast<uint32_t>(AscendC::GetBlockIdx());
    } else {
        cluster = static_cast<uint32_t>(AscendC::GetBlockIdx()) / static_cast<uint32_t>(AscendC::GetTaskRatio());
    }
    layout_ = Layout(td, cluster);
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
// S2 门级装载预取（pingpong_design 的 L1 双缓冲 + Set/Wait 分离精神的单事件形态）。
// 结构：门 k 的 GatePrepare 尾（WaitMte1ToM 已排空 MTE1 ⇒ aK/b 槽对 MTE2 自由）
// 发射门 k+1 的 A/b 装载（异槽），随后 Mmad[k]→drain[k]→C2V→AIV merge→V2C 的
// 整段空窗与装载重叠；门 k+1 顶的 WaitMte2ToMte1 恰好只覆盖这批装载（发射点与
// 消费点之间无其他 MTE2——bias 现搬均在 wait 前发射，被同一 wait 覆盖）。
// 槽位：aK slot = 线性组序 gIdx&1（A 每组一片，r/u 两门共用）；b slot = 线性门序
// &1（pass0：r 门恒 0 / u 门恒 1；pass1：gIdx&1）。跨核结构零改动（C2V/V2C 轮次、
// 深度与 B1 完全一致），Mmad 序列不变 ⇒ 逐位不变。
// ---------------------------------------------------------------------------
__aicore__ inline GruBlockCellKernel::GrpPlan GruBlockCellKernel::PlanGroup(uint32_t s, uint32_t g) const
{
    const Layout& lyt = layout_;
    GrpPlan p;
    // A1：有效性上界 = 本核 sliceHiCol（旧路==padHidden）——越域卷绕 ⇒ 预取自然截止。
    p.valid = (s < lyt.sliceHiCol) && (g < lyt.cGroups);
    if (!p.valid) {
        p.s = s;
        p.w = 0;
        p.frameN = 0;
        p.k0 = 0;
        p.kw = 0;
        p.isX = false;
        return p;
    }
    p.s = s;
    p.w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
    p.frameN = gbc::layout::MinU(lyt.nL0c, lyt.nAl - s);
    p.isX = (g < lyt.cGroupsX);
    const uint32_t kg = p.isX ? lyt.kgX : lyt.kgH;
    const uint32_t kLim = p.isX ? lyt.inputSize : lyt.hiddenSize;
    p.k0 = p.isX ? (g * lyt.kgX) : ((g - lyt.cGroupsX) * lyt.kgH);
    p.kw = gbc::layout::MinU(kLim - p.k0, kg);
    return p;
}

__aicore__ inline GruBlockCellKernel::GrpPlan GruBlockCellKernel::PlanNext(uint32_t s, uint32_t g) const
{
    return (g + 1 < layout_.cGroups) ? PlanGroup(s, g + 1) : PlanGroup(s + layout_.nL0c, 0);
}

__aicore__ inline void GruBlockCellKernel::IssueWeightTile(const AscendC::GlobalTensor<float>& wGM, uint32_t wRow,
                                                           uint32_t wCol, uint32_t gw, uint32_t kcNow, uint32_t ns,
                                                           uint32_t bSlot) const
{
    const Layout& lyt = layout_;
    const uint32_t rowAvail = (gw > wCol) ? (gw - wCol) : 0;
    const uint32_t copyCols = gbc::layout::MinU(ns, rowAvail);
    if (copyCols > 0) {
        gbc::cube::CopyInNd2Nz(wGM, wRow, wCol, kcNow, copyCols, gw,
                               lyt.bOff + bSlot * lyt.bElems * static_cast<uint32_t>(sizeof(float)));
    }
}

__aicore__ inline void GruBlockCellKernel::IssueATile(const AscendC::GlobalTensor<float>& aGM, uint32_t rowBase,
                                                      uint32_t k0, uint32_t mNow, uint32_t kw, uint32_t kLim,
                                                      uint32_t aSlot) const
{
    const Layout& lyt = layout_;
    gbc::cube::CopyInNd2Nz(aGM, rowBase, k0, mNow, kw, kLim,
                           lyt.aKOff + aSlot * lyt.aKElems * static_cast<uint32_t>(sizeof(float)));
}

__aicore__ inline void GruBlockCellKernel::GatePrepare(uint32_t aL1Off, bool doSplitA, uint32_t a1Cols, uint32_t kcNow,
                                                       uint32_t sliceN, bool withBias, uint32_t bL1Off,
                                                       uint32_t mNow) const
{
    const Layout& lyt = layout_;
    gbc::sync::WaitMte2ToMte1(); // 覆盖：本门 A/b 预取 + 组首 bias 现搬（发射点均在本 wait 前）
    if (doSplitA) {
        gbc::cube::SplitA(aL1Off, a1Cols, 0, kcNow, mNow);
    }
    gbc::cube::SplitB(bL1Off, kcNow, sliceN);
    if (withBias) {
        gbc::cube::LoadBiasToBT(lyt.biasSlotOff, 0, sliceN);
    }
    gbc::sync::WaitMte1ToM();    // L0 就绪（M 管屏障：Mmad 排在 SplitA/B/BT 之后）
    gbc::sync::WaitMte1ToMte2(); // MTE2 管屏障：调用点随后发射的下一门装载排在 SplitA/B/BT
                                 // 对本门 aK/b/BT 槽的读之后（⚠ WaitFlag 只屏障目标管、
                                 // 不阻标量流——缺本事件则预取 MTE2 与 SplitB 竞态覆槽，
                                 // 实测 r/u/c/h 大面积"接近但错"的 stale 数据）
}

// ---------------------------------------------------------------------------
// CubeChunkPf — S2 预取版 CubeChunk（pf 域 = nSlice ≥ nL0c）。与旧路逐门同构
// （同 Mmad 序/drain/C2V-V2C 轮次），仅装载发射前移一门 ⇒ 输出逐位一致。
// 1-ahead：门尾发射下一门 b/A 进异槽；pass1 h 段 A=hr 由 AIV 写（按段互斥）。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::CubeChunkPf(uint32_t rowBase, uint32_t mNow, bool firstChunk, uint32_t phase)
{
    const Layout& lyt = layout_;
    const uint32_t twoH = static_cast<uint32_t>(GRU_GATE_NUM) * lyt.hiddenSize;
    const uint32_t aKBytes = lyt.aKElems * static_cast<uint32_t>(sizeof(float)); // aK/aHR 槽距
    const uint32_t bBytes = lyt.bElems * static_cast<uint32_t>(sizeof(float));   // b 槽距
    // pf2 = pass1 深度-2 使能（cGroupsX≥3）：奇偶 need 分级 + feedback 落轮首 L1 静默窗。
    // ⚠ 勿改纯深度-2（奇偶一律 it−2）——L1 无写仲裁，重叠期并发访问实测大面积错（见存档）。
    const bool pf2 = (lyt.cGroupsX >= 3u);
    // A1 列片域（旧路全幅）+ 本域片数（V2C 配平口径）
    const uint32_t nTilesLocal = gbc::layout::CeilDiv(lyt.sliceHiCol - lyt.sliceLoCol, lyt.nL0c);

    if (phase != PHASE_P1) {
        if (!firstChunk) {
            gbc::sync::CubeWaitVec(); // 前一块 epilogue 完成（rBar/uBar 可覆写）
        }

        // ---- pass0 prologue：门 0（组 0 r 门）的 A/b 装载 ----
        {
            const GrpPlan p0 = PlanGroup(lyt.sliceLoCol, 0);
            IssueATile(p0.isX ? gmX_ : gmHPrev_, rowBase, p0.k0, mNow, p0.kw, p0.isX ? lyt.inputSize : lyt.hiddenSize,
                       0);
            // r 门权重列基 = 本片域首列 p0.s（旧路 p0.s==0 逐位同值）
            IssueWeightTile(gmWRu_, p0.isX ? p0.k0 : (lyt.inputSize + p0.k0), p0.s, twoH, p0.kw, p0.frameN, 0);
        }

        // ---- pass0 · r/u 两门（1-ahead 预取；S3'' 深度-2）----
        // 事件链（⚠ WaitFlag 只屏障目标管、不阻标量流——发射点安全全靠定向事件对）：
        //   GatePrepare → 发射下一门装载(MTE2 ∥ 本门 Mmad) → Mmad → drain(FIX) → C2V。
        // 深度-2 依据：pass0 的 V2C 为 WAR-only 语义（无跨核数据交接），stale wait 仍安全。
        // V2C 配平：环内消费至 gIdx−1，尾环至 n0Total−1，末枚留给 pass0→pass1 界。
        uint32_t consumed0 = 0;
        uint32_t sIdx = 0; // 本核列片域内局部片序（gIdx/槽轮换与 AIV 同源）
        for (uint32_t s = lyt.sliceLoCol; s < lyt.sliceHiCol; s += lyt.nL0c, ++sIdx) {
            for (uint32_t g = 0; g < lyt.cGroups; ++g) {
                const GrpPlan cur = PlanGroup(s, g);
                const GrpPlan nxt = PlanNext(s, g);
                const uint32_t gIdx = sIdx * lyt.cGroups + g;
                const uint32_t aSlot = gIdx & 1U;
                const uint32_t need0 = (gIdx >= 2) ? (gIdx - 1) : 0u; // 深度-2：消费至 V2C[gIdx−2]
                while (consumed0 < need0) {
                    gbc::sync::CubeWaitVec(); // AIV 已消费轮 gIdx−2 的 rBar/uBar
                    ++consumed0;
                }
                if (gIdx > 0) {
                    gbc::sync::WaitFixToM(); // 上一轮 drain 排空后方可覆写 L0C
                }
                // -- r 门（bias 仅组首现搬，先于本门 wait 发射、被同一 wait 覆盖）--
                if (g == 0 && s < lyt.hiddenSize) {
                    gbc::cube::CopyInBiasPad(gmBRu_, s, twoH, gbc::layout::MinU(lyt.hiddenSize - s, cur.w),
                                             lyt.biasSlotOff);
                }
                GatePrepare(lyt.aKOff + aSlot * aKBytes, true, cur.kw, cur.kw, cur.frameN, g == 0, lyt.bOff, mNow);
                IssueWeightTile(gmWRu_, cur.isX ? cur.k0 : (lyt.inputSize + cur.k0), lyt.hiddenSize + s, twoH, cur.kw,
                                cur.frameN, 1); // u 门 b（槽 1）——与 r Mmad/drain/握手重叠
                gbc::cube::MmadGate(mNow, cur.frameN, 0, cur.frameN, cur.kw, (g == 0) ? 0 : 1, true);
                gbc::sync::WaitMToMte1(); // 下门 SplitB/BT 复用 L0B/BT 排在 r Mmad 读之后
                gbc::sync::WaitMToFix();
                gbc::cube::DrainToUB((aSlot != 0u) ? lyt.rBar1Off : lyt.rBarOff, 0, cur.w, mNow, cur.frameN, cur.w);
                gbc::sync::WaitFixToM(); // L0C 复用于 u 门
                if (g == 0 && s < lyt.hiddenSize) {
                    gbc::cube::CopyInBiasPad(gmBRu_, lyt.hiddenSize + s, twoH,
                                             gbc::layout::MinU(lyt.hiddenSize - s, cur.w), lyt.biasSlotOff);
                }
                // -- u 门（L0A 仍持 A[g]，免 SplitA；r 门尾的 M→MTE1 已序化 L0B/BT 复用）--
                GatePrepare(lyt.aKOff + aSlot * aKBytes, false, cur.kw, cur.kw, cur.frameN, g == 0, lyt.bOff + bBytes,
                            mNow);
                if (nxt.valid) { // 下一组 r 门 b（槽 0）+ 下一组 A（aK 异槽）
                    IssueWeightTile(gmWRu_, nxt.isX ? nxt.k0 : (lyt.inputSize + nxt.k0), nxt.s, twoH, nxt.kw,
                                    nxt.frameN, 0);
                    IssueATile(nxt.isX ? gmX_ : gmHPrev_, rowBase, nxt.k0, mNow, nxt.kw,
                               nxt.isX ? lyt.inputSize : lyt.hiddenSize, (gIdx + 1) & 1U);
                }
                gbc::cube::MmadGate(mNow, cur.frameN, 0, cur.frameN, cur.kw, (g == 0) ? 0 : 1, true);
                gbc::sync::WaitMToMte1(); // 下一组 r 门 SplitA/B 复用 L0 排在 u Mmad 读之后
                gbc::sync::WaitMToFix();
                gbc::cube::DrainToUB((aSlot != 0u) ? lyt.uBar1Off : lyt.uBarOff, 0, cur.w, mNow, cur.frameN, cur.w);
                gbc::sync::CubeSignalVec(); // 本轮 r/u 部分和已落 UB（列片 s）
            }
        }
        // 尾环：消费至 n0Total−1 枚（末枚留给 pass0→pass1 界）
        const uint32_t n0Total = nTilesLocal * lyt.cGroups;
        while (consumed0 + 1 < n0Total) {
            gbc::sync::CubeWaitVec();
            ++consumed0;
        }
    } // phase != PHASE_P1（pass0 段）

    if (phase != PHASE_P0) {
        // ---- pass0 末轮 V2C 收尾（PHASE_BOTH）/ pending V2C 消费（PHASE_P1：pass0 相位
        // 末块 ∨ 前一 pass1 块的末枚——同 flag 同 WAR 语义，屏障已在 Half 层走过）----
        gbc::sync::CubeWaitVec();

        // ---- pass1 prologue：门 0/1 的 A/b 装载（2-ahead 冷启动需前两门）----
        {
            const GrpPlan p0 = PlanGroup(lyt.sliceLoCol, 0);
            IssueATile(gmX_, rowBase, p0.k0, mNow, p0.kw, lyt.inputSize, 0); // x[0]→aK0
            IssueWeightTile(gmWC_, p0.k0, p0.s, lyt.hiddenSize, p0.kw, p0.frameN, 0);
            const GrpPlan p1 = PlanNext(lyt.sliceLoCol, 0);
            if (p1.valid) { // b[1]→b1；x 段才发 A（h 段的 A=hr 由 AIV 特例搭载写入）
                IssueWeightTile(gmWC_, p1.isX ? p1.k0 : (lyt.inputSize + p1.k0), p1.s, lyt.hiddenSize, p1.kw, p1.frameN,
                                1);
                if (p1.isX) {
                    IssueATile(gmX_, rowBase, p1.k0, mNow, p1.kw, lyt.inputSize, 1);
                }
            }
        }

        // ---- pass1 · c 门（S3' 深度-2 + 前瞻-3 hr + 2-ahead 预取）----
        // cBar 双缓冲（cBar0≡uBarOff、cBar1≡rBarOff，零 UB 增量）；h 段 A=hr[it] 由 AIV
        // 前瞻-3 写 aHR[it%3]（⚠ 前瞻勿减为 2——跨核 L1 写可见性裕量不足，实测翻车）。
        // 特例 cGroupsX≤2：g<3 的 h 门由 AIV 在 merge[(s,0)] 一并写入，AIC 门 (s,1) need=it。
        // V2C 配平：环内消费至 it−1，尾环至 total−1，末枚留给块界（跨块 FIFO 零残留）。
        uint32_t consumed = 0;
        uint32_t sIdx = 0; // 本核列片域内局部片序（it 与 AIV 同源；A1 前全幅域时逐位相同）
        for (uint32_t s = lyt.sliceLoCol; s < lyt.sliceHiCol; s += lyt.nL0c, ++sIdx) {
            for (uint32_t g = 0; g < lyt.cGroups; ++g) {
                const uint32_t it = sIdx * lyt.cGroups + g;
                const GrpPlan cur = PlanGroup(s, g);
                const uint32_t slot = it & 1U;
                // pf2 奇偶分级：奇门 need=it（消费至 V2C[it−1]——轮 it−1 完整排空，含其
                // 轮首 feedback ⇒ 奇门的全部 L1 访问结构性晚于最近一轮 feedback）；偶门
                // need=it−1（消费至 V2C[it−2]，与轮 it−1 重叠——hr[it] 由轮 it−2 轮首
                // feedback 携带于 V2C[it−2]，cBar WAR 同界）。!pf2：深度-1；特例 cGroupsX≤1：
                // 门 (s,1) 的 hr 由 V2C[(s,0)] 搭载，need 提升为 it。
                uint32_t need = pf2 ? (((it & 1U) != 0u) ? it : ((it >= 2) ? (it - 1) : 0u)) : ((it >= 1) ? it : 0u);
                if (!pf2 && lyt.cGroupsX <= 1 && g == 1) {
                    need = it;
                }
                while (consumed < need) {
                    gbc::sync::CubeWaitVec();
                    ++consumed;
                }
                if (it > 0) {
                    gbc::sync::WaitFixToM(); // L0C WAR：Mmad[it] 排在 drain[it−1] 之后
                }
                if (g == 0 && s < lyt.hiddenSize) {
                    gbc::cube::CopyInBiasPad(gmBC_, s, lyt.hiddenSize, gbc::layout::MinU(lyt.hiddenSize - s, cur.w),
                                             lyt.biasSlotOff);
                }
                const uint32_t aOff = cur.isX ? (lyt.aKOff + slot * aKBytes) : (lyt.aHROff + (it % 3u) * aKBytes);
                GatePrepare(aOff, true, cur.kw, cur.kw, cur.frameN, g == 0, lyt.bOff + slot * bBytes, mNow);
                // 2-ahead 发射：b[it+2]→b[it&1]、x[it+2]→aK[it&1]（仅 x 段；h 段 aHR 由
                // AIV 写）。跨列片按线性门序还原域内片基 + sliceLoCol，越域由 sliceHiCol 截止。
                // ⚠ pf2 下本发射窗与 merge[it−1] 轮首 FeedbackToL1 结构性错开（L1 无写仲裁）。
                const uint32_t it2 = it + 2;
                const GrpPlan nxt2 = PlanGroup(lyt.sliceLoCol + (it2 / lyt.cGroups) * lyt.nL0c, it2 % lyt.cGroups);
                if (nxt2.valid) {
                    IssueWeightTile(gmWC_, nxt2.isX ? nxt2.k0 : (lyt.inputSize + nxt2.k0), nxt2.s, lyt.hiddenSize,
                                    nxt2.kw, nxt2.frameN, it2 & 1U);
                    if (nxt2.isX) {
                        IssueATile(gmX_, rowBase, nxt2.k0, mNow, nxt2.kw, lyt.inputSize, it2 & 1U);
                    }
                }
                gbc::cube::MmadGate(mNow, cur.frameN, 0, cur.frameN, cur.kw, (g == 0) ? 0 : 1, true);
                gbc::sync::WaitMToMte1(); // 下一门 SplitA/B 复用 L0 排在本门 Mmad 读之后
                gbc::sync::WaitMToFix();
                gbc::cube::DrainToUB((slot != 0u) ? lyt.rBarOff : lyt.cBarOff, 0, cur.w, mNow, cur.frameN, cur.w);
                gbc::sync::CubeSignalVec(); // C2V[it]：本轮部分和已落 cBar[it&1]
            }
        }
        // 尾环：消费至 total−1 枚（末枚留给块界）
        const uint32_t n1Total = nTilesLocal * lyt.cGroups;
        while (consumed + 1 < n1Total) {
            gbc::sync::CubeWaitVec();
            ++consumed;
        }
    } // phase != PHASE_P0（pass1 段）
}

// ---------------------------------------------------------------------------
// 列片 ↔ GM 的行跨步搬运（S1 双路径，逐位不变）：UB 侧 8 对齐用单条 blockCount=rows
// 批量 DataCopyPad，否则逐行回退（任意 H 成立）。⚠ stride 单位非对称：GM 侧=1 字节、
// UB 侧=32B（dav_c310 实现语义，且 UB 行距向上圆整 32B ⇒ blockLen 须 32B 整除）；
// 批量条件只看 UB 侧。⚠ 勿用 stride=0 单条整块搬（行距会取 blockLen，第 1 行起错位）。
// ubPitch = UB 平面行距（片宽 w 或 hr 切片 kwP）；gmPitch = hiddenSize。
// ---------------------------------------------------------------------------
__aicore__ inline bool TileCopyBatchable(uint32_t rows, uint32_t ubPitch, uint32_t tileCols)
{
    return rows > 0 && (tileCols % gbc::layout::C0F) == 0 && ((ubPitch - tileCols) % gbc::layout::C0F) == 0;
}

__aicore__ inline void CopyTileToGm(const AscendC::GlobalTensor<float>& gm, uint64_t gmElemOff,
                                    const AscendC::LocalTensor<float>& ub, uint32_t rows, uint32_t ubPitch,
                                    uint32_t gmPitch, uint32_t tileCols)
{
    AscendC::DataCopyExtParams e;
    e.blockCount = 1;
    e.blockLen = tileCols * sizeof(float);
    e.srcStride = 0;
    e.dstStride = 0;
    if (TileCopyBatchable(rows, ubPitch, tileCols)) {
        e.blockCount = static_cast<uint16_t>(rows);
        e.srcStride = (ubPitch - tileCols) / gbc::layout::C0F; // UB 侧：32B 单位
        e.dstStride = (gmPitch - tileCols) * sizeof(float);    // GM 侧：字节单位
        AscendC::DataCopyPad(gm[gmElemOff], ub, e);
        return;
    }
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
    if (TileCopyBatchable(rows, ubPitch, tileCols)) {
        e.blockCount = static_cast<uint16_t>(rows);
        e.srcStride = (gmPitch - tileCols) * sizeof(float);    // GM 侧：字节单位
        e.dstStride = (ubPitch - tileCols) / gbc::layout::C0F; // UB 侧：32B 单位
        AscendC::DataCopyPad(ub, gm[gmElemOff], e, pad);
        return;
    }
    for (uint32_t rr = 0; rr < rows; ++rr) {
        AscendC::DataCopyPad(ub[rr * ubPitch], gm[gmElemOff + static_cast<uint64_t>(rr) * gmPitch], e, pad);
    }
}

// ---------------------------------------------------------------------------
// RecomputeHr — AIV：hr[gN] = hPrev[:,gN] ⊙ r[:,gN] 现场重算 → aHR 槽（!pf2/!pf 路）。
// t1/t2 为 scratch（hr 落 t1）；尾组 kwP=CeilAlign(kw,8) 零填充（pad 列不贡献）。
// 事件链全定向，收尾由调用点 VecSignalCube（PIPE_MTE3）跨核发布。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::RecomputeHr(uint32_t gN, uint32_t aHRSlotOff,
                                                       const gbc::layout::RowStripe& stripe, uint32_t gRow,
                                                       uint32_t mNow, const AscendC::LocalTensor<float>& t1,
                                                       const AscendC::LocalTensor<float>& t2) const
{
    const Layout& lyt = layout_;
    const uint32_t k0n = (gN - lyt.cGroupsX) * lyt.kgH;
    const uint32_t kwn = gbc::layout::MinU(lyt.hiddenSize - k0n, lyt.kgH);
    const uint32_t kwPn = gbc::layout::CeilAlign(kwn, gbc::layout::C0F);
    const uint32_t workHr = stripe.count * kwPn;
    const uint64_t gmOffHr = static_cast<uint64_t>(gRow) * lyt.hiddenSize + k0n;
    // ⚠ MTE3→V 闭合：!pf2&&pf 特例轮同轮连调本函数 2~3 次，开头 Duplicate(t1)（V 写）
    // 与上一次 FeedbackToL1 的 MTE3 读 t1 零窗相邻（实测 hr 近全零落槽）；跨轮调用已由
    // 轮首 PIPE_ALL 覆盖，本事件即刻通过。t2 无 MTE3 读者。
    gbc::sync::WaitMte3ToV();
    AscendC::Duplicate(t1, 0.0f, workHr); // hp 平面清零（含 pad 列）
    AscendC::Duplicate(t2, 0.0f, workHr); // r 平面清零
    gbc::sync::WaitVToMte2();             // Duplicate（V 写 t1/t2）→ MTE2 复写
    CopyTileFromGm(gmHPrev_, gmOffHr, t1, stripe.count, kwPn, lyt.hiddenSize, kwn);
    CopyTileFromGm(gmR_, gmOffHr, t2, stripe.count, kwPn, lyt.hiddenSize, kwn);
    gbc::sync::WaitMte2ToV();         // MTE2 写 t1/t2 → V 读（Mul）
    AscendC::Mul(t1, t1, t2, workHr); // hr = hPrev ⊙ r（pad 列 0⊙0=0，落 t1）
    gbc::sync::WaitVToMte3();         // V 写 hr → MTE3（FeedbackToL1）读
    const AscendC::LocalTensor<float> aHR(AscendC::TPosition::A1, aHRSlotOff, lyt.aKElems);
    gbc::vector::FeedbackToL1(aHR, t1,
                              gbc::layout::NzLayout(mNow, gbc::layout::MaxU(lyt.kgX, lyt.kgH), gbc::layout::C0F, mNow),
                              kwPn, 0, stripe);
}

// ---------------------------------------------------------------------------
// StageHr — pf2 轮尾：hr[gN] → **t3 暂存**（不回灌）。事件链 V(清零)→MTE2→V(Mul)；
// t3 跨轮存续，下一轮轮首 FeedbackHr 的 MTE3 读经轮首 PIPE_ALL 闭合。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::StageHr(uint32_t gN, const gbc::layout::RowStripe& stripe, uint32_t gRow,
                                                   const AscendC::LocalTensor<float>& t1,
                                                   const AscendC::LocalTensor<float>& t2,
                                                   const AscendC::LocalTensor<float>& t3) const
{
    const Layout& lyt = layout_;
    const uint32_t k0n = (gN - lyt.cGroupsX) * lyt.kgH;
    const uint32_t kwn = gbc::layout::MinU(lyt.hiddenSize - k0n, lyt.kgH);
    const uint32_t kwPn = gbc::layout::CeilAlign(kwn, gbc::layout::C0F);
    const uint32_t workHr = stripe.count * kwPn;
    const uint64_t gmOffHr = static_cast<uint64_t>(gRow) * lyt.hiddenSize + k0n;
    AscendC::Duplicate(t1, 0.0f, workHr); // hp 平面清零（含 pad 列）
    AscendC::Duplicate(t2, 0.0f, workHr); // r 平面清零
    gbc::sync::WaitVToMte2();             // Duplicate（V 写 t1/t2）→ MTE2 复写
    CopyTileFromGm(gmHPrev_, gmOffHr, t1, stripe.count, kwPn, lyt.hiddenSize, kwn);
    CopyTileFromGm(gmR_, gmOffHr, t2, stripe.count, kwPn, lyt.hiddenSize, kwn);
    gbc::sync::WaitMte2ToV();         // MTE2 写 t1/t2 → V 读（Mul）
    AscendC::Mul(t3, t1, t2, workHr); // hr = hPrev ⊙ r → t3 暂存（pad 列 0）
}

// ---------------------------------------------------------------------------
// FeedbackHr — pf2 轮首：t3 暂存的 hr[gN] 回灌 aHR 槽（落 AIC 的 L1 静默窗）。
// 收尾由 VecSignalCube 的 PIPE_MTE3 管序承载（V2C[it] 蕴含本轮回灌已排空）。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::FeedbackHr(uint32_t gN, uint32_t aHRSlotOff,
                                                      const gbc::layout::RowStripe& stripe, uint32_t mNow,
                                                      const AscendC::LocalTensor<float>& t3) const
{
    const Layout& lyt = layout_;
    const uint32_t k0n = (gN - lyt.cGroupsX) * lyt.kgH;
    const uint32_t kwn = gbc::layout::MinU(lyt.hiddenSize - k0n, lyt.kgH);
    const uint32_t kwPn = gbc::layout::CeilAlign(kwn, gbc::layout::C0F);
    const AscendC::LocalTensor<float> aHR(AscendC::TPosition::A1, aHRSlotOff, lyt.aKElems);
    gbc::vector::FeedbackToL1(aHR, t3,
                              gbc::layout::NzLayout(mNow, gbc::layout::MaxU(lyt.kgX, lyt.kgH), gbc::layout::C0F, mNow),
                              kwPn, 0, stripe);
}

// ---------------------------------------------------------------------------
// CubeChunk — 一个 m-chunk 的 cube 侧：pass0（r/u 两门）→ CubeWaitVec → pass1（c 门）。
// 每 (列片,组) 一轮 C2V/V2C 握手；前一块收尾 V2C 由块顶 CubeWaitVec 消费。
// phase 语义见类声明处（PHASE_BOTH 旧路 / P0/P1 A1 相位拆分）。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::CubeChunk(uint32_t rowBase, uint32_t mNow, bool firstChunk, uint32_t phase)
{
    const Layout& lyt = layout_;
    // 片上张量由 gbc::cube:: Te 助手按 Layout 槽偏移现构，不预建 LocalTensor 视图。
    if (lyt.nSlice >= lyt.nL0c) {
        CubeChunkPf(rowBase, mNow, firstChunk, phase); // S2 预取路（每门单 n0 片域；两侧同判据）
        return;
    }

    if (phase != PHASE_P1) {
        if (!firstChunk) {
            gbc::sync::CubeWaitVec(); // 前一块 epilogue 完成（rBar/uBar 可覆写）
        }

        // ---- pass0 · r/u 两门（旧路串行）：每组 A 现搬 aK → r 门 Mmad/drain → u 门
        // Mmad/drain → C2V（同轮交付两组部分和）。bias 仅首组种子、按列片现搬共享槽。
        for (uint32_t s = lyt.sliceLoCol; s < lyt.sliceHiCol; s += lyt.nL0c) {
            const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
            const uint32_t frameN = gbc::layout::MinU(lyt.nL0c, lyt.nAl - s); // L0C 帧宽（Mmad cParent 同源）
            for (uint32_t g = 0; g < lyt.cGroups; ++g) {
                if (s > lyt.sliceLoCol || g > 0) {
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
    } // phase != PHASE_P1（pass0 段；末枚 V2C 留 pending，口径同 CubeChunkPf）

    if (phase != PHASE_P0) {
        // ---- pass0 末轮 V2C 收尾（B1：pass0 不再回灌 aH，此处仅消费末列片末组的 V2C
        // 使 pass0 的 V2C 轮次零残留；CubeWaitVec 尾随的 PIPE_ALL 亦排空 pass0 末轮 drain
        // 的 FIX，交还 L0C 给 pass1）。pass1 的 h 段 hr 由 AIV 逐组现场重算（见下）。
        // A1 相位拆分（PHASE_P1）：同一枚 wait 消费 pending V2C（pass0 相位末块 ∨ 前一
        // pass1 块），全局屏障已在 Half 层的 SyncAll 走过。----
        gbc::sync::CubeWaitVec();

        // ---- pass1 · c 门（旧路串行）：x 段现搬 aK；h 段读 AIV 回灌的 aHR（B1）。
        // 每组 Mmad → drain cBar → C2V；轮间 CubeWaitVec = cBar WAR + hr 就绪蕴含。----
        for (uint32_t s = lyt.sliceLoCol; s < lyt.sliceHiCol; s += lyt.nL0c) {
            const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
            const uint32_t frameN = gbc::layout::MinU(lyt.nL0c, lyt.nAl - s);
            for (uint32_t g = 0; g < lyt.cGroups; ++g) {
                if (s > lyt.sliceLoCol || g > 0) {
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
                    // B1：h 段 A = AIV 逐组重算进 aHR 槽 0 的 hr 切片（!pf 恒槽 0，AIV 同
                    // 判据同槽）；组顶 CubeWaitVec 蕴含回灌完成。本支无 bias（seedMode 恒 Plain）。
                    GateGemmRange(lyt.aHROff, kw, 0, kw, gmWC_, lyt.inputSize + k0, 0, lyt.hiddenSize, 1, 0u, s, mNow);
                }
                gbc::sync::WaitMToFix();
                gbc::cube::DrainToUB(lyt.cBarOff, 0, w, mNow, frameN, w);
                gbc::sync::CubeSignalVec(); // 本轮部分和已落 UB（列片 s）
            }
        }
    } // phase != PHASE_P0（pass1 段）
}

// ---------------------------------------------------------------------------
// CubeHalf — 本 cluster 的全部 m-chunk。旧路（splitMode 0/1）：逐块 pass0+pass1，无跨核
// 栅栏。A1（splitMode=2）：相位拆分——全部 pass0 块 → SyncAll<false>() → 全部 pass1 块
// （pass1 h 段跨核读 GM 全 H 列 r；屏障 = AIV MTE3 管序 + AIC FFTS 全局汇聚）。
// ⚠ 死锁契约：blockDim 内全 core 到场、每核恰调用一次——host 保证无空转核（rowChunks≤B、
// coresUsed=rc×nsc），本路径无早退分支，防御域空行块（nChunk=0）亦直达屏障。
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::CubeHalf()
{
    const Layout& lyt = layout_;
    // per-core 分派已在 Layout 构造完成（⚠ 950 核号约定：AIC 的 GetBlockIdx() 即
    // cluster 号；AIV 扁平 subcore 号 ÷ 核比——Init 同源）
    if (lyt.splitMode == 2) {
        const uint32_t nChunk = gbc::layout::CeilDiv(lyt.rowsThisCore, lyt.mChunk);
        for (uint32_t ci = 0; ci < nChunk; ++ci) {
            const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, lyt.rowsThisCore - ci * lyt.mChunk);
            CubeChunk(lyt.rowBase + ci * lyt.mChunk, mNow, ci == 0, PHASE_P0);
        }
        AscendC::SyncAll<false>(); // pass0 全局排空汇聚（见函数头 ⚠）
        for (uint32_t ci = 0; ci < nChunk; ++ci) {
            const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, lyt.rowsThisCore - ci * lyt.mChunk);
            CubeChunk(lyt.rowBase + ci * lyt.mChunk, mNow, false, PHASE_P1);
        }
        if (nChunk > 0) {
            gbc::sync::CubeWaitVec(); // 消费末块末轮 V2C（跨 launch 零残留）
        }
        AscendC::PipeBarrier<PIPE_ALL>;
        return;
    }
    if (lyt.rowBase >= lyt.batchSize) {
        return; // 无行的 cluster：与伴 AIV 一同早退，不参与任何 flag 轮
    }
    const uint32_t nChunk = gbc::layout::CeilDiv(lyt.rowsThisCore, lyt.mChunk);
    for (uint32_t ci = 0; ci < nChunk; ++ci) {
        const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, lyt.rowsThisCore - ci * lyt.mChunk);
        CubeChunk(lyt.rowBase + ci * lyt.mChunk, mNow, ci == 0, PHASE_BOTH);
    }
    // 收尾等待：消费末块末轮的 V2C，使整个 launch 的 V2C 轮次零残留（跨 launch
    // 的 flag 残留可能满足下一次 launch 的早期等待——防患于未然）。
    gbc::sync::CubeWaitVec();
    AscendC::PipeBarrier<PIPE_ALL>;
}

// ---------------------------------------------------------------------------
// VectorChunk — 一个 m-chunk 的 AIV 侧：pass0 全列片 → pass1 全列片
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::VectorChunk(uint32_t rowBase, uint32_t mNow, uint32_t phase)
{
    const Layout& lyt = layout_;
    AscendC::LocalTensor<float> rBar(AscendC::TPosition::VECOUT, lyt.rBarOff, lyt.planeElems);
    AscendC::LocalTensor<float> uBar(AscendC::TPosition::VECOUT, lyt.uBarOff, lyt.planeElems);
    AscendC::LocalTensor<float> rBar1(AscendC::TPosition::VECOUT, lyt.rBar1Off, lyt.planeElems); // S3'' 奇轮
    AscendC::LocalTensor<float> uBar1(AscendC::TPosition::VECOUT, lyt.uBar1Off, lyt.planeElems); // S3'' 奇轮
    AscendC::LocalTensor<float> cBar(AscendC::TPosition::VECOUT, lyt.cBarOff, lyt.planeElems);
    AscendC::LocalTensor<float> rAcc(AscendC::TPosition::VECCALC, lyt.rAccOff, lyt.planeElems);
    AscendC::LocalTensor<float> rComp(AscendC::TPosition::VECCALC, lyt.rCompOff, lyt.planeElems);
    AscendC::LocalTensor<float> uAcc(AscendC::TPosition::VECCALC, lyt.uAccOff, lyt.planeElems);
    AscendC::LocalTensor<float> uComp(AscendC::TPosition::VECCALC, lyt.uCompOff, lyt.planeElems);
    AscendC::LocalTensor<float> cAcc(AscendC::TPosition::VECCALC, lyt.cAccOff, lyt.planeElems);
    AscendC::LocalTensor<float> cComp(AscendC::TPosition::VECCALC, lyt.cCompOff, lyt.planeElems);
    AscendC::LocalTensor<float> t1(AscendC::TPosition::VECCALC, lyt.t1Off, lyt.planeElems);
    AscendC::LocalTensor<float> t2(AscendC::TPosition::VECCALC, lyt.t2Off, lyt.planeElems);
    AscendC::LocalTensor<float> t3(AscendC::TPosition::VECCALC, lyt.t3Off, lyt.planeElems);
    AscendC::LocalTensor<uint8_t> msk(
        AscendC::TPosition::VECCALC, lyt.mskOff,
        gbc::layout::CeilAlign(lyt.planeElems / gbc::layout::BITS_PER_BYTE, gbc::layout::C0F));
    // S3'：pass1 cBar 双缓冲——cBar1 复用 rBar 平面（pass1 中空闲：hr 已移驻 t1），
    // 与 AIC 的 DrainToUB(rBarOff/cBarOff 奇偶轮换) 同奇偶；!pf 旧路恒 cBar0。
    AscendC::LocalTensor<float> cBar1(AscendC::TPosition::VECOUT, lyt.rBarOff, lyt.planeElems);
    const uint32_t aKBytes = lyt.aKElems * static_cast<uint32_t>(sizeof(float)); // aHR 槽距
    const bool pf = (lyt.nSlice >= lyt.nL0c);
    const bool pf2 = pf && (lyt.cGroupsX >= 3u); // pass1 奇偶分级深度-2（与 AIC 同判据，见 AIC 段 ⚠）

    // 与 drain 相同的 stripe（本 AIV 独占的行区间）
    const gbc::layout::RowStripe stripe = gbc::layout::DrainStripe(mNow, true,
                                                                   static_cast<uint32_t>(AscendC::GetSubBlockIdx()));
    const bool hasRows = stripe.count > 0;
    const uint32_t gRow = rowBase + stripe.base; // 本 AIV 起始的全局行

    // ---- pass0：逐列片 r/u 补偿累加 → 门激活 → 落 GM ----
    // 每组：VecWaitCube → 归并 → VecSignalCube；pf 域 rBar/uBar 奇偶轮换（S3''）。
    // ⚠ 列片收尾必须并入末组轮内再置位（V2C FIFO 计数配对，单独成轮即失配）。
    if (phase != PHASE_P1) {
        uint32_t sIdx0 = 0; // 本核列片域内局部片序（gIdx 与 AIC 同源；A1 前全幅域逐位相同）
        for (uint32_t s = lyt.sliceLoCol; s < lyt.sliceHiCol; s += lyt.nL0c, ++sIdx0) {
            const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
            const uint32_t tileCols = (s < lyt.hiddenSize) ? gbc::layout::MinU(w, lyt.hiddenSize - s) : 0;
            const uint32_t work = stripe.count * w;
            const uint64_t gmOff = static_cast<uint64_t>(gRow) * lyt.hiddenSize + s;

            if (hasRows) {
                AscendC::Duplicate(rComp, 0.0f, work);
                AscendC::Duplicate(uComp, 0.0f, work);
            }
            for (uint32_t g = 0; g < lyt.cGroups; ++g) {
                const uint32_t gIdx = sIdx0 * lyt.cGroups + g; // pass0 线性轮序（与 AIC 同源）
                const AscendC::LocalTensor<float> rBarP = (pf && ((gIdx & 1U) != 0U)) ? rBar1 : rBar;
                const AscendC::LocalTensor<float> uBarP = (pf && ((gIdx & 1U) != 0U)) ? uBar1 : uBar;
                gbc::sync::VecWaitCube();
                AscendC::PipeBarrier<PIPE_ALL>();
                if (hasRows) {
                    if (g == 0) {
                        AscendC::DataCopy(rAcc, rBarP, work);
                        AscendC::DataCopy(uAcc, uBarP, work);
                        // 轮尾定向事件（范式误区3错误3）：rBar/uBar 的 V 读须先于本轮
                        // VecSignalCube 的 MTE3 跨核置位完成——命名 V→MTE3 hazard。
                        gbc::sync::WaitVToMte3();
                    } else {
                        // Neumaier（Knuth TwoSum 无分支形态）：舍入损失进补偿项；PIPE_V 为
                        // t1 回拷读完成前的同管防御标记。
                        AscendC::Add(t1, rAcc, rBarP, work);
                        AscendC::Sub(t2, rAcc, t1, work);
                        AscendC::Add(t2, t2, rBarP, work);
                        AscendC::Add(rComp, rComp, t2, work);
                        AscendC::DataCopy(rAcc, t1, work);
                        AscendC::PipeBarrier<PIPE_V>(); // t1 拷贝读完成前不得复写（同管保序）
                        AscendC::Add(t1, uAcc, uBarP, work);
                        AscendC::Sub(t2, uAcc, t1, work);
                        AscendC::Add(t2, t2, uBarP, work);
                        AscendC::Add(uComp, uComp, t2, work);
                        AscendC::DataCopy(uAcc, t1, work);
                        // 轮尾定向事件（非末组轮）：uBar/rBar 的 V 读须先于 VecSignalCube 的
                        // MTE3 跨核置位——命名 V→MTE3 hazard。
                        gbc::sync::WaitVToMte3();
                    }
                    if (g == lyt.cGroups - 1) {
                        // 列片收尾（并入末轮，见上 ⚠）。B1：pass0 只输出 r/u 到 GM，不再
                        // 现场算 hr/回灌 aH（hr 改由 pass1 逐组从 GM 的 r+hPrev 重算）。
                        gbc::layout::FinFinalize(rAcc, rComp, t1, t2, msk, work); // r_bar 就绪
                        gbc::layout::FinFinalize(uAcc, uComp, t1, t2, msk, work); // u_bar 就绪
                        AscendC::PipeBarrier<PIPE_V>(); // FinFinalize 与 SigmoidVec 同为 V 管指令，同管保序
                        gbc::vector::SigmoidVec(rAcc, t1, t2, work); // r = σ(r_bar)（就地）
                        gbc::vector::SigmoidVec(uAcc, t1, t2, work); // u = σ(u_bar)（就地）
                        // 定向事件：V 写 rAcc/uAcc → MTE3（CopyOut r/u 读）。
                        gbc::sync::WaitVToMte3();
                        // 输出 r（fp32 直出，字节精确；pass1 hr 重算与 blend 均回读）/ u（pass1 blend 经 t2 回读）
                        CopyTileToGm(gmR_, gmOff, rAcc, stripe.count, w, lyt.hiddenSize, tileCols);
                        CopyTileToGm(gmU_, gmOff, uAcc, stripe.count, w, lyt.hiddenSize, tileCols);
                    }
                }
                gbc::sync::VecSignalCube(); // ⚠ count==0 的 AIV 也必须 set——barrier
            }
        }
    } // phase != PHASE_P1（pass0 段；A1 相位拆分时全局屏障在 VectorHalf 层）

    // ---- pass1：逐列片 c_bar 补偿累加 + 候选激活 + 状态更新 ----
    // 每组：VecWaitCube → Neumaier 归并 → VecSignalCube；g=0 以首组部分和为初值。
    if (phase != PHASE_P0) {
        uint32_t sIdx = 0; // 本核列片域内局部片序（it 与 AIC 同源）
        for (uint32_t s = lyt.sliceLoCol; s < lyt.sliceHiCol; s += lyt.nL0c, ++sIdx) {
            const uint32_t w = gbc::layout::MinU(lyt.nL0c, lyt.padHidden - s);
            const uint32_t tileCols = (s < lyt.hiddenSize) ? gbc::layout::MinU(w, lyt.hiddenSize - s) : 0;
            const uint32_t work = stripe.count * w;
            const uint64_t gmOff = static_cast<uint64_t>(gRow) * lyt.hiddenSize + s;
            const uint16_t rep = static_cast<uint16_t>(AscendC::CeilDivision(work, VL_F32));

            if (hasRows) {
                AscendC::Duplicate(cComp, 0.0f, work); // 补偿项清零（首组前）
            }
            for (uint32_t g = 0; g < lyt.cGroups; ++g) {
                const uint32_t it = sIdx * lyt.cGroups + g; // pass1 线性门序（与 AIC 同源）
                // S3'：pf 域 cBar 奇偶轮换（cBar1≡rBar 平面），与 AIC drain 同奇偶；!pf 恒 cBar0
                const AscendC::LocalTensor<float> cBarP = (pf && ((it & 1U) != 0U)) ? cBar1 : cBar;
                gbc::sync::VecWaitCube();
                AscendC::PipeBarrier<PIPE_ALL>();
                if (hasRows) {
                    // ---- pf2 轮首回灌：上一轮 StageHr 暂存的 hr → aHR[(it+2)%3]。
                    // 落点 = AIC 的 L1 静默窗（L1 无写仲裁，深度-2 安全前提）。----
                    const uint32_t gF = g + 2;
                    const bool didFb = pf2 && (g >= 1) && (gF < lyt.cGroups) && (gF >= lyt.cGroupsX);
                    if (didFb) {
                        FeedbackHr(gF, lyt.aHROff + ((it + 2) % 3u) * aKBytes, stripe, mNow, t3);
                    }
                    if (g == 0) {
                        AscendC::DataCopy(cAcc, cBarP, work); // 首组部分和即累加器初值
                        // 轮尾定向事件（原 PIPE_ALL）：cBar 的 V 读先于本轮 VecSignalCube 的
                        // MTE3 跨核置位——命名 V→MTE3 hazard。
                        gbc::sync::WaitVToMte3();
                    } else {
                        // Neumaier：大数吃小数的舍入损失进补偿项
                        AscendC::Add(t1, cAcc, cBarP, work);  // t = cAcc + p
                        AscendC::Sub(t2, cAcc, t1, work);     // cAcc − t
                        AscendC::Add(t2, t2, cBarP, work);    // + p
                        AscendC::Add(cComp, cComp, t2, work); // cComp += (cAcc − t) + p
                        AscendC::DataCopy(cAcc, t1, work);    // cAcc = t
                        // 轮尾定向事件（非末组轮，原 PIPE_ALL）：cBar 的 V 读先于 VecSignalCube
                        // 的 MTE3 跨核置位——命名 V→MTE3 hazard。
                        gbc::sync::WaitVToMte3();
                    }
                    // ---- B1+S3'：hr 前瞻（hPrev⊙r 切片 → aHR 槽 / t3 暂存）----
                    // pf2：轮尾 StageHr 暂存 hr[g+3]→t3，下一轮轮首 FeedbackHr 回灌
                    //   aHR[(it+3)%3]（轮首窗 = AIC L1 静默窗，深度-2 安全前提）。
                    // !pf2&&pf：轮尾一体重算+回灌（前瞻-3）；特例 g<3 的 h 门于
                    //   merge[(s,0)] 一并写入（AIC 门 (s,1) need 提升为 it）。
                    // !pf：本轮写 hr[g+1]→aHR 槽 0（AIC 下一轮组顶消费后读）。
                    if (pf2) {
                        const uint32_t gN3 = g + 3;
                        if (gN3 < lyt.cGroups && gN3 >= lyt.cGroupsX) {
                            if (didFb) {
                                gbc::sync::WaitMte3ToV(); // 轮首 feedback 的 MTE3 读 t3 先于 V 复写
                            }
                            StageHr(gN3, stripe, gRow, t1, t2, t3);
                            gbc::sync::WaitVToMte3(); // Mul（V 写 t3）先于 VecSignalCube 的 MTE3 置位
                        }
                    } else if (pf) {
                        const uint32_t gN3 = g + 3;
                        if (gN3 < lyt.cGroups && gN3 >= lyt.cGroupsX) {
                            RecomputeHr(gN3, lyt.aHROff + (it % 3u) * aKBytes, stripe, gRow, mNow, t1, t2);
                        }
                        if (g == 0 && lyt.cGroupsX < 3u) {
                            const uint32_t gSpecialHi = gbc::layout::MinU(3u, lyt.cGroups);
                            for (uint32_t gS = lyt.cGroupsX; gS < gSpecialHi; ++gS) {
                                RecomputeHr(gS, lyt.aHROff + ((it + gS) % 3u) * aKBytes, stripe, gRow, mNow, t1, t2);
                            }
                        }
                    } else {
                        const uint32_t gN = g + 1;
                        if (gN < lyt.cGroups && gN >= lyt.cGroupsX) {
                            RecomputeHr(gN, lyt.aHROff, stripe, gRow, mNow, t1, t2);
                        }
                    }
                }
                gbc::sync::VecSignalCube(); // ⚠ count==0 的 AIV 也必须 set——barrier（PIPE_MTE3 亦排空 hr 回灌）
            }
            if (hasRows) {
                gbc::layout::FinFinalize(cAcc, cComp, t1, t2, msk, work); // c_bar 就绪（含有限性守卫）
                AscendC::PipeBarrier<PIPE_V>(); // FinFinalize 与 TanhVec 同为 V 管指令，同管保序
                // Tanh 的第 3 scratch 落 rAcc 平面（≡t3Off；rAcc 自 pass0 末列片 σ(r)+r CopyOut
                // 后无人触及，pass1 各组亦只用 cAcc/cComp/t1/t2）
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
                // h = c + u·(hp − c)（VF 主链；h 落 t3≡rAcc staging）。⚠ 不得改回 cAcc
                // 就地回写：与下一片轮 0 DataCopy(cAcc) 零窗竞态（Fix B，实测翻车）。
                asc_vf_call<GruBlockCellBlendVF<float>>(reinterpret_cast<__ubuf__ float*>(t3.GetPhyAddr()),
                                                        reinterpret_cast<__ubuf__ float*>(cAcc.GetPhyAddr()),
                                                        reinterpret_cast<__ubuf__ float*>(t2.GetPhyAddr()),
                                                        reinterpret_cast<__ubuf__ float*>(t1.GetPhyAddr()), work,
                                                        VL_F32, rep);
                AscendC::PipeBarrier<PIPE_ALL>(); // VF 异步读闭合 → MTE3 输出 h（必须全栅栏，同上 VF 规则）
                CopyTileToGm(gmH_, gmOff, t3, stripe.count, w, lyt.hiddenSize, tileCols); // 输出 h（t3≡rAcc staging）
                // t3≡rAcc 的下一 V 写者（下一片 TanhVec s3 / 下一块 pass0 DataCopy(rAcc)）
                // 常规靠轮间自然时延，此定向事件为架构闭合（MTE3 读排空先于后续 V 写）。
                gbc::sync::WaitMte3ToV();
            }
            // 末列片末组的 VecSignalCube 即本块的 V2C 收尾（下一块 pass0 drain 的 WAR
            // 屏障；本片后续 tanh/blend/copyout 只读 AIV 私有平面与 GM，与 cube 无竞争）。
        }
    } // phase != PHASE_P0（pass1 段）
}

// ---------------------------------------------------------------------------
// VectorHalf — 本 cluster 的全部 m-chunk（与 CubeHalf 同一套分派标量；A1 相位拆分
// 与屏障契约见 CubeHalf 函数头——AIV 侧 SyncAll：set_intra(PIPE_MTE3, SYNC_AIV_FLAG)
// 的 MTE3 管序 ⇒ 本核 pass0 的 r/u CopyOut 排空后才置位，随后等 AIC 释放）
// ---------------------------------------------------------------------------
__aicore__ inline void GruBlockCellKernel::VectorHalf()
{
    const Layout& lyt = layout_;
    if (lyt.splitMode == 2) {
        const uint32_t nChunk = gbc::layout::CeilDiv(lyt.rowsThisCore, lyt.mChunk);
        for (uint32_t ci = 0; ci < nChunk; ++ci) {
            const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, lyt.rowsThisCore - ci * lyt.mChunk);
            VectorChunk(lyt.rowBase + ci * lyt.mChunk, mNow, PHASE_P0);
        }
        AscendC::SyncAll<false>(); // pass0 全局排空汇聚（每核恰一次，无早退分支）
        for (uint32_t ci = 0; ci < nChunk; ++ci) {
            const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, lyt.rowsThisCore - ci * lyt.mChunk);
            VectorChunk(lyt.rowBase + ci * lyt.mChunk, mNow, PHASE_P1);
        }
        AscendC::PipeBarrier<PIPE_ALL>;
        return;
    }
    if (lyt.rowBase >= lyt.batchSize) {
        return;
    }
    const uint32_t nChunk = gbc::layout::CeilDiv(lyt.rowsThisCore, lyt.mChunk);
    for (uint32_t ci = 0; ci < nChunk; ++ci) {
        const uint32_t mNow = gbc::layout::MinU(lyt.mChunk, lyt.rowsThisCore - ci * lyt.mChunk);
        VectorChunk(lyt.rowBase + ci * lyt.mChunk, mNow, PHASE_BOTH);
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
