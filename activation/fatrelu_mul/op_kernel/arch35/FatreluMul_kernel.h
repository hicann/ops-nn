/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// fatrelu_mul_package/op_kernel/arch35/FatreluMul_kernel.h
// =============================================================================
//
// ROLE: FatreluMul device-side kernel implementation (arch35 / DAV_3510, pure
//   Vector operator).
//   Kernel template class FatreluMulKernel<T, kPath> + the cross-branch shared
//   fp32-domain VF compute chain FatreluMulVF. Implemented per:
//     - docs/fatrelu_mul/design/Kernel.md — shared skeleton: shared VF chain
//       (CompareScalar(LE) + Select + Mul, one asc_vf_call consumes the whole
//       chain), shared ComputeTile entry, fp16/bf16 Cast up/down organization,
//       per-dtype TBuf count (P_fp32 = 3 / P_fp16·bf16 = 6, no cross-dtype
//       kPhysNodes constant);
//     - docs/fatrelu_mul/design/branches/DESIGN-BRANCH-0.md §3–§5 — small-tail
//       row-group path (tilingKey = 0, FATRELUMUL_PATH_SMALL_TAIL):
//         §5.1 Init   : GM binding (x / threshold / y) + per-dtype TBuf
//                       allocation (one role one buffer) + threshold scalar GM
//                       direct read + this-core row range / row pitch
//                       precompute + sync event IDs (FetchEventID);
//         §5.2 CopyIn : row-group strided multi-row DataCopyPad (x1 / x2 half
//                       zones, blockCount = gRows, blockLen = d·S, GM srcStride
//                       = d·S byte gap, UB dstStride = 0 → hardware lays each
//                       row at CeilAlign(d·S, 32B) pitch);
//         §5.3 Compute: flat in-group single shared-chain call — fp32 direct
//                       ComputeTile; fp16/bf16 Cast(CAST_NONE)↑ ×2 →
//                       ComputeTile → Cast(CAST_RINT)↓;
//         §5.4 CopyOut: symmetric row-group multi-row DataCopyPad back to y
//                       (only d·S valid bytes per row, pitch dummies dropped);
//         §5.5 Process: TBuf single-buffer group pipeline — in-group
//                       MTE2→V→MTE3 ordered by MTE2_V / V_MTE3 RAW pairs,
//                       cross-group buffer release by V_MTE2 / MTE3_V WAR
//                       pairs (first group skips WAR Wait, last group skips
//                       WAR Set; next-group CopyIn overlaps this-group CopyOut
//                       — disjoint buffers).
//   Empty-tensor short-circuit sub-path (batch_size == 0 || half_dim == 0,
//   host sets rows_former = 0 / tile_elems = 0 / rows_per_group = 0): Process
//   returns at rowCount_ == 0 with no CopyIn / Compute / CopyOut / Sync, and
//   Init skips TBuf allocation (tile_elems == 0) and pitch derivation
//   (half_dim == 0, prevents 0-divisor).
//
//   The big-tail path (kPath = FATRELUMUL_PATH_BIG_TAIL, tilingKey = 1,
//   docs/fatrelu_mul/design/branches/DESIGN-BRANCH-1.md §3–§5) is dispatched via
//   `if constexpr` in the shared class (same members / same shared VF chain):
//         §5.1 Init   : shared with small-tail (GM binding + per-dtype TBuf
//                       allocation + threshold scalar GM direct read + sync
//                       event IDs); no row-group pitch precompute;
//         §5.2 CopyIn : in-row flat segmentation — per segment two
//                       blockCount=1 DataCopyPad (gate seg → B0, up seg → B1;
//                       blockLen = segLen·S valid bytes, non-32B-align legal;
//                       the two half-zone segs are GM (d − segLen) elements
//                       apart → no blockCount=2 batching);
//         §5.3 Compute: per-segment shared-chain call — fp32 direct
//                       ComputeTile; fp16/bf16 Cast(CAST_NONE)↑ ×2 →
//                       ComputeTile → Cast(CAST_RINT)↓; segLen runtime-derived
//                       (full seg = TE / tail seg = d % TE, UpdateMask masks
//                       the tail);
//         §5.4 CopyOut: per-segment single DataCopyPad back to y
//                       (blockLen = segLen·S valid bytes, dense single path);
//         §5.5 Process: row × segment double loop (rowStart/rowCount per
//                       §2 multicore split; x1/x2/y seg offsets r·2d+s·TE /
//                       r·2d+d+s·TE / r·d+s·TE); per-segment four-event sync
//                       (MTE2_V / V_MTE3 RAW in-segment; V_MTE2 / MTE3_V WAR
//                       cross-segment — first segment skips WAR Wait, last
//                       segment skips WAR Set; Set/Wait strictly paired);
//                       PipeBarrier<PIPE_ALL> drains the pipeline at exit.
//
// CONTENTS:
//   - FatreluMulVF            — shared fp32-domain VF chain (Kernel.md「范式
//                                特有 Kernel 子节 · 共享 VF 计算链」)
//   - class FatreluMulKernel<T, kPath> — Init / Process / CopyIn / Compute /
//                                CopyOut / ComputeTile
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FatreluMul   — FatreluMulKernel class prefix, VF fn prefix
//   UPPER        : FATRELUMUL   — TPL path macros (FatreluMul_struct.h)
//
// =============================================================================

#pragma once

#include <type_traits>

#include "kernel_operator.h"        // Ascend C core framework (AscendC:: namespace)
#include "FatreluMul_struct.h"      // FATRELUMUL_PATH_SMALL_TAIL / BIG_TAIL (= TilingKey.md)
#include "FatreluMul_tiling_data.h" // FatreluMulTilingData / kMaxInputSlots / kMaxOutputSlots (= TilingData.md)

// ---------------------------------------------------------------------------
// FatreluMulMin — int64_t 最小值（设备侧工具；libstdc++ std::min 被 ccec 标记
// 为 [host] 函数，不可从 [aicore] 设备函数调用，故内置等价实现）
// ---------------------------------------------------------------------------
__aicore__ inline int64_t FatreluMulMin(int64_t a, int64_t b) { return (a < b) ? a : b; }

// ===========================================================================
// FatreluMulVF — 共享 VF 计算链（fp32 域，跨分支跨 dtype 共用；C 函数格式、
// 非类成员，成员变量一律在 asc_vf_call 调用侧提取为参数传入 — VF 硬约束 5）
//
// 语义（docs/fatrelu_mul/design/Overview.md「数学公式(LaTeX)」，spec.yaml
// math_semantics 同口径）：
//   act = (x1f <= t) ? 0 : x1f   —— LE 极性（Kernel.md「mask 极性分析与逐元素
//                                   对应」）：NaN <= t 为 False → NaN 保留；
//                                   x1 == t 单点边界同置 0
//   yf  = act * x2f              —— 单次比较/选择精确、单次乘法仅一次输出舍入；
//                                   NaN / ±Inf 随 Mul 按 IEEE 754 传播
// 一条 asc_vf_call 吃掉整链（LoadAlign → CompareScalar → Duplicate → Select →
// Mul → StoreAlign，禁止拆成多个操作）；cmpMask / act 全程寄存器、不落 UB。
// ===========================================================================
__simd_vf__ inline void FatreluMulVF(__ubuf__ float* yfAddr, __ubuf__ float* x1fAddr, __ubuf__ float* x2fAddr,
                                     float thresholdF32, uint32_t totalElems, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> x1fReg, x2fReg, actReg, zeroReg, yfReg;
    AscendC::Reg::MaskReg mask, cmpMask;
    constexpr uint32_t kRepF32 = 256 / sizeof(float); // VL = 256B → 64 元素 / repeat
    uint32_t remaining = totalElems;                  // UpdateMask 引用递减源（调用后不可手动再减）
    for (uint16_t i = 0; i < repeatTime; ++i) {       // VF 硬约束：从 0 起、uint16_t 控制变量
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining); // remaining 内部自动递减 VL_T（尾段 mask）
        AscendC::Reg::LoadAlign(x1fReg, x1fAddr + off); // UB → 寄存器（地址 32B 对齐：off × 4B 为 256B 倍数）
        AscendC::Reg::LoadAlign(x2fReg, x2fAddr + off);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(cmpMask, x1fReg, thresholdF32, mask);
        AscendC::Reg::Duplicate(zeroReg, 0.0f, mask);                  // 0 向量寄存器构造，不占 UB
        AscendC::Reg::Select<float>(actReg, zeroReg, x1fReg, cmpMask); // cmpMask=1 → 0；=0 → x1f（NaN 保留）
        AscendC::Reg::Mul(yfReg, actReg, x2fReg, mask);      // NaN / ±Inf 按 IEEE 754 传播，不饱和不报错
        AscendC::Reg::StoreAlign(yfAddr + off, yfReg, mask); // 寄存器 → UB（带 mask）
    }
}

// ===========================================================================
// class FatreluMulKernel<T, kPath>
//
// Template parameters:
//   T     — data type of x / threshold / y (DTYPE_X injected per OpDef dtype
//           column by the build system: float / half / bfloat16)
//   kPath — TPL path selector (FatreluMul_struct.h):
//             FATRELUMUL_PATH_SMALL_TAIL = 0 — row-group packing (this file)
//             FATRELUMUL_PATH_BIG_TAIL   = 1 — in-row segmentation
//           Branch-specific bodies dispatch via if constexpr; both class
//           instantiations must compile.
//
// Shared members (both paths):
//   pipe_        — TPipe for TBuf static allocation
//   td_          — FatreluMulTilingData (global non-templated, TilingData.md)
//   gmX_ / gmThreshold_ / gmY_ — GM tensor views (x / threshold / y)
//   threshold_   — scalar threshold read once from GM in T dtype
//   buf_[]       — per-dtype P TBuf slots (fp32: B0=x1/B1=x2/B2=y;
//                  fp16/bf16: B0=x1(T)/B1=x2(T)/B2=x1f/B3=x2f/B4=yf/B5=y(T),
//                  Cast breakpoint src/dst independent buffers)
//   ev*          — sync event IDs (FetchEventID, never hard-coded)
//
// Small-tail only members:
//   rowStart_ / rowCount_      — this-core row range (row-domain multicore split)
//   rowPitchElems_             — UB row pitch in elements = CeilAlign(d·S,32)/S
//   rowsPerGroupEff_           — pitch-safe effective group rows = min(
//                                rows_per_group, tileElems/pitchElems, 4095)
// ===========================================================================
template <typename T, int64_t kPath>
class FatreluMulKernel {
public:
    // -----------------------------------------------------------------------
    // Init — GM binding + TBuf allocation + threshold direct read + row range
    //        / pitch precompute + event IDs (DESIGN-BRANCH-0.md §5.1)
    // -----------------------------------------------------------------------
    __aicore__ inline void Init(GM_ADDR* ins, GM_ADDR* outs, const FatreluMulTilingData* td)
    {
        td_ = td;
        // GM 绑定（kMaxInputSlots=2 / kMaxOutputSlots=1，TilingData.md §1；
        // threshold 为单元素标量 Tensor，GM 直读、不占 UB 槽位）
        gmX_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ins[0]));
        gmThreshold_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ins[1]));
        gmY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outs[0]));
        // threshold 标量 GM 直读：按输入 dtype T 读入（Kernel.md「Init」共享口径 #1；
        // 先舍入到 x dtype 再比较，对齐 PyTorch threshold.to<scalar_t>()，
        // Overview.md「精度敏感点」①）
        threshold_ = T(gmThreshold_.GetValue(0));
        // TBuf 分配（§4 一角色一 buffer；tile_elems 已 256B 对齐 ⇒ 各 buffer 字节数
        // 32B 整数倍，满足 InitBuffer 约束；空 Tensor 短路 tile_elems=0 → 跳过分配）
        if (td_->tile_elems > 0) {
            const uint32_t bytesT = static_cast<uint32_t>(td_->tile_elems * static_cast<int64_t>(sizeof(T)));
            const uint32_t bytesF = static_cast<uint32_t>(td_->tile_elems * static_cast<int64_t>(sizeof(float)));
            if constexpr (std::is_same_v<T, float>) { // fp32：P=3，全程原生 fp32
                pipe_.InitBuffer(buf_[0], bytesF);    // B0 = x1
                pipe_.InitBuffer(buf_[1], bytesF);    // B1 = x2
                pipe_.InitBuffer(buf_[2], bytesF);    // B2 = y
            } else {                                  // fp16/bf16：P=6，Cast 断链独立 buffer
                pipe_.InitBuffer(buf_[0], bytesT);    // B0 = x1(T)
                pipe_.InitBuffer(buf_[1], bytesT);    // B1 = x2(T)
                pipe_.InitBuffer(buf_[2], bytesF);    // B2 = x1f(fp32)
                pipe_.InitBuffer(buf_[3], bytesF);    // B3 = x2f(fp32)
                pipe_.InitBuffer(buf_[4], bytesF);    // B4 = yf(fp32)
                pipe_.InitBuffer(buf_[5], bytesT);    // B5 = y(T)
            }
        }
        if constexpr (kPath == FATRELUMUL_PATH_SMALL_TAIL) {
            // 本核行区间（行域多核切分消费，TilingData.md「kernel 消费点」；
            // 入口已做 blockIdx ≥ need_core_num 守卫）
            const int64_t bIdx = AscendC::GetBlockIdx();
            rowStart_ = bIdx * td_->rows_former + FatreluMulMin(bIdx, td_->rows_tail_core);
            rowCount_ = td_->rows_former + (bIdx < td_->rows_tail_core ? 1 : 0);
            // 行距与 pitch 安全有效组行数（kernel 运行期推导，不入 TilingData——
            // 「尾块运行期推导」约束，见 §2.5）；half_dim == 0（空 Tensor 短路）
            // 时不做除法（防 0 除子：rowCount_==0 直接返回，不消费派生量）
            if (td_->half_dim > 0) {
                const int64_t S = static_cast<int64_t>(sizeof(T));
                const int64_t pitchBytes = (td_->half_dim * S + 31) / 32 * 32; // CeilAlign(d·S,32)
                rowPitchElems_ = pitchBytes / S; // UB 内行距（元素）＝ d（32B 对齐时）
                if (rowPitchElems_ > 0) {
                    rowsPerGroupEff_ = FatreluMulMin(
                        FatreluMulMin(td_->rows_per_group, td_->tile_elems / rowPitchElems_),
                        static_cast<int64_t>(4095));
                    // 4095 = DataCopyExtParams.blockCount 上限（防御性；arch35 下 pitch
                    // 钳制已保证 ≤ 2640）
                }
            }
        }
        // 流水线同步事件 ID（FetchEventID 托管分配，不可硬编码；同类型事件全程共用
        // 1 个，见 §5.5；GetTPipePtr 为全局作用域函数，非 AscendC 命名空间成员）
        evMte2ToV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        evVtoMte3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        evVtoMte2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        evMte3ToV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    // -----------------------------------------------------------------------
    // Process — small-tail 行组流水（同步编排；首组跳跨组 WaitFlag、末组跳跨组
    //           SetFlag，Set/Wait 逐组配对，DESIGN-BRANCH-0.md §5.5）
    // big-tail（kPath = FATRELUMUL_PATH_BIG_TAIL）行×段双循环按
    // DESIGN-BRANCH-1.md §5.3：段偏移/段长全部运行期推导（§2 公式），
    // per-段三段式流水 CopyIn → Compute → CopyOut（同步对在各自方法内按
    // §5.5 四事件表落位），kernel 退出前 PipeBarrier<PIPE_ALL> 排空。
    // -----------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        if constexpr (kPath == FATRELUMUL_PATH_SMALL_TAIL) {
            if (rowCount_ == 0) {
                return;
            } // 空 Tensor 短路子路径：0 行直接返回（无搬运/计算/同步）
            const int64_t groups = (rowCount_ + rowsPerGroupEff_ - 1) / rowsPerGroupEff_;
            for (int64_t gi = 0; gi < groups; ++gi) {
                const int64_t r0 = rowStart_ + gi * rowsPerGroupEff_; // 组首全局行号
                const int64_t gRows = FatreluMulMin(rowsPerGroupEff_,
                                                    rowCount_ - gi * rowsPerGroupEff_); // 尾组运行期推导
                const int64_t gElems = gRows * rowPitchElems_; // 组内平坦元素数（含行距哑元）
                // —— MTE2 阶段：本组 CopyIn（可与上组 CopyOut 重叠：B0/B1 与 B5/B2 不相交）——
                if (gi > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMte2_);
                } // WAR：上组 V 读毕 B0/B1
                CopyIn(r0, gRows); // x1→B0, x2→B1（MTE2）
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2ToV_);
                // —— V 阶段：组内平坦计算 ——
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2ToV_); // RAW：B0/B1 就绪
                if (gi > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3ToV_);
                } // WAR：上组 MTE3 读毕输出 buffer
                Compute(gElems); // [Cast↑×2→]ComputeTile[→Cast↓]
                if (gi < groups - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMte2_);
                } // 释放 B0/B1 给下组
                // —— MTE3 阶段：本组 CopyOut（与下组 CopyIn 重叠）——
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMte3_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMte3_); // RAW：输出 buffer 就绪
                CopyOut(r0, gRows);                                        // y←B5/B2（MTE3）
                if (gi < groups - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evMte3ToV_);
                } // 释放输出 buffer 给下组
            }
        } else {
            // FATRELUMUL_PATH_BIG_TAIL（tilingKey=1，行内分段）：DESIGN-BRANCH-1.md §5.3
            // 本核行区间（§2 公式(2)，行域多核切分；入口已做 blockIdx ≥ need_core_num 守卫；
            // 进入本分支即非空 batch_size ≥ 1 且 d > tileElems，每核 rowCount ≥ 1）
            const int64_t b = AscendC::GetBlockIdx();
            const int64_t d = td_->half_dim; // d > tile_elems（本分支判界）
            const int64_t te = td_->tile_elems;
            const int64_t rowStart = b * td_->rows_former + FatreluMulMin(b, td_->rows_tail_core); // §2 公式(2)
            const int64_t rowCount = td_->rows_former + (b < td_->rows_tail_core ? 1 : 0);
            const int64_t lastRow = rowStart + rowCount - 1;
            const int64_t fullSegs = d / te; // 主段数（主段长恒 TE）
            const int64_t tailSeg = d % te;  // 尾段长（=0 无尾段；运行期推导，不入 TilingData）
            const int64_t segCount = fullSegs + (tailSeg > 0 ? 1 : 0); // d > TE ⇒ ≥ 2
            bool isFirst = true;
            for (int64_t r = rowStart; r <= lastRow; ++r) {
                for (int64_t s = 0; s < segCount; ++s) {
                    const int64_t segLen = (s < fullSegs) ? te : tailSeg; // §2 公式(3)
                    const int64_t x1Off = r * 2 * d + s * te;             // gate 段首（元素索引）
                    const int64_t x2Off = r * 2 * d + d + s * te;         // up 段首
                    const int64_t yOff = r * d + s * te;                  // y 段首
                    const bool isLast = (r == lastRow) && (s == segCount - 1);
                    CopyIn(x1Off, x2Off, segLen, isFirst); // §5.2（MTE2）
                    Compute(segLen, isFirst, isLast);      // §5.3（V）
                    CopyOut(yOff, segLen, isLast);         // §5.4（MTE3）
                    isFirst = false;
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>(); // kernel 退出前排空全部流水线（broadcast 范式纪律）
        }
    }

private:
    // -----------------------------------------------------------------------
    // ComputeTile — 跨分支共享 Compute 入口（kernel 类成员，__aicore__ 域；
    // VF 硬约束 5：成员变量调用侧提取）
    // fp32 路径：yfAddr / x1fAddr / x2fAddr 即 T buffer（T = float）；
    // fp16 / bf16 路径：为 Cast 上行后的 fp32 buffer（x1f / x2f）与下行前的
    // fp32 buffer（yf）。
    // -----------------------------------------------------------------------
    __aicore__ inline void ComputeTile(__ubuf__ float* yfAddr, __ubuf__ float* x1fAddr, __ubuf__ float* x2fAddr,
                                       uint32_t count)
    {
        constexpr uint32_t kRepF32 = AscendC::GetVecLen() / sizeof(float); // 256 / 4 = 64
        const uint16_t repeatTime = static_cast<uint16_t>((count + kRepF32 - 1) / kRepF32);
        // T → fp32 精确上行（fp16/bf16 ⊂ fp32，无二次舍入）：threshold 先舍入到 x dtype
        // 再比较、比较前无二次舍入（Overview.md 精度敏感点①）。bf16 为编译器内建
        // __bf16 类型，bisheng 后端不支持其 →float 的标量 static_cast（"not support
        // bf16 type cast"），须走 CANN 标量转换 API AscendC::ToFloat（kernel_scalar_
        // convert.h，与内置 drop_out_do_mask 读标量同场景用法）。
        float thresholdF32 = 0.0f;
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            thresholdF32 = AscendC::ToFloat(threshold_);
        } else {
            thresholdF32 = static_cast<float>(threshold_);
        }
        asc_vf_call<FatreluMulVF>(yfAddr, x1fAddr, x2fAddr, thresholdF32, count, repeatTime);
    }

    // -----------------------------------------------------------------------
    // CopyIn — small-tail GM → UB（MTE2）：行组带 stride 多行整块搬运
    // （DESIGN-BRANCH-0.md §5.2）
    //   r0 = 组首全局行号，gRows = 组行数（尾组运行期推导）
    // x1 / x2 半区各一次 DataCopyPad：blockCount = gRows、blockLen = d·S（不要求
    // 32B 对齐，硬件只写有效字节）；GM 侧 srcStride = d 元素（行尾→下一行同半区
    // 行头的 byte gap），UB 侧 dstStride = 0（32B datablock 单位 gap=0 ⇒ 行按
    // CeilAlign(d·S,32) 间距落位，行尾哑元不污染有效数据）。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyIn(int64_t r0, int64_t gRows)
    {
        const int64_t d = td_->half_dim;
        const int64_t S = static_cast<int64_t>(sizeof(T));
        AscendC::DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(gRows); // ≤ 4095（rowsPerGroupEff_ 已钳制，§5.1）
        params.blockLen = static_cast<uint32_t>(d * S);   // 行有效字节（≥2B，不要求 32B 对齐）
        params.srcStride = d * S; // GM 侧 gap（byte）：x1 行尾→下行 x1 行头 = d 元素
        params.dstStride = 0;     // UB 侧 gap（32B 单位）=0：硬件按 CeilAlign(d·S,32) 推进
        params.rsv = 0;           // 必须显式填 0
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // gate 半区：组内第 i 行 x1 首地址 = (r0+i)·2d（元素）
        AscendC::DataCopyPad(buf_[0].Get<T>(), gmX_[r0 * 2 * d], params, padParams);
        // up 半区：组内第 i 行 x2 首地址 = (r0+i)·2d + d；GM gap 同为 d 元素
        AscendC::DataCopyPad(buf_[1].Get<T>(), gmX_[r0 * 2 * d + d], params, padParams);
    }

    // -----------------------------------------------------------------------
    // Compute — small-tail 组内平坦计算（V pipe；Cast 与 VF 链同属 Vector
    // pipe，指令序保序，无 LocalMemBar）（DESIGN-BRANCH-0.md §5.3）
    //   gElems = gRows × rowPitchElems（含行距哑元；哑元随平坦序列流过 VF 链，
    //   结果落 B5/B2 padding 槽位，CopyOut 只写有效字节）
    // -----------------------------------------------------------------------
    __aicore__ inline void Compute(int64_t gElems)
    {
        const uint32_t count = static_cast<uint32_t>(gElems);
        if constexpr (std::is_same_v<T, float>) {
            // fp32 路径 — P=3（B0=x1 / B1=x2 / B2=y，全程原生 fp32）
            // 执行前: 持有=[B0, B1]；执行中: 持有=[B0, B1, B2]（P_fp32=3 峰值，
            // cmpMask/act 全程寄存器不占 UB）；执行后: 持有=[B2] ← B0/B1 释放给下一组
            ComputeTile(reinterpret_cast<__ubuf__ float*>(buf_[2].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[0].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[1].Get<float>().GetPhyAddr()),
                        count); // 共享 VF 链（Kernel.md「共享 VF 计算链」）
        } else {
            // fp16/bf16 路径 — P=6（B0/B1/B5 T 路 + B2/B3/B4 fp32 路），存活峰值 3 / 分配 6
            // Cast↑×2（CAST_NONE 无损上行，断链，src/dst 独立）：
            //   执行中: 持有=[B0, B1, B2] → [B1, B2, B3]（峰值 3）
            AscendC::Cast(buf_[2].Get<float>(), buf_[0].Get<T>(), AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(buf_[3].Get<float>(), buf_[1].Get<T>(), AscendC::RoundMode::CAST_NONE, count);
            // VF 链（fp32 域）：执行中: 持有=[B2, B3, B4]（存活峰值 3）；执行后: 持有=[B4]
            ComputeTile(reinterpret_cast<__ubuf__ float*>(buf_[4].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[2].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[3].Get<float>().GetPhyAddr()), count);
            // Cast↓（fp16/bf16 统一 CAST_RINT 舍入到最近偶数；4 参数形式 roundMode 不可省略）：
            //   执行中: 持有=[B4, B5]；执行后: 持有=[B5]
            AscendC::Cast(buf_[5].Get<T>(), buf_[4].Get<float>(), AscendC::RoundMode::CAST_RINT, count);
        }
    }

    // -----------------------------------------------------------------------
    // CopyOut — small-tail UB → GM（MTE3）：与 CopyIn 对称的行组多行整块
    // DataCopyPad（3 参形式无 padParams）（DESIGN-BRANCH-0.md §5.4）
    //   blockCount = gRows、blockLen = d·S（只写有效字节，哑元丢弃）；
    //   UB 侧 srcStride = 0（硬件按 CeilAlign(d·S,32) 推进跳过行尾哑元），
    //   GM 侧 dstStride = 0（y 行连续：y 第 r 行首 r·d，blockLen 恰好首尾相接）。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOut(int64_t r0, int64_t gRows)
    {
        const int64_t d = td_->half_dim;
        AscendC::DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(gRows);
        params.blockLen = static_cast<uint32_t>(d * static_cast<int64_t>(sizeof(T))); // 有效字节，不要求 32B 对齐
        params.srcStride = 0; // UB 侧 gap（32B 单位）=0：硬件按 CeilAlign(d·S,32) 推进，跳过行尾哑元
        params.dstStride = 0; // GM 侧 gap（byte）=0：y 行连续
        params.rsv = 0;
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopyPad(gmY_[r0 * d], buf_[2].Get<T>(), params); // B2 = y(fp32)
        } else {
            AscendC::DataCopyPad(gmY_[r0 * d], buf_[5].Get<T>(), params); // B5 = y(T)
        }
    }

    // -----------------------------------------------------------------------
    // CopyIn — big-tail GM → UB（MTE2）：行内 flat 分段搬运
    // （DESIGN-BRANCH-1.md §5.2）
    //   x1Off / x2Off = gate / up 段首全局元素偏移（§2 公式(3)：r·2d+s·TE /
    //   r·2d+d+s·TE），segLen = 段长（主段 TE / 尾段 d%TE，运行期推导）。
    // 每段两笔 blockCount=1 的 DataCopyPad：两半区段在 GM 相隔 (d − segLen)
    // 元素（尾段间隔与主段不同、字节量不保证 32B 整数倍），不合批 blockCount=2
    // + srcStride 单笔搬运；blockLen = segLen·S 有效字节（不要求 32B 对齐）；
    // isPad=false —— 段尾 dummy 不参与计算，由 VF 链 UpdateMask 按有效元素数
    // 屏蔽。同步（§5.5 事件 1/2）：!isFirst 时 WaitFlag(V_MTE2) 跨段 WAR
    // （上段 V 读毕 B0/B1 → 本段 MTE2 才可覆写，首轮跳过）；MTE2 写完 B0/B1
    // 后 SetFlag(MTE2_V) RAW 发布。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyIn(int64_t x1Off, int64_t x2Off, int64_t segLen, bool isFirst)
    {
        if (!isFirst) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMte2_); // WAR 反向同步(跨段)，§5.5 事件 2
        }
        AscendC::DataCopyExtParams copParams;
        copParams.blockCount = 1; // 单块 [1, 4095]
        copParams.blockLen = static_cast<uint32_t>(segLen *
                                                   static_cast<int64_t>(sizeof(T))); // 有效字节，非 32B 对齐合法
        copParams.srcStride = 0; // GM 侧 gap=0（单块无块间空隙）
        copParams.dstStride = 0; // UB 侧 gap=0
        copParams.rsv = 0;       // 必须显式填 0
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // MTE2 写 B0 ← GM gate 段；MTE2 写 B1 ← GM up 段（GlobalTensor::operator[] 元素索引）
        AscendC::DataCopyPad(buf_[0].Get<T>(), gmX_[x1Off], copParams, padParams);
        AscendC::DataCopyPad(buf_[1].Get<T>(), gmX_[x2Off], copParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2ToV_); // RAW 正向同步，§5.5 事件 1
    }

    // -----------------------------------------------------------------------
    // Compute — big-tail per-段计算（V pipe；DESIGN-BRANCH-1.md §5.3）
    //   segLen = 段元素数（与 VF 链 UpdateMask 同一计数，尾段任意长度均被
    //   掩码精确处理，段尾 UB 残留不被读写）：fp32 直连共享入口 ComputeTile
    // （B0/B1 → B2）；fp16/bf16 走共享子链 Cast↑×2 → ComputeTile → Cast↓
    // （B0/B1 → B2/B3 → B4 → B5，Kernel.md「Cast 上/下行组织」）。
    // 同步（§5.5 事件 1/2/3/4）：WaitFlag(MTE2_V) RAW 段内（等 MTE2 写完
    // B0/B1）；!isFirst 时 WaitFlag(MTE3_V) 跨段 WAR（上段 MTE3 读毕 y buffer，
    // 首轮跳过）；!isLast 时 SetFlag(V_MTE2) WAR 发布（释放 B0/B1 给下段
    // MTE2，末段跳过）；SetFlag(V_MTE3) RAW 发布（V 写完 y buffer → 本段
    // CopyOut 消费）。
    // -----------------------------------------------------------------------
    __aicore__ inline void Compute(int64_t segLen, bool isFirst, bool isLast)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2ToV_); // RAW：等 MTE2 写完 B0/B1，§5.5 事件 1
        if (!isFirst) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                evMte3ToV_); // WAR(跨段)：上段 MTE3 读毕 y buffer，§5.5 事件 4
        }
        const uint32_t count = static_cast<uint32_t>(segLen);
        if constexpr (std::is_same_v<T, float>) {
            // fp32: VF 链直连 T buffer —— 执行中持有 [B0, B1, B2] = P_fp32 峰值 3
            ComputeTile(reinterpret_cast<__ubuf__ float*>(buf_[2].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[0].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[1].Get<float>().GetPhyAddr()), count);
        } else {
            // fp16/bf16: Cast 断链 ×3 共享子链 —— 存活峰值 3 / 分配 P=6
            AscendC::Cast(buf_[2].Get<float>(), buf_[0].Get<T>(), AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(buf_[3].Get<float>(), buf_[1].Get<T>(), AscendC::RoundMode::CAST_NONE, count);
            ComputeTile(reinterpret_cast<__ubuf__ float*>(buf_[4].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[2].Get<float>().GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(buf_[3].Get<float>().GetPhyAddr()), count);
            AscendC::Cast(buf_[5].Get<T>(), buf_[4].Get<float>(), AscendC::RoundMode::CAST_RINT, count);
        }
        if (!isLast) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMte2_); // WAR 发布(跨段)：释放 B0/B1，§5.5 事件 2
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMte3_); // RAW 发布：y buffer 就绪，§5.5 事件 3
    }

    // -----------------------------------------------------------------------
    // CopyOut — big-tail UB → GM（MTE3）：单路径 dense 段写回
    // （DESIGN-BRANCH-1.md §5.4，与 CopyIn 对称）
    //   yOff = r·d + s·TE（y 段首全局元素偏移）；blockLen = segLen·S 有效字节
    // （非 32B 对齐合法——UB→GM 统一 DataCopyPad、不用普通 DataCopy，UB 侧
    // 不足 32B 由硬件补 dummy 读出、写 GM 时自动丢弃）。同步（§5.5 事件
    // 3/4）：WaitFlag(V_MTE3) RAW 段内（等 V 写完 y buffer）；!isLast 时
    // SetFlag(MTE3_V) WAR 发布（释放 y buffer 给下段 V，末段跳过）。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOut(int64_t yOff, int64_t segLen, bool isLast)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
            evVtoMte3_); // RAW：等 V 写完 y buffer（fp32: B2 / fp16·bf16: B5）
        AscendC::DataCopyExtParams copParams;
        copParams.blockCount = 1;                                                             // 单块
        copParams.blockLen = static_cast<uint32_t>(segLen * static_cast<int64_t>(sizeof(T))); // 有效字节
        copParams.srcStride = 0;                                                              // UB 侧 gap=0
        copParams.dstStride = 0; // GM 侧 gap=0（段间 GM 连续由 yOff 步进保证）
        copParams.rsv = 0;       // 必须显式填 0
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopyPad(gmY_[yOff], buf_[2].Get<T>(), copParams); // B2 = y(fp32)
        } else {
            AscendC::DataCopyPad(gmY_[yOff], buf_[5].Get<T>(), copParams); // B5 = y(T)
        }
        if (!isLast) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evMte3ToV_); // WAR 发布(跨段)，§5.5 事件 4
        }
    }

    // ---- shared members ----
    AscendC::TPipe pipe_;                               // TBuf 静态分配
    const FatreluMulTilingData* td_ = nullptr;          // 全局非模板 TilingData（TilingData.md）
    AscendC::GlobalTensor<T> gmX_;                      // x：gate_up 拼接张量（行模型 (batch, 2d)）
    AscendC::GlobalTensor<T> gmThreshold_;              // threshold：单元素标量 Tensor（GM 直读）
    AscendC::GlobalTensor<T> gmY_;                      // y：(batch, d)
    T threshold_ = static_cast<T>(0);                   // 标量阈值（寄存器参与比较，不占 UB）
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[6]; // per-dtype P 份（fp32 用 B0-B2 / fp16·bf16 用 B0-B5）

    // ---- small-tail path members（DESIGN-BRANCH-0.md §5.1）----
    int64_t rowStart_ = 0;        // 本核行区间起点 = blockIdx·rows_former + min(blockIdx, rows_tail_core)
    int64_t rowCount_ = 0;        // 本核行数 = rows_former + (blockIdx < rows_tail_core ? 1 : 0)
    int64_t rowPitchElems_ = 0;   // UB 内行距（元素）= CeilAlign(d·S, 32) / S
    int64_t rowsPerGroupEff_ = 0; // pitch 安全有效组行数 = min(rows_per_group, tileElems/pitchElems, 4095)

    // ---- sync event IDs（DESIGN-BRANCH-0.md §5.5；FetchEventID 托管分配）----
    int32_t evMte2ToV_ = 0; // MTE2_V：组内 RAW（CopyIn 写 B0/B1 → Compute 读）
    int32_t evVtoMte3_ = 0; // V_MTE3：组内 RAW（Compute 写输出 buffer → CopyOut 读）
    int32_t evVtoMte2_ = 0; // V_MTE2：跨组 WAR（Compute 读毕 B0/B1 → 下组 CopyIn 覆写）
    int32_t evMte3ToV_ = 0; // MTE3_V：跨组 WAR（CopyOut 读毕输出 buffer → 下组 Compute 覆写）
};
