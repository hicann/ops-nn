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
// gn_training_update_package/op_kernel/arch35/gn_training_update_kernel.h
// =============================================================================
//
// ROLE: Ascend C kernel implementation for GnTrainingUpdate on arch35
//   (Ascend 950). Real implementation of design/Kernel.md §1/§9 and
//   design/branches/DESIGN-BRANCH-0.md §5 (both tilingKey branches share the
//   same templated class; only the RANK template parameter differs).
//
//   Math chain (fp32 main chain, spec.yaml numerical_stability.fp32_main_chain):
//     mean = sum * invM                                   (VF1)
//     var  = square_sum * invM - mean^2                   (VF2a, correction=0;
//             inf 安全补丁：sq·invM=+inf 且 mean² 上溢 +inf（mean 有限，fp64
//             golden 中 mean² 恒有限 → var=+inf）时 fp32 直算得 NaN，
//             按 golden 语义修补为 +inf；mean=±inf 时 golden 亦 NaN，不修补)
//     rstd = 1 / sqrt(var + epsilon)                      (VF2b)
//     仿射路径（hasAffine=1，按组自适应乘法序，避免中间量上溢：
//             fp32 主链任何固定序都有溢出盲区（L1_066：x=-FLT_MAX、|scale|<1
//             时 x·rstd 先溢；L1_336：x=0、scale=±FLT_MAX 时 rstd·scale 先溢），
//             fp64 golden 终值仍有界。按 rstd 量级选序可证安全：
//             rstd≤1 → t=xm·rstd 必不溢（A 序）；rstd>1 → t=xm·scale，
//             要么不溢、要么真值本身 >FLT_MAX（golden cast 同为 ±inf））:
//     xm   = x - mean                                     (VF3a)
//     t    = xm * (rstd>1 ? scale : rstd)                 (VF3b)
//     h    =      (rstd>1 ? rstd : scale)                 (VF3b，覆写 rstd 槽)
//     y    = t * h + offset                               (VF4)
//     纯归一化路径（hasAffine=0，无 scale 吸收溢出，±inf 与 golden 一致）:
//     yhat = (x - mean) * rstd                            (VF3)
//   Bypass outputs: batch_mean = mean, batch_variance = var (no epsilon),
//   written per covered (n,g) group (1 element each), y written whole-tile.
//
//   Buffer organization (DESIGN-BRANCH-0.md §3/§4, kPhysNodes = P = 3):
//     B0: x(fp16) -> square_sum -> var -> rstd -> offset -> y(fp16)
//     B1: xf(fp32 main chain) -> yhat -> y(fp32)
//     B2: sum -> mean -> scale
//
//   NOTE (design reconciliation): DESIGN-BRANCH-0.md §5.5 sync table row 4
//   requires the batch_variance CopyOut (MTE3 reads the var slot) to complete
//   BEFORE rstd overwrites the same slot (MTE3_V event pair, "VF2 续 rstd").
//   A single asc_vf_call cannot interleave an MTE3 read mid-chain, so VF2 is
//   issued as two calls at the StoreAlign-var boundary:
//     asc_vf_call<VarVF>  (var -> B0) -> V_MTE3 -> CopyOut batch_variance
//     -> MTE3_V -> asc_vf_call<RstdVF> (rstd in-place B0).
//   Total chain length is unchanged (3 + 3 = 6 <= 7).
// =============================================================================

#pragma once
#include <type_traits>
#include "kernel_operator.h"                  // Ascend C core framework
#include "gn_training_update_tiling_struct.h" // GnTrainingUpdateTilingData<RANK>, MAX_INPUT_SLOTS, PHYS_NODES
#include "gn_training_update_struct.h"        // GN_TRAINING_UPDATE_RANK_4/8

// ---------------------------------------------------------------------------
// VF forward declarations (fp32 main chain only; signatures per API.md §1,
// VarRstdVF split into VarVF + RstdVF per the note above).
// ---------------------------------------------------------------------------
__simd_vf__ inline void MeanVF(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, float invM, uint32_t count,
                               uint32_t oneRepeatSize, uint16_t repeatTimes);
__simd_vf__ inline void VarVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr, float invM,
                              uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);
__simd_vf__ inline void RstdVF(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, float epsilon, uint32_t count,
                               uint32_t oneRepeatSize, uint16_t repeatTimes);
__simd_vf__ inline void NormalizeVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr,
                                    __ubuf__ float* src2Addr, uint32_t count, uint32_t oneRepeatSize,
                                    uint16_t repeatTimes);
__simd_vf__ inline void SubMeanVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr,
                                  uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);
__simd_vf__ inline void AdaptiveMulVF(__ubuf__ float* dstAddr, __ubuf__ float* dstHAddr, __ubuf__ float* srcXmAddr,
                                      __ubuf__ float* srcRstdAddr, __ubuf__ float* srcScaleAddr, uint32_t count,
                                      uint32_t oneRepeatSize, uint16_t repeatTimes);
__simd_vf__ inline void AffineVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr,
                                 uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes);
__simd_vf__ inline void AffineNoOffsetVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, uint32_t count,
                                         uint32_t oneRepeatSize, uint16_t repeatTimes);

// ---------------------------------------------------------------------------
// Generic helpers (broadcast-standard-kernel-template.md §3;
// CalcOffset returns ELEMENT counts — gm[] index is an element index).
// ---------------------------------------------------------------------------
__aicore__ inline void GetCoreRange(int64_t coreId, int64_t tilesMain, int64_t coresTail, int64_t& start, int64_t& end)
{
    if (coreId < coresTail) {
        start = coreId * (tilesMain + 1);
        end = start + tilesMain + 1;
    } else {
        start = coresTail * (tilesMain + 1) + (coreId - coresTail) * tilesMain;
        end = start + tilesMain;
    }
}

__aicore__ inline int64_t GetUBSplitRange(int64_t aOOff, int64_t aO, int64_t aI, int64_t aITail)
{
    return (aOOff == aO - 1) ? aITail : aI;
}

__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* maxBroShape, int64_t rank, int64_t splitAxis,
                                            int64_t aI, int64_t aO, int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++) {
        effCoord[d] = 0;
    }
    if (aO <= 0) {
        return false;
    }
    int64_t aOOff = flat % aO;
    int64_t outer = flat / aO;
    for (int64_t d = splitAxis - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[splitAxis] = aOOff * aI;
    return true;
}

__aicore__ inline int64_t CalcOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) {
        offset += effCoord[d] * strides[d];
    }
    return offset; // elements
}

// PadTo — UB 最内维占用向上对齐到 unit 的倍数。实测硬件约束（plog errcode
// 80 "MTE non-atomic instruction address is not aligned"）：
//   (1) NDDMA dim0（最内循环）loopSize 须为 8 元素倍数（dtype 无关，广播维
//       srcStride=0 也不例外）：case00002/03/04 + diagC（M=8）PASS；
//       case00005 fp32 M=4、diagE fp16 M=4 均 VEC_ERROR；
//   (2) MTE3（DataCopyPad UB→GM）UB 侧地址须 32B 对齐：fp32 行距 32B 全过、
//       fp16 行距 16B（diagE/diagF，Pad8(=8 元素)=16B）均 VEC_ERROR。
// 综合：UB 行距粒度统一为 32B（fp32=8 元素 / fp16=16 元素），pad 通道由广播
// stride=0 重复填充（统计量）或保留垃圾（x/y，不写出），VF 按 padded
// count 处理，数学结果不受影响。
__aicore__ inline constexpr int64_t PadTo(int64_t x, int64_t unit) { return (x + unit - 1) / unit * unit; }

// ===========================================================================
// class GnTrainingUpdateKernel<T, RANK>
//   T    — x/y dtype (half / float), injected via DTYPE_X
//   RANK — GN_TRAINING_UPDATE_RANK_4 (tilingKey 0) / GN_TRAINING_UPDATE_RANK_8
//          (tilingKey 1, defensive); identical structure, only array dims and
//          ND differ (ND = min(RANK, 5); RANK > 5 uses the outerIters path).
// ===========================================================================
template <typename T, int64_t RANK>
class GnTrainingUpdateKernel {
    static constexpr int64_t kMaxRank = GN_TRAINING_UPDATE_RANK_8;
    static constexpr int64_t kMaxNddmaDims = 5;
    static constexpr int64_t ND = (RANK <= kMaxNddmaDims) ? RANK : kMaxNddmaDims;
    static constexpr bool kNeedCast = !std::is_same_v<T, float>;
    static constexpr uint32_t kVlF32 = AscendC::GetVecLen() / sizeof(float);
    // UB 行距粒度（元素）：32B（MTE3 UB 地址 32B 对齐 + NDDMA dim0 8 元素粒度，
    // 见 PadTo 注释）；fp32=8 / fp16=16
    static constexpr int64_t kPadElems = 32 / static_cast<int64_t>(sizeof(T));
    __aicore__ inline constexpr int64_t PadV(int64_t x) const { return (x + kPadElems - 1) / kPadElems * kPadElems; }

    // TBuf slot indices (DESIGN-BRANCH-0.md §4 buffer table)
    static constexpr int64_t kB0 = 0; // x(fp16)/square_sum/var/rstd/offset/y(fp16)
    static constexpr int64_t kB1 = 1; // xf/yhat/y (fp32 main chain)
    static constexpr int64_t kB2 = 2; // sum/mean/scale
    // Consumed input slots (5/6 = mean/variance: IR reserved, never consumed)
    static constexpr int64_t kInX = 0;
    static constexpr int64_t kInSum = 1;
    static constexpr int64_t kInSqSum = 2;
    static constexpr int64_t kInScale = 3;
    static constexpr int64_t kInOffset = 4;
    // Output slots
    static constexpr int64_t kOutY = 0;
    static constexpr int64_t kOutBatchMean = 1;
    static constexpr int64_t kOutBatchVar = 2;

    AscendC::TPipe pipe_;                                        // UB buffer + events
    const GnTrainingUpdateTilingData<RANK>* td_;                 // tiling data (host filled)
    AscendC::GlobalTensor<T> gmX_;                               // slot 0: x
    AscendC::GlobalTensor<float> gmSum_;                         // slot 1: sum
    AscendC::GlobalTensor<float> gmSqSum_;                       // slot 2: square_sum
    AscendC::GlobalTensor<float> gmScale_;                       // slot 3: scale (hasAffine=1)
    AscendC::GlobalTensor<float> gmOffset_;                      // slot 4: offset (hasAffine=1)
    AscendC::GlobalTensor<T> gmY_;                               // out 0: y
    AscendC::GlobalTensor<float> gmBatchMean_;                   // out 1: batch_mean
    AscendC::GlobalTensor<float> gmBatchVar_;                    // out 2: batch_variance
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES]; // B0/B1/B2, perBufBytes each
    const __gm__ T* xRaw_ = nullptr;                             // x 标量兜底直读指针（CopyInXSmallL）
    AscendC::MultiCopyParams<T, ND> nddmaX_;                     // NDDMA params: x (dtype T)
    AscendC::MultiCopyParams<float, ND> nddmaStats_[4];          // NDDMA params: slots 1..4 (fp32)
    int64_t nddmaOuterIters_[kMaxNddmaDims];                     // per consumed slot (RANK > 5 only)
    int64_t nddmaDims_;                                          // min(RANK - axis, ND)

public:
    // -----------------------------------------------------------------------
    // Init — GM binding (slots 5/6 reserved: not bound; slots 3/4 guarded by
    // hasAffine), 3 TBuf allocation, NDDMA static params precompute
    // (DESIGN-BRANCH-0.md §5.1).
    // -----------------------------------------------------------------------
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const GnTrainingUpdateTilingData<RANK>* td)
    {
        td_ = td;
        gmX_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputs[kInX]));
        xRaw_ = reinterpret_cast<const __gm__ T*>(inputs[kInX]);
        gmSum_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputs[kInSum]));
        gmSqSum_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputs[kInSqSum]));
        if (td_->hasAffine == 1) {
            gmScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputs[kInScale]));
            if (td_->hasOffset == 1) {
                gmOffset_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputs[kInOffset]));
            }
        }
        gmY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputs[kOutY]));
        gmBatchMean_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputs[kOutBatchMean]));
        gmBatchVar_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputs[kOutBatchVar]));

        for (int64_t i = 0; i < PHYS_NODES; i++) {
            pipe_.InitBuffer(buf_[i], td_->perBufBytes);
        }

        const int64_t k = td_->split.axis;
        nddmaDims_ = (RANK - k <= ND) ? (RANK - k) : ND;
        FillNddma<T>(nddmaX_, kInX);
        FillNddma<float>(nddmaStats_[0], kInSum);
        FillNddma<float>(nddmaStats_[1], kInSqSum);
        FillNddma<float>(nddmaStats_[2], kInScale);
        FillNddma<float>(nddmaStats_[3], kInOffset);
    }

    // -----------------------------------------------------------------------
    // Process — per-tile pipeline (DESIGN-BRANCH-0.md §5.3/§5.5):
    //   CopyInBrc -> [Cast] -> VF1 -> CopyOut batch_mean -> VF2a ->
    //   CopyOut batch_variance -> VF2b -> VF3 -> [CopyInBrc scale/offset ->
    //   VF4] -> [Cast] -> CopyOut y.
    // -----------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        // N=0 空 batch：totalTiles=0 零循环短路，三输出空 tensor（Kernel.md §9.3）
        if (td_->multicore.totalTiles == 0) {
            return;
        }
        event_t evMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        event_t evVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        event_t evVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        event_t evMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        event_t evMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start = 0;
        int64_t end = 0;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tilesMain, td_->multicore.coresTail, start, end);
        // UB 侧 innerCount：最内维按 Pad8 计（NDDMA dim0 粒度约束，见 Pad8 注释）；
        // GM 侧 innerCountGm 保持真实元素数（CopyOutY 只写回真实元素）
        int64_t innerCount = 1;
        int64_t innerCountGm = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++) {
            innerCount *= (d == RANK - 1) ? PadV(td_->maxBroShape[d]) : td_->maxBroShape[d];
            innerCountGm *= td_->maxBroShape[d];
        }

        int64_t coord[kMaxRank] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t aISeg = GetUBSplitRange(flat % td_->split.aO, td_->split.aO, td_->split.aI, td_->split.aITail);
            // 切分轴即最内维时，本 tile 的最内段长为 aISeg（尾块任意值），同样 Pad8
            const bool splitLast = (td_->split.axis == RANK - 1);
            int64_t count = splitLast ? PadV(aISeg) : aISeg * innerCount; // UB/VF 元素数
            int64_t countGm = aISeg * innerCountGm;                       // GM 真实元素数
            FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, td_->split.axis, td_->split.aI, td_->split.aO, coord);
            uint16_t rep = AscendC::CeilDivision(count, kVlF32);

            // 跨 tile WAR：上轮 CopyOut(MTE3) 读 B0/B1/B2 完毕 → 本轮 MTE2 覆写
            if (flat != start) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
            }
            // 跨 tile WAR：上轮 V 读（仿射 scale/offset 等）完毕 → 本轮 MTE2 覆写
            if (flat != start) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
            }

            // S1: CopyInBrc x（NDDMA 全稠密）
            // 执行前: 持有=[]      执行中: 持有=[B0/B1]   执行后: 持有=[B0/B1]
            if constexpr (kNeedCast) {
                CopyInBrc(coord, kInX, kB0, aISeg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                // S2: Cast 断点 1：B0(fp16) → B1(fp32)（fp32 主链起点）
                // 执行前: 持有=[B0]  执行中: 持有=[B0,B1]  执行后: 持有=[B1]（B0 释放）
                AscendC::Cast(buf_[kB1].template Get<float>(), buf_[kB0].template Get<T>(),
                              AscendC::RoundMode::CAST_NONE, count);
            } else {
                // fp32 路径：x 直接入 B1，无 Cast 断点
                CopyInBrc(coord, kInX, kB1, aISeg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
            }

            // S3: CopyInBrc sum → B2（NDDMA 随路广播：M 维 GM stride=0）
            // 执行前: 持有=[B1]    执行中: 持有=[B1,B2]   执行后: 持有=[B1,B2]
            CopyInBrc(coord, kInSum, kB2, aISeg);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
            // S4: VF1 mean = sum · invM（原地 B2；对应 μ = sum/M）
            // 执行前: 持有=[B1,B2] 执行中: 持有=[B1,B2]   执行后: 持有=[B1,B2]（B2 现为 mean）
            asc_vf_call<MeanVF>((__ubuf__ float*)buf_[kB2].template Get<float>().GetPhyAddr(),
                                (__ubuf__ float*)buf_[kB2].template Get<float>().GetPhyAddr(), td_->invM,
                                static_cast<uint32_t>(count), kVlF32, rep);
            // S5: CopyOut batch_mean ← B2（旁路输出 1，逐 (n,g) 写 1 元素）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
            CopyOutStats(kOutBatchMean, kB2, coord, aISeg);

            // S6: CopyInBrc square_sum → B0（槽 0 复用；上轮角色 x 被 Cast 读过 → V_MTE2）
            // 执行前: 持有=[B1,B2] 执行中: 持有=[B0,B1,B2] 执行后: 持有=[B0,B1,B2]
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
            CopyInBrc(coord, kInSqSum, kB0, aISeg);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
            // S7a: VF2a var = square_sum·invM − mean²（原地 B0；correction=0，负 var 不 clamp）
            // 执行前: 持有=[B0,B1,B2] 执行中: 持有=[B0,B1,B2] 执行后: 持有=[B0,B1,B2]（B0 现为 var）
            asc_vf_call<VarVF>((__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                               (__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                               (__ubuf__ float*)buf_[kB2].template Get<float>().GetPhyAddr(), td_->invM,
                               static_cast<uint32_t>(count), kVlF32, rep);
            // S8: CopyOut batch_variance ← B0（旁路输出 2，var 未加 ε，逐 (n,g) 写 1 元素）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
            CopyOutStats(kOutBatchVar, kB0, coord, aISeg);
            if (td_->hasAffine == 1) {
                // S10/S11 的 MTE2 覆写 B2/B0 须等 S5/S8 的 MTE3 读完毕（MTE3_MTE2）
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
            }
            // S7b: VF2b rstd = 1/sqrt(var+ε)（覆写同槽 B0；MTE3 读 var 完毕 → MTE3_V）
            // 执行后: 持有=[B0,B1,B2]（B0 现为 rstd）
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V);
            asc_vf_call<RstdVF>((__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                                (__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(), td_->epsilon,
                                static_cast<uint32_t>(count), kVlF32, rep);

            if (td_->hasAffine == 1) {
                // 仿射路径（按组自适应乘法序：y = [xm·g]·h + offset，
                // rstd>1 → g=scale/h=rstd（C 序），否则 g=rstd/h=scale（A 序）；
                // 安全性证明见文件头注释。两序之选为逐元素 Select，混合组 tile 安全）。
                // S9a: VF3a xm = xf − mean → B1（P=3 峰值：B0+B1+B2）
                // 执行前: 持有=[B0,B1,B2] 执行中: 持有=[B0,B1,B2] ← P=3 峰值
                // 执行后: 持有=[B1]（B2 mean 无未来消费者，释放）
                asc_vf_call<SubMeanVF>((__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                       (__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                       (__ubuf__ float*)buf_[kB2].template Get<float>().GetPhyAddr(),
                                       static_cast<uint32_t>(count), kVlF32, rep);
                // S10: CopyInBrc scale → B2（槽位复用；
                //   上轮角色 mean 被 VF3a(V) 读过 → V_MTE2；被 S5(MTE3) 读过 → MTE3_MTE2）
                // 执行前: 持有=[B1]  执行中: 持有=[B1,B2] 执行后: 持有=[B1,B2]
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
                CopyInBrc(coord, kInScale, kB2, aISeg);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                // S11: VF3b 自适应乘序：t = xm·g → B1、h → B0（覆写 rstd 槽）
                // 执行前: 持有=[B0,B1,B2] 执行中: 持有=[B0,B1,B2] 执行后: 持有=[B0,B1,B2]
                asc_vf_call<AdaptiveMulVF>((__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)buf_[kB2].template Get<float>().GetPhyAddr(),
                                           static_cast<uint32_t>(count), kVlF32, rep);
                if (td_->hasOffset == 1) {
                    // S12: CopyInBrc offset → B2（槽位复用；scale 被 VF3b(V) 读过 → V_MTE2）
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
                    CopyInBrc(coord, kInOffset, kB2, aISeg);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                    // S13: VF4 y = t·h + offset → B1（MulAddDst 硬件融合）
                    // 执行后: 持有=[B1]（B0 h / B2 offset 无未来消费者，释放）
                    asc_vf_call<AffineVF>((__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                          (__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                                          (__ubuf__ float*)buf_[kB2].template Get<float>().GetPhyAddr(),
                                          static_cast<uint32_t>(count), kVlF32, rep);
                } else {
                    // S13': offset 缺失（golden: y = t·h，offset 恒等 0）→ B1；
                    // B2 不覆写（scale 已无消费者）
                    asc_vf_call<AffineNoOffsetVF>((__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                                  (__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                                                  static_cast<uint32_t>(count), kVlF32, rep);
                }
            } else {
                // S9: VF3 ŷ = (xf − mean) · rstd → B1（纯归一化；无 scale 吸收
                // 溢出，fp32 中间 ±inf 与 fp64 golden cast 回 fp32 的 ±inf 一致）
                // 执行前: 持有=[B0,B1,B2] 执行中: 持有=[B0,B1,B2] ← P=3 峰值
                // 执行后: 持有=[B1]（B0 rstd / B2 mean 无未来消费者，释放）
                asc_vf_call<NormalizeVF>((__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                         (__ubuf__ float*)buf_[kB1].template Get<float>().GetPhyAddr(),
                                         (__ubuf__ float*)buf_[kB2].template Get<float>().GetPhyAddr(),
                                         (__ubuf__ float*)buf_[kB0].template Get<float>().GetPhyAddr(),
                                         static_cast<uint32_t>(count), kVlF32, rep);
            }
            // hasAffine=0：y = ŷ（纯归一化），跳过 S10–S13（Kernel.md §9.4）

            if constexpr (kNeedCast) {
                // S13: Cast 断点 2：B1(fp32) → B0(fp16)（NPU 饱和语义）
                // 执行前: 持有=[B1] 执行中: 持有=[B0,B1] 执行后: 持有=[B0]（B1 释放）
                AscendC::Cast(buf_[kB0].template Get<T>(), buf_[kB1].template Get<float>(),
                              AscendC::RoundMode::CAST_RINT, count);
                // S14: CopyOut y ← B0
                // 执行前: 持有=[B0] 执行中: 持有=[B0] 执行后: 持有=[]
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
                CopyOutY(kB0, coord, countGm);
            } else {
                // fp32 路径：CopyOut y ← B1（无 Cast 断点 2）
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
                CopyOutY(kB1, coord, countGm);
            }

            // 跨 tile 反向同步：末轮跳过 SetFlag
            if (flat != end - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>(); // 退出前排空所有流水线（API.md §2）
    }

private:
    // -----------------------------------------------------------------------
    // FillNddma — NDDMA static params for one consumed slot (5 fields all
    // initialized: loopSize/loopSrcStride/loopDstStride/loopLpSize=0/
    // loopRpSize=0; loopSize of the split axis is patched per tile in
    // CopyInBrc). NDDMA dim[0] = innermost (nd = RANK-1-d). Strides are in
    // ELEMENTS; broadcast dims carry GM stride 0 (along-the-way expansion).
    // -----------------------------------------------------------------------
    template <typename DT>
    __aicore__ inline void FillNddma(AscendC::MultiCopyParams<DT, ND>& p, int64_t slot)
    {
        const int64_t* dstShape = td_->maxBroShape;
        const int64_t k = td_->split.axis;
        int64_t inner = 1;
        int64_t nd = 0;
        for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
            // 最内维 loopSize 按 Pad8 虚增（NDDMA dim0 粒度约束，见 Pad8 注释）；
            // 统计量槽该维 srcStride=0，虚增仅重复读同一 GM 元素，安全
            p.loopInfo.loopSize[nd] = (d == k) ? 0 : ((d == RANK - 1) ? PadV(dstShape[d]) : dstShape[d]);
            p.loopInfo.loopSrcStride[nd] = td_->inputStrides[slot][d];
            p.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
            p.loopInfo.loopLpSize[nd] = 0;
            p.loopInfo.loopRpSize[nd] = 0;
            inner *= (d == k) ? ((d == RANK - 1) ? PadV(td_->split.aI) : td_->split.aI) : p.loopInfo.loopSize[nd];
            nd++;
        }
        for (; nd < ND; nd++) {
            p.loopInfo.loopSize[nd] = 1;
            p.loopInfo.loopSrcStride[nd] = 0;
            p.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
            p.loopInfo.loopLpSize[nd] = 0;
            p.loopInfo.loopRpSize[nd] = 0;
        }
        p.constantValue = DT(0);
        nddmaOuterIters_[slot] = 1;
        for (int64_t d = k; d < RANK - nddmaDims_; d++) {
            nddmaOuterIters_[slot] *= (d == k) ? td_->split.aI : dstShape[d];
        }
    }

    // -----------------------------------------------------------------------
    // CopyInBrc — GM→UB NDDMA copy with along-the-way broadcast
    // (DESIGN-BRANCH-0.md §5.2). slot 0 = x (dtype T), slots 1..4 = fp32
    // statistics/affine. Slots 5/6 (mean/variance) never arrive here.
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInBrc(const int64_t* coord, int64_t slot, int64_t bufIdx, int64_t aISeg)
    {
        if (slot == kInX) {
            // x 为唯一全稠密输入（MergeAxes 后 tile 在 GM 连续）。最内段非 8
            // 元素倍数时走 CopyInXSmallL（NDDMA Pad8 虚增 dim0 + tensor 末尾
            // tile 标量兜底，见该函数注释）；统计量槽 inner 维 srcStride=0，
            // Pad8 虚增只是重复读同一 GM 元素，安全，始终走 NDDMA。
            const int64_t innerSeg = (td_->split.axis == RANK - 1) ? aISeg : td_->maxBroShape[RANK - 1];
            if (innerSeg % kPadElems != 0) {
                CopyInXSmallL(coord, bufIdx, aISeg);
                return;
            }
            CopyInBrcTyped<T>(coord, slot, bufIdx, aISeg, nddmaX_, gmX_);
        } else if (slot == kInSum) {
            CopyInStats(coord, slot, bufIdx, aISeg, nddmaStats_[0], gmSum_);
        } else if (slot == kInSqSum) {
            CopyInStats(coord, slot, bufIdx, aISeg, nddmaStats_[1], gmSqSum_);
        } else if (slot == kInScale) {
            CopyInStats(coord, slot, bufIdx, aISeg, nddmaStats_[2], gmScale_);
        } else {
            CopyInStats(coord, slot, bufIdx, aISeg, nddmaStats_[3], gmOffset_);
        }
    }

    // -----------------------------------------------------------------------
    // CopyInStats — 统计量/仿射槽（fp32）GM→UB 拷贝。
    //   inner 维 srcStride==0（广播）时 PadV 虚增仅重复读同一 GM 元素，安全，
    //   始终走 NDDMA；srcStride==1（折叠 squeeze 成 1D，如 [G]/[N]）且尺寸非
    //   kPadElems 倍数时，PadV 虚增将越出 tensor 末尾——与 x 同规则处理：
    //   非末尾 tile 走 NDDMA（过读落在 tensor 内、pad 通道不被消费），含 tensor
    //   末元素的 tile 走 CopyInStatsSmallL 标量兜底（SAFE.01 修复）。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInStats(const int64_t* coord, int64_t slot, int64_t bufIdx, int64_t aISeg,
                                       const AscendC::MultiCopyParams<float, ND>& base,
                                       AscendC::GlobalTensor<float>& gm)
    {
        const int64_t k = td_->split.axis;
        const int64_t innerStride = 0; // SAFE01-DEBUG: 强制不转向
        if (innerStride == 0) {
            CopyInBrcTyped<float>(coord, slot, bufIdx, aISeg, base, gm);
            return;
        }
        const int64_t innerSeg = (k == RANK - 1) ? aISeg : td_->maxBroShape[RANK - 1];
        if (innerSeg % kPadElems == 0) {
            CopyInBrcTyped<float>(coord, slot, bufIdx, aISeg, base, gm);
            return;
        }
        const int64_t off = CalcOffset(coord, td_->inputStrides[slot], RANK); // elements
        int64_t countGm = aISeg;
        for (int64_t d = k + 1; d < RANK; d++) {
            countGm *= td_->maxBroShape[d];
        }
        int64_t numel = 1;
        for (int64_t d = 0; d < RANK; d++) {
            numel *= td_->inputShapes[slot][d];
        }
        if (off + countGm < numel) {
            CopyInBrcTyped<float>(coord, slot, bufIdx, aISeg, base, gm);
            return;
        }
        CopyInStatsSmallL(coord, slot, bufIdx, aISeg, gm, off, countGm);
    }

    // -----------------------------------------------------------------------
    // CopyInStatsSmallL — 统计量/仿射槽 tensor 末尾 tile 的标量兜底
    // （结构同 CopyInXSmallL：跨 tile WAR 前置 MTE3_S/V_S 同步，尾部 S_V 同步）。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInStatsSmallL(const int64_t* coord, int64_t slot, int64_t bufIdx, int64_t aISeg,
                                             AscendC::GlobalTensor<float>& gm, int64_t off, int64_t countGm)
    {
        const int64_t k = td_->split.axis;
        event_t evMte3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evMte3S);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evMte3S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evVS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVS);
        const __gm__ float* src = reinterpret_cast<const __gm__ float*>(gm.GetPhyAddr());
        __ubuf__ float* dst = reinterpret_cast<__ubuf__ float*>(buf_[bufIdx].template Get<float>().GetPhyAddr());
        if (k == RANK - 1) {
            for (int64_t i = 0; i < countGm; i++) {
                dst[i] = src[off + i];
            }
        } else {
            const int64_t last = td_->maxBroShape[RANK - 1];
            const int64_t rows = countGm / last;
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t m = 0; m < last; m++) {
                    dst[r * PadV(last) + m] = src[off + r * last + m];
                }
            }
        }
        event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(evSV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(evSV);
    }

    // -----------------------------------------------------------------------
    // CopyInXSmallL — x 全稠密、最内段非 8 元素倍数的通用路径。
    //   (1) 非 tensor 末尾 tile：直接 NDDMA（FillNddma/CopyInBrcTyped 已把
    //       dim0 loopSize 虚增为 Pad8；srcStride=1 的过读 ≤7 元素落在 tensor
    //       内的下一行/组，读入的是 UB pad 通道，VF 处理但 CopyOutY/
    //       CopyOutStats 均不消费 pad 通道，数学无害）。
    //   (2) 含 tensor 末元素的 tile（每次 launch 至多一个）：过读将越出
    //       tensor 末尾，改标量逐元素按 Pad8 布局拷贝（慢但形式上零越界；
    //       切分轴即最内维时 UB 为整段连续布局，pad 在尾部）。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInXSmallL(const int64_t* coord, int64_t bufIdx, int64_t aISeg)
    {
        const int64_t k = td_->split.axis;
        const int64_t off = CalcOffset(coord, td_->inputStrides[kInX], RANK); // elements
        int64_t countGm = aISeg;
        for (int64_t d = k + 1; d < RANK; d++) {
            countGm *= td_->maxBroShape[d];
        }
        int64_t xNumel = 1;
        for (int64_t d = 0; d < RANK; d++) {
            xNumel *= td_->inputShapes[kInX][d];
        }
        if (off + countGm < xNumel) {
            CopyInBrcTyped<T>(coord, kInX, bufIdx, aISeg, nddmaX_, gmX_);
            return;
        }
        // tensor 末尾 tile：标量兜底
        // 跨 tile WAR 前置：标量管道(S)覆写目标 buf 前，须等上轮 MTE3（CopyOutY/
        // CopyOutStats 读 B0/B1/B2）与 V（VF 读/写同槽）全部完成——循环顶的
        // MTE3_MTE2 / V_MTE2 只约束 MTE2 管道，覆盖不到本路径的 S 管道写
        // （实测：末两 tile y 首段被 x 原值/NaN 污染，窗口随运行滑动）。
        event_t evMte3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evMte3S);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evMte3S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evVS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVS);
        __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(buf_[bufIdx].template Get<T>().GetPhyAddr());
        if (k == RANK - 1) {
            for (int64_t i = 0; i < countGm; i++) {
                dst[i] = xRaw_[off + i];
            }
        } else {
            const int64_t last = td_->maxBroShape[RANK - 1];
            const int64_t rows = countGm / last;
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t m = 0; m < last; m++) {
                    dst[r * PadV(last) + m] = xRaw_[off + r * last + m];
                }
            }
        }
        // 标量管道写 UB → 后续 MTE2/V 读取前做 S_V 同步
        event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(evSV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(evSV);
    }

    template <typename DT>
    __aicore__ inline void CopyInBrcTyped(const int64_t* coord, int64_t slot, int64_t bufIdx, int64_t aISeg,
                                          const AscendC::MultiCopyParams<DT, ND>& base, AscendC::GlobalTensor<DT>& gm)
    {
        const int64_t k = td_->split.axis;
        const int64_t off = CalcOffset(coord, td_->inputStrides[slot], RANK); // elements
        const int64_t* dstShape = td_->maxBroShape;

        auto params = base;
        const int64_t kNd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == kNd) {
                // 切分轴即最内维（kNd==0）时按 Pad8 虚增段长（尾块粒度约束；
                // 统计量槽 srcStride=0 重复读安全；x 该情形已由 CopyInDensePad
                // 拦截，走到这里说明 aISeg%8==0，Pad8 为恒等）
                params.loopInfo.loopSize[nd] = static_cast<uint32_t>((kNd == 0) ? PadV(aISeg) : aISeg);
            }
            params.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(inner);
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};
        if constexpr (RANK <= kMaxNddmaDims) {
            AscendC::DataCopy<DT, ND, cfg>(buf_[bufIdx].template Get<DT>(), gm[off], params);
        } else {
            // RANK > 5: dims beyond the NDDMA 5-dim limit via software outer loop
            AscendC::LocalTensor<DT> buf = buf_[bufIdx].template Get<DT>();
            const int64_t elemBase = off;
            for (int64_t oi = 0; oi < nddmaOuterIters_[slot]; oi++) {
                int64_t elemAdj = 0;
                int64_t tmp = oi;
                for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                    const int64_t sz = (d == k) ? aISeg : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[slot][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<DT, ND, cfg>(buf[oi * inner], gm[elemBase + elemAdj], params);
            }
        }
    }

    // -----------------------------------------------------------------------
    // CopyOutY — main output y, DataCopyPad valid-bytes write-back
    // (DESIGN-BRANCH-0.md §5.4)。countGm 为 GM 真实元素数；UB 为 Pad8 布局时
    // （最内维 L%8!=0 且切分轴非最内维）按行写回：每行 L 真实元素，UB 行距
    // Pad8(L)，GM 行背靠背（dstStride=0），pad 通道垃圾不写出。
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOutY(int64_t bufIdx, const int64_t* coord, int64_t countGm)
    {
        const int64_t off = CalcOffset(coord, td_->outputStrides[kOutY], RANK); // elements
        const int64_t last = td_->maxBroShape[RANK - 1];
        AscendC::DataCopyExtParams extParams;
        extParams.dstStride = 0;
        if (td_->split.axis == RANK - 1 || last % kPadElems == 0) {
            // UB 前 countGm 元素稠密有效（切分轴在最内维时 x 为整段连续；
            // 或 L 本就 8 元素对齐、Pad8 为恒等）
            extParams.blockCount = 1;
            extParams.blockLen = static_cast<uint32_t>(countGm * sizeof(T));
            extParams.srcStride = 0;
        } else {
            // Pad8 布局：按行写回真实元素（UB 行距 Pad8(L)，GM 行背靠背）。
            // 逐行 blockCount=1 DataCopyPad（与 CopyOutStats 同款、语义已验证），
            // 规避多块 srcStride 单位语义不确定性
            AscendC::LocalTensor<T> src = buf_[bufIdx].template Get<T>();
            const int64_t rows = countGm / last;
            extParams.blockCount = 1;
            extParams.blockLen = static_cast<uint32_t>(last * sizeof(T));
            extParams.srcStride = 0;
            for (int64_t r = 0; r < rows; r++) {
                AscendC::DataCopyPad(gmY_[off + r * last], src[r * PadV(last)], extParams);
            }
            return;
        }
        AscendC::DataCopyPad(gmY_[off], buf_[bufIdx].template Get<T>(), extParams);
    }

    // -----------------------------------------------------------------------
    // CopyOutStats — bypass outputs batch_mean (outSlot 1) / batch_variance
    // (outSlot 2): per covered (n,g) group write exactly 1 fp32 element
    // (Kernel.md §9.2; values identical across tiles of one group, so
    // multi-tile groups converge deterministically).
    //
    // Group set of a tile = covered-coordinate combinations over dims d ≥ k
    // with outputShapes[outSlot][d] > 1 (dims d < k are fixed per tile and
    // fold into gmBase; size-1 dims carry stride 0 and contribute nothing).
    // UB group-first offset uses the tile-local dense row-major layout
    // (ubStride[k] = innerCount, ubStride[d] = Π_{j>d} maxBroShape[j]).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOutStats(int64_t outSlot, int64_t bufIdx, const int64_t* coord, int64_t aISeg)
    {
        const int64_t* statsShape = td_->outputShapes[outSlot];
        const int64_t* statsStrides = td_->outputStrides[outSlot];
        const int64_t* bro = td_->maxBroShape;
        const int64_t k = td_->split.axis;

        int64_t gmBase = 0;
        for (int64_t d = 0; d < k; d++) {
            gmBase += coord[d] * statsStrides[d];
        }
        // UB 组偏移行主序 stride：最内维按 Pad8 计（与 CopyInBrc/CopyOutY 的
        // padded 布局一致；statsShape[last] 恒为 1，ubStride[last] 不被消费）
        int64_t ubStride[kMaxRank] = {};
        int64_t inner = 1;
        for (int64_t d = RANK - 1; d >= k; d--) {
            ubStride[d] = inner;
            inner *= (d == RANK - 1) ? PadV(bro[d]) : bro[d];
        }
        AscendC::LocalTensor<float> src = buf_[bufIdx].template Get<float>();
        AscendC::GlobalTensor<float>* gm = (outSlot == kOutBatchMean) ? &gmBatchMean_ : &gmBatchVar_;

        // 最内维为组维（M=1 被 squeeze、G 落最内折叠维，如 case00007）时，
        // 逐元素写的 UB 源地址 = base + idx*4B，违反 MTE3（DataCopyPad UB→GM）
        // UB 侧地址须 32B 对齐的硬件约束（errcode 80 实测）。此时沿最内维的
        // 组值在 UB/GM 两侧均为 stride=1 连续 run（ubStride[last]=1 恒成立；
        // statsStrides[last] 在 size>1 时恒为 1），合并为单次 DataCopyPad：
        // run 起点 ubOff 恒为 PadV 倍数（32B 对齐），blockLen=runLen*4。
        const bool lastIsGroup = (statsShape[RANK - 1] > 1);
        const int64_t runLen = lastIsGroup ? ((k == RANK - 1) ? aISeg : statsShape[RANK - 1]) : 1;

        int64_t idx[kMaxRank] = {};
        while (true) {
            int64_t gmOff = gmBase;
            int64_t ubOff = 0;
            for (int64_t d = k; d < RANK; d++) {
                if (d == RANK - 1 && lastIsGroup) {
                    // run 起点：splitLast 时 GM 侧含 coord[k] 基址，UB 侧从 0 起
                    if (d == k) {
                        gmOff += coord[k] * statsStrides[d];
                    }
                    continue;
                }
                if (statsShape[d] > 1) {
                    const int64_t c = (d == k) ? (coord[k] + idx[d]) : idx[d];
                    gmOff += c * statsStrides[d];
                    ubOff += idx[d] * ubStride[d];
                }
            }
            AscendC::DataCopyExtParams extParams;
            extParams.blockCount = 1;
            extParams.blockLen = static_cast<uint32_t>(runLen * sizeof(float));
            extParams.srcStride = 0;
            extParams.dstStride = 0;
            AscendC::DataCopyPad((*gm)[gmOff], src[ubOff], extParams);
            // odometer carry over group-varying dims d ≥ k（最内组维已并入 run，跳过）
            int64_t d = lastIsGroup ? (RANK - 2) : (RANK - 1);
            for (; d >= k; d--) {
                if (statsShape[d] <= 1) {
                    continue;
                }
                const int64_t len = (d == k) ? aISeg : bro[d];
                idx[d]++;
                if (idx[d] < len) {
                    break;
                }
                idx[d] = 0;
            }
            if (d < k) {
                break;
            }
        }
    }
};

// ===========================================================================
// VF implementations (fp32 main chain; RegBase __simd_vf__ + asc_vf_call,
// broadcast-standard-kernel-template.md §5.3).
// ===========================================================================

// VF1: mean = sum · invM —— 链长 1（Reg::Muls），原地写回 sum 槽（B2）
__simd_vf__ inline void MeanVF(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, float invM, uint32_t count,
                               uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> vReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining); // remaining 自动递减
        AscendC::Reg::LoadAlign(vReg, srcAddr, aReg);
        AscendC::Reg::Muls(vReg, vReg, invM, mask); // mean = sum * (1/M)
        AscendC::Reg::StoreAlign(dstAddr, vReg, aReg, mask);
    }
}

// VF2a: var = square_sum·invM − mean² —— 链长 3（Muls/Mul/Sub），
// 结果落 B0 供 batch_variance CopyOut（correction=0，负 var 不 clamp，对齐 A2）。
// inf 安全补丁（fp64 golden 语义对齐）：sq·invM=+inf 且 mean² 在 fp32 上溢为
// +inf 而 mean 有限时（mean=sum/M ≤ FLT_MAX → fp64 中 mean² ≤ 1.2e77 恒有限，
// golden var = +inf − 有限 = +inf），fp32 直算 inf−inf=NaN 偏离 golden，
// 修补为 +inf（batch_variance=+inf、rstd=0、y≈offset）。mean=±inf 时 golden
// var 同为 NaN（inf−inf），不在修补范围。
__simd_vf__ inline void VarVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr, float invM,
                              uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    const float kPosInf = __builtin_inff();
    AscendC::Reg::RegTensor<float> qReg, mReg, tReg, infReg;
    AscendC::Reg::MaskReg mask, mSqInf, mMsqInf, mMeanInf, mPatch;
    AscendC::Reg::AddrReg aReg;
    AscendC::Reg::Duplicate(infReg, kPosInf); // +inf 常量
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(qReg, src0Addr, aReg); // square_sum
        AscendC::Reg::LoadAlign(mReg, src1Addr, aReg); // mean
        AscendC::Reg::Muls(qReg, qReg, invM, mask);    // square_sum / M
        AscendC::Reg::Mul(tReg, mReg, mReg, mask);     // mean²
        // 修补掩码：mPatch = (sq/M == +inf) && (mean² == +inf) && (mean 有限)
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(mSqInf, qReg, kPosInf, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(mMsqInf, tReg, kPosInf, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(mMeanInf, mReg, kPosInf, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(mPatch, mReg, -kPosInf, mask);
        AscendC::Reg::Or(mMeanInf, mMeanInf, mPatch, mask); // mean == ±inf
        AscendC::Reg::Not(mMeanInf, mMeanInf, mask);        // mean 有限
        AscendC::Reg::And(mPatch, mSqInf, mMsqInf, mask);
        AscendC::Reg::And(mPatch, mPatch, mMeanInf, mask);
        AscendC::Reg::Sub(qReg, qReg, tReg, mask);        // var = sq/M − mean²
        AscendC::Reg::Select(qReg, infReg, qReg, mPatch); // 修补位 → +inf
        AscendC::Reg::StoreAlign(dstAddr, qReg, aReg, mask);
    }
}

// VF2b: rstd = 1/sqrt(var + ε) —— 链长 3（Adds/Sqrt/Div），原地覆写 var 槽（B0）；
// ε>0 保证被开方量恒正（除零防护，spec.yaml numerical_stability.epsilon_guard）
__simd_vf__ inline void RstdVF(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, float epsilon, uint32_t count,
                               uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> vReg, oneReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    AscendC::Reg::Duplicate(oneReg, 1.0f); // 被除数 1.0f
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(vReg, srcAddr, aReg);  // var
        AscendC::Reg::Adds(vReg, vReg, epsilon, mask); // var + ε
        AscendC::Reg::Sqrt(vReg, vReg, mask);          // sqrt(var + ε)
        AscendC::Reg::Div(vReg, oneReg, vReg, mask);   // rstd = 1/sqrt(var+ε)
        AscendC::Reg::StoreAlign(dstAddr, vReg, aReg, mask);
    }
}

// VF3a: xm = xf − mean —— 链长 1（Sub），原地覆写 xf 槽（B1）；仿射路径专用
__simd_vf__ inline void SubMeanVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr,
                                  uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> xReg, mReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(xReg, src0Addr, aReg); // xf
        AscendC::Reg::LoadAlign(mReg, src1Addr, aReg); // mean
        AscendC::Reg::Sub(xReg, xReg, mReg, mask);     // xm = xf − mean
        AscendC::Reg::StoreAlign(dstAddr, xReg, aReg, mask);
    }
}

// VF3b: 自适应乘序 —— 链长 4（CompareScalar/Select/Select/Mul），仿射路径专用。
//   cond = (rstd > 1) 逐元素判定（组内恒定，混合组 tile 安全）：
//   cond  → C 序 g=scale, h=rstd（rstd>1 时 A 序中间量 xm·rstd 可能上溢而
//           真值有界，如 L1_066 x=-FLT_MAX；C 序若 xm·scale 上溢则真值
//           |xm·scale|·rstd > FLT_MAX，golden cast 同为 ±inf，安全）
//   !cond → A 序 g=rstd, h=scale（rstd≤1 时 xm·rstd 必不上溢，安全；
//           反向固定 B 序 rstd·scale 会在 scale=±FLT_MAX 级时先溢，如 L1_336）
//   rstd=NaN（var+eps<0）时 cond=false 走 A 序，NaN 传播与 golden 一致。
// 输出：t = xm·g → dstAddr（B1，原地覆写 xm 槽）；h → dstHAddr（B0，覆写 rstd 槽）
__simd_vf__ inline void AdaptiveMulVF(__ubuf__ float* dstAddr, __ubuf__ float* dstHAddr, __ubuf__ float* srcXmAddr,
                                      __ubuf__ float* srcRstdAddr, __ubuf__ float* srcScaleAddr, uint32_t count,
                                      uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> xReg, rReg, sReg, gReg, hReg;
    AscendC::Reg::MaskReg mask, mCond;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(xReg, srcXmAddr, aReg);    // xm
        AscendC::Reg::LoadAlign(rReg, srcRstdAddr, aReg);  // rstd
        AscendC::Reg::LoadAlign(sReg, srcScaleAddr, aReg); // scale
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(mCond, rReg, 1.0f, mask);
        AscendC::Reg::Select(gReg, sReg, rReg, mCond); // g = rstd>1 ? scale : rstd
        AscendC::Reg::Select(hReg, rReg, sReg, mCond); // h = rstd>1 ? rstd : scale
        AscendC::Reg::Mul(xReg, xReg, gReg, mask);     // t = xm·g
        AscendC::Reg::StoreAlign(dstAddr, xReg, aReg, mask);
        AscendC::Reg::StoreAlign(dstHAddr, hReg, aReg, mask);
    }
}

// VF3: ŷ = (xf − mean) · rstd —— 链长 2（Sub/Mul），原地覆写 xf 槽（B1）；
// 仅 hasAffine=0 纯归一化路径使用（无 scale 吸收溢出，fp32 中间 ±inf 与
// fp64 golden cast 回 fp32 的 ±inf 一致）
__simd_vf__ inline void NormalizeVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr,
                                    __ubuf__ float* src2Addr, uint32_t count, uint32_t oneRepeatSize,
                                    uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> xReg, mReg, rReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(xReg, src0Addr, aReg); // xf
        AscendC::Reg::LoadAlign(mReg, src1Addr, aReg); // mean
        AscendC::Reg::LoadAlign(rReg, src2Addr, aReg); // rstd
        AscendC::Reg::Sub(xReg, xReg, mReg, mask);     // xf − mean
        AscendC::Reg::Mul(xReg, xReg, rReg, mask);     // ŷ = (xf−mean)·rstd
        AscendC::Reg::StoreAlign(dstAddr, xReg, aReg, mask);
    }
}

// VF4: y = t·h + offset —— 链长 1（Reg::MulAddDst 硬件融合），结果写回 B1；
// dstReg 载入 offset，t 从 dstAddr 读入：MulAddDst(off, h, t) = h·t + off
// （src0Addr = h 槽 B0，src1Addr = offset 槽 B2；自适应乘序见 VF3b 注释）
__simd_vf__ inline void AffineVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, __ubuf__ float* src1Addr,
                                 uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> yReg, sReg, oReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(oReg, src1Addr, aReg);   // offset → dstReg
        AscendC::Reg::LoadAlign(yReg, dstAddr, aReg);    // ŷ
        AscendC::Reg::LoadAlign(sReg, src0Addr, aReg);   // scale
        AscendC::Reg::MulAddDst(oReg, sReg, yReg, mask); // y = scale·ŷ + offset
        AscendC::Reg::StoreAlign(dstAddr, oReg, aReg, mask);
    }
}

// VF4': y = t·h（hasAffine=1 且 hasOffset=0；golden: offset 缺失按恒等 0），
// 结果写回 B1；t 从 dstAddr 读入，src0Addr = h 槽 B0
__simd_vf__ inline void AffineNoOffsetVF(__ubuf__ float* dstAddr, __ubuf__ float* src0Addr, uint32_t count,
                                         uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> yReg, sReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(yReg, dstAddr, aReg);  // ŷ
        AscendC::Reg::LoadAlign(sReg, src0Addr, aReg); // scale
        AscendC::Reg::Mul(yReg, yReg, sReg, mask);     // y = ŷ·scale
        AscendC::Reg::StoreAlign(dstAddr, yReg, aReg, mask);
    }
}
